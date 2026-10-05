"""Generate notebooks/faultlines_run.ipynb: runs Gemma 4 on N dev tasks with the mini-harness on Kaggle."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "faultlines_run.ipynb"

nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# FaultLines run: Gemma 4 on the competition's dev tasks

Runs a local Gemma 4 agent on N of the 129 dev tasks with **faultlines-mini**. It's a re-implementation of the (unreleased) `swegemma` harness, built from the competition's `HARNESS_README.md`: same 9 tools, prompt sections, nudges, limits and pytest verification. It also saves the model's **raw text** for every turn. Then it times the runs and labels them.

**Before you run (right-hand panel):**
1. **Settings → Accelerator:** GPU T4 x2 (or better). **Internet:** On.
2. **Add Input → Competitions:** `gemma-4-developer-agent`.
3. **Add Input → Models:** Gemma 4 → Transformers → `gemma-4-e4b-it`. It's the variant that fits a T4. The leaderboard model (31B W4A16) needs Ampere-class GPUs.

**Stages:**
1. **Gold replay (no GPU).** Applies each task's reference fix to check the environment and measure setup and verify time. It should resolve about 100%.
2. **Start vLLM and check the chat template.** Look at the printed output once.
3. **Run the model on N tasks,** then show timing, the projected runs per week, and the labels.

Outputs go to `/kaggle/working/fl_runs/`: `runs.jsonl`, `labeled.jsonl`, patches and the vLLM log.""")

md("## 0. Write the FaultLines package")
files = ["pyproject.toml"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*")
                                    if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in str(p))
dirs = sorted({str(Path(f).parent) for f in files if "/" in f})
code("!mkdir -p " + " ".join(f"faultlines/{d}" for d in dirs))
for f in files:
    code(f"%%writefile faultlines/{f}\n" + (ROOT / f).read_text())

md("## 1. Settings")
code('''N_TASKS = 5              # how many dev tasks to run
TASK_IDS = None          # or a list, e.g. ["fastapi_11194", "rich_3454"]
MODEL_HINT = "e4b"       # substring that picks the model folder under /kaggle/input
MODE = "strict"          # strict = like the official harness; guard = with toolguard repairs
TIME_MINUTES = 30        # per-task agent time budget (official default is 60)
MAX_TOOL_CALLS = 50
MAX_MODEL_LEN = 32768    # lower to 16384 if vLLM runs out of memory
THINKING = "default"     # "on" | "off" | "default"
RUN_GOLD_CHECK = True
OUT = "/kaggle/working/fl_runs"''')

md("## 2. Find inputs and GPUs")
code('''import glob, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.abspath("faultlines/src"))
from faultlines.harness import DataLayout

layout = DataLayout.find("/kaggle/input")
print("dataset:", layout.root, "| tasks:", sum(1 for _ in open(layout.tasks)))
cfgs = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
        if MODEL_HINT in p.lower() and "snapshots" not in p]
assert cfgs, f"No model folder matching '{MODEL_HINT}' — add Gemma 4 as an input (see top cell)."
MODEL_DIR = os.path.dirname(sorted(cfgs, key=len)[0])
print("model:", MODEL_DIR)

gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip().splitlines()
print("GPUs:", gpus)
N_GPU = max(1, len(gpus))
DTYPE = "half" if any(x in " ".join(gpus) for x in ("T4", "P100", "V100")) else "bfloat16"
print("dtype:", DTYPE, "| tensor parallel:", N_GPU)''')

md("## 3. Install vLLM and Python 3.12\n\nThe task wheels target Python 3.12/3.13, so the per-task venvs use a Python 3.12 managed by `uv`.")
code('''%%bash
pip install -q -U vllm uv 2>&1 | tail -3
uv python install 3.12 2>&1 | tail -1''')
code('''PY312 = subprocess.run(["uv", "python", "find", "3.12"], capture_output=True, text=True).stdout.strip()
print("task python:", PY312)
FL = [sys.executable, "-m", "faultlines.cli"]
ENV = {**os.environ, "PYTHONPATH": os.path.abspath("faultlines/src")}

def fl(*args):
    """Run a faultlines command, streaming its output."""
    p = subprocess.Popen(FL + list(args), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    return p.wait()

task_args = (["--task-ids", *TASK_IDS] if TASK_IDS else ["--limit", str(N_TASKS)])''')

md("## 4. Gold replay: environment check (no GPU needed)\n\nEvery task should resolve. Any failure here is an environment problem, not a model problem. Fix those before trusting model results.")
code('''if RUN_GOLD_CHECK:
    t0 = time.time()
    fl("harness", "--data", str(layout.root), "--gold", *task_args, "--out", f"{OUT}/gold",
       "--python", PY312, "--online-fallback", "--time-minutes", "10")
    print(f"gold replay wall time: {time.time() - t0:.0f}s")''')

md("## 5. Start the vLLM server")
code('''log = open(f"/kaggle/working/vllm.log", "w")
srv = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL_DIR,
                        "--served-model-name", "gemma", "--dtype", DTYPE, "--tensor-parallel-size", str(N_GPU),
                        "--max-model-len", str(MAX_MODEL_LEN), "--gpu-memory-utilization", "0.90", "--port", "8000"],
                       stdout=log, stderr=subprocess.STDOUT)
t0 = time.time()
while True:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); break
    except Exception:
        if srv.poll() is not None:
            print(open("/kaggle/working/vllm.log").read()[-4000:]); raise RuntimeError("vLLM exited")
        if time.time() - t0 > 1800:
            raise TimeoutError("vLLM did not start in 30 min")
        time.sleep(10)
print(f"vLLM ready in {time.time() - t0:.0f}s")''')

md("## 6. Chat template and tool-call check\n\n**Look at this output once.** Check that tools are declared in the prompt, that the model's reply contains `<|tool_call>call:get_status{}<tool_call|>` (or similar), and that the tool result shows up in the second render.")
code('''from faultlines.harness.model import CompletionEndpointModel
from faultlines.toolguard import ToolGuard, ToolRegistry
kw = {} if THINKING == "default" else {"enable_thinking": THINKING == "on"}
m = CompletionEndpointModel("http://127.0.0.1:8000", "gemma", MODEL_DIR, max_model_len=MAX_MODEL_LEN,
                            chat_template_kwargs=kw)
tools = [s.to_declaration() for s in ToolRegistry.default_harness().specs.values()]
msgs = [{"role": "system", "content": "You fix bugs using tools."},
        {"role": "user", "content": "Call the get_status tool now."}]
print("---- prompt tail ----\\n", m.render(msgs, tools)[-1200:])
t = m.generate(msgs, tools)
print("---- raw reply ----\\n", repr(t.text[:1500]), "| finish:", t.finish_reason)
print("---- guard ----\\n", ToolGuard(ToolRegistry.default_harness()).check(t.text).to_dict())
msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                 "function": {"name": "get_status", "arguments": {}}}]},
                {"role": "tool", "name": "get_status", "content": json.dumps({"tool_calls_used": 0})}]
print("---- render with tool result ----\\n", m.render(msgs2, tools)[-900:])''')

md("## 7. Run the model on N tasks")
code('''t0 = time.time()
fl("harness", "--data", str(layout.root), *task_args, "--out", f"{OUT}/model",
   "--endpoint", "http://127.0.0.1:8000", "--model", "gemma", "--tokenizer", MODEL_DIR,
   "--mode", MODE, "--time-minutes", str(TIME_MINUTES), "--max-tool-calls", str(MAX_TOOL_CALLS),
   "--max-model-len", str(MAX_MODEL_LEN), "--thinking", THINKING, "--python", PY312, "--online-fallback",
   "--label", f"{os.path.basename(MODEL_DIR)}/{MODE}")
MODEL_WALL = time.time() - t0
print(f"model run wall time: {MODEL_WALL:.0f}s")''')

md("## 8. Timing and how many runs fit in your GPU quota")
code('''runs = [json.loads(l) for l in open(f"{OUT}/model/runs.jsonl")]
print(f"{'task':24s} {'resolved':>8s} {'exit':>12s} {'turns':>5s} {'setup_s':>8s} {'agent_s':>8s} {'verify_s':>8s}")
for r in runs:
    tm = r["timing"]
    print(f"{r['task_id']:24s} {str(r['resolved']):>8s} {r['exit_reason']:>12s} {len(r['steps']):5d} "
          f"{tm.get('setup_s', 0):8.0f} {tm.get('agent_s', 0):8.0f} {tm.get('verify_s', 0):8.0f}")
per_run = MODEL_WALL / max(1, len(runs))
print(f"\\nmean wall time per run: {per_run / 60:.1f} min")
for hours in (30, 60):
    print(f"runs that fit in {hours} GPU-hours: ~{int(hours * 3600 / per_run)}")''')

md("## 9. Label and summarize")
code('''fl("label", f"{OUT}/model/runs.jsonl", "-o", f"{OUT}/model/labeled.jsonl")
fl("summarize", f"{OUT}/model/labeled.jsonl")
fl("replay", f"{OUT}/model/labeled.jsonl")''')
code('''# Peek at the raw model text of the first run (what the paper's step labels are built from)
r = [json.loads(l) for l in open(f"{OUT}/model/labeled.jsonl")][0]
print(r["task_id"], r["primary_failure"], r["secondary"], r["evidence"])
for s in r["steps"][:6]:
    print(f"\\n--- turn {s['index']} | {s['call_status']} {s['call_issues']} | tool={s['tool_name']} | events={s['events']}")
    print(s["raw_output"][-800:])''')
code('''srv.terminate()
print("done — outputs in", OUT)''')

OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(C)} cells, {len(files)} package files)")
