"""faultlines command-line interface.

    faultlines check  "<model output>"            guard verdict for one turn
    faultlines ingest --results DIR [--tasks tasks.jsonl] --model NAME -o runs.jsonl
    faultlines shape  traces/trace_X.json            key/type tree (calibrate the adapter)
    faultlines label  runs.jsonl -o labeled.jsonl  auto-label runs
    faultlines summarize labeled.jsonl             RQ1/RQ2 tables
    faultlines replay labeled.jsonl                RQ3 on real turns
    faultlines agreement labeled.jsonl             kappa vs hand labels
    faultlines compare before.jsonl after.jsonl    RQ4 paired test
    faultlines stress [--from labeled.jsonl]       RQ3 stress set
    faultlines infer-schemas labeled.jsonl -o tools.json
    faultlines harness --data DIR --endpoint URL --model NAME --tokenizer PATH --out DIR [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import sys

from .atlas import label_run, read_runs, write_runs
from .atlas.report import agreement, compare, replay, summarize
from .stress import build_cases, evaluate, report, sample_calls
from .toolguard import ToolGuard, ToolRegistry, infer_registry, validate_call


def _registry(path: str | None) -> ToolRegistry:
    return ToolRegistry.from_file(path) if path else ToolRegistry.default_harness()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="faultlines")
    ap.add_argument("--tools", help="tool declarations JSON (default: built-in harness guess)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check"); p.add_argument("text", nargs="?"); p.add_argument("--file")
    p.add_argument("--lossy", action="store_true")
    p = sub.add_parser("ingest"); p.add_argument("--results", required=True); p.add_argument("--tasks")
    p.add_argument("--model", default=""); p.add_argument("--config-hash", default="")
    p.add_argument("--context-limit", type=int, default=32768); p.add_argument("-o", "--out", required=True)
    p.add_argument("--tokenizer", help="decode logged completion_token_ids to raw text (keeps special tokens)")
    p.add_argument("--harness", default="swegemma")
    p = sub.add_parser("shape", help="print the key/type tree of a JSON trace or first JSONL line")
    p.add_argument("path")
    p = sub.add_parser("label"); p.add_argument("runs"); p.add_argument("-o", "--out", required=True)
    for name in ("summarize", "replay", "agreement"):
        p = sub.add_parser(name); p.add_argument("runs")
    p = sub.add_parser("compare"); p.add_argument("before"); p.add_argument("after")
    p = sub.add_parser("stress"); p.add_argument("--from", dest="src"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lossy", action="store_true")
    p = sub.add_parser("infer-schemas"); p.add_argument("runs"); p.add_argument("-o", "--out", required=True)
    p = sub.add_parser("harness", help="run tasks with the mini-harness against a vLLM endpoint")
    p.add_argument("--data", required=True, help="competition dataset root (has tasks.jsonl, snapshots/)")
    p.add_argument("--endpoint", help="e.g. http://127.0.0.1:8000")
    p.add_argument("--model", help="served model name")
    p.add_argument("--tokenizer", help="path or HF id for the chat template")
    p.add_argument("--gold", action="store_true",
                   help="no model: replay each task's reference patch (environment check; should resolve ~100%%)")
    p.add_argument("--out", required=True)
    p.add_argument("--task-ids", nargs="*"); p.add_argument("--limit", type=int)
    p.add_argument("--mode", choices=["strict", "guard"], default="strict")
    p.add_argument("--max-tool-calls", type=int, default=50); p.add_argument("--time-minutes", type=float, default=30)
    p.add_argument("--max-output-tokens", type=int, default=16384); p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", choices=["on", "off", "default"], default="default")
    p.add_argument("--system-prompt", help="file with the system instruction (e.g. sample_submission prompt)")
    p.add_argument("--python", default="python3", help="interpreter for task venvs (e.g. a Python 3.12)")
    p.add_argument("--online-fallback", action="store_true", help="allow online pip if offline install fails")
    p.add_argument("--tool-style", choices=["gemma_inline", "tool_role"], default="gemma_inline")
    p.add_argument("--label", default="")

    a = ap.parse_args(argv)
    reg = _registry(a.tools)

    if a.cmd == "check":
        text = open(a.file).read() if a.file else (a.text if a.text is not None else sys.stdin.read())
        print(json.dumps(ToolGuard(reg, allow_lossy=a.lossy).check(text).to_dict(), indent=2))
    elif a.cmd == "ingest":
        from .atlas.adapters import load_swegemma_results
        decode = None
        if a.tokenizer:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
        n = write_runs(load_swegemma_results(a.results, a.tasks, a.model, a.config_hash, a.context_limit,
                                             decode, a.harness), a.out)
        print(f"wrote {n} runs to {a.out}")
    elif a.cmd == "shape":
        from .atlas.adapters import trace_shape
        text = open(a.path).read()
        obj = json.loads(text.splitlines()[0]) if a.path.endswith(".jsonl") else json.loads(text)
        print(trace_shape(obj))
    elif a.cmd == "label":
        n = write_runs((label_run(r, reg) for r in read_runs(a.runs)), a.out)
        print(f"labeled {n} runs -> {a.out}")
    elif a.cmd == "summarize":
        print(summarize(list(read_runs(a.runs))))
    elif a.cmd == "replay":
        print(replay(list(read_runs(a.runs))))
    elif a.cmd == "agreement":
        print(agreement(list(read_runs(a.runs))))
    elif a.cmd == "compare":
        print(compare(list(read_runs(a.before)), list(read_runs(a.after)), "before", "after"))
    elif a.cmd == "stress":
        if a.src:
            calls = [(s.tool_name, s.tool_args or {}) for r in read_runs(a.src) if not r.synthetic
                     for s in r.steps if s.tool_name and not validate_call(s.tool_name, s.tool_args or {}, reg)]
        else:
            calls = sample_calls()
            print("> using built-in sample calls; pass --from labeled.jsonl for real ones\n")
        print(report(evaluate(build_cases(calls, reg, a.seed), reg, a.lossy)))
    elif a.cmd == "infer-schemas":
        calls = [(s.tool_name, s.tool_args or {}) for r in read_runs(a.runs)
                 for s in r.steps if s.tool_name and not (s.runtime_error or "").startswith("TOOL_ERROR")]
        infer_registry(calls).dump(a.out)
        print(f"wrote inferred declarations for {len({c[0] for c in calls})} tools to {a.out}")
    elif a.cmd == "harness":
        return _harness(a)
    return 0


def _gold_model(task):
    from .harness import ScriptedModel
    from .toolguard import render_call
    cmd = f"cd /workspace && git apply <<'FL_GOLD_EOF'\n{task.patch}\nFL_GOLD_EOF"
    return ScriptedModel([render_call("run_command", {"command": cmd}), render_call("submit_patch", {})],
                         name="gold-replay")


def _harness(a) -> int:
    import time
    from pathlib import Path
    from .harness import CompletionEndpointModel, DataLayout, Limits, RunConfig, load_tasks, run_task
    from .harness.prompt import DEFAULT_SYSTEM
    layout = DataLayout.find(a.data)
    tasks = load_tasks(layout.tasks, a.task_ids, a.limit)
    kwargs = {} if a.thinking == "default" else {"enable_thinking": a.thinking == "on"}
    if a.gold:
        model = None
    else:
        if not (a.endpoint and a.model and a.tokenizer):
            raise SystemExit("--endpoint, --model and --tokenizer are required unless --gold is set")
        model = CompletionEndpointModel(a.endpoint, a.model, a.tokenizer, a.max_output_tokens, a.max_model_len,
                                        a.temperature, seed=a.seed, chat_template_kwargs=kwargs,
                                        tool_message_style=a.tool_style)
    cfg = RunConfig(limits=Limits(time_minutes=a.time_minutes, tool_calls=a.max_tool_calls), mode=a.mode,
                    system_prompt=Path(a.system_prompt).read_text() if a.system_prompt else DEFAULT_SYSTEM,
                    python=a.python, allow_online_fallback=a.online_fallback, context_limit=a.max_model_len,
                    seed=a.seed, label=a.label)
    out = Path(a.out); (out / "patches").mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks | mode={a.mode} | config={cfg.hash()} | out={out}", flush=True)
    for i, t in enumerate(tasks, 1):
        t0 = time.time()
        m = model if model is not None else _gold_model(t)
        r = run_task(t, layout, m, cfg)
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(r.to_dict()) + "\n")
        (out / "patches" / f"{t.instance_id}.patch").write_text(r.agent_patch or "")
        print(f"[{i}/{len(tasks)}] {t.instance_id}: resolved={r.resolved} exit={r.exit_reason} turns={len(r.steps)} "
              f"timing={r.timing} wall={time.time() - t0:.0f}s notes={r.notes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
