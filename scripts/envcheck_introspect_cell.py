# FaultLines environment check, step 1: look inside the official harness (CPU only, no model).
import inspect, json, os, pkgutil, sys, textwrap, traceback
from pathlib import Path
import swegemma

OUT = Path("/kaggle/working/fl_envcheck"); OUT.mkdir(parents=True, exist_ok=True)
root = Path(swegemma.__file__).parent
mods = []
for m in pkgutil.walk_packages([str(root)], prefix="swegemma."):
    mods.append(m.name)
listing = []
for p in sorted(root.rglob("*.py")):
    listing.append(f"{p.relative_to(root)}\t{p.stat().st_size}")
(OUT / "modules.txt").write_text("\n".join(listing))
print(len(listing), "python files in swegemma")

# Copy the whole package source as plain text, one file per module (readable from the Output tab).
src_dir = OUT / "src"
for p in sorted(root.rglob("*.py")):
    dst = src_dir / p.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(p.read_text(errors="replace"))
# Also adk_submission (the agent side) if present.
try:
    import adk_submission
    aroot = Path(adk_submission.__file__).parent
    for p in sorted(aroot.rglob("*.py")):
        dst = OUT / "src_adk" / p.relative_to(aroot)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(p.read_text(errors="replace"))
except Exception as e:
    print("adk_submission:", e)

# One concatenated text file for easy searching.
with open(OUT / "swegemma_all.txt", "w") as f:
    for p in sorted(root.rglob("*.py")):
        f.write(f"\n\n######## FILE {p.relative_to(root)} ########\n")
        f.write(p.read_text(errors="replace"))

# Task fields and data layout.
info = {}
try:
    from swegemma.models import load_tasks
    DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
    tasks = load_tasks(DATA_DIR / "tasks.jsonl")
    t0 = tasks[0]
    info["n_tasks"] = len(tasks)
    info["task_type"] = type(t0).__name__
    info["task_fields"] = {k: type(v).__name__ for k, v in (vars(t0) if hasattr(t0, "__dict__") else {}).items()}
    if not info["task_fields"] and hasattr(type(t0), "model_fields"):
        info["task_fields"] = {k: type(getattr(t0, k)).__name__ for k in type(t0).model_fields}
    info["has_gold_patch"] = sum(1 for t in tasks if getattr(t, "patch", None))
    info["data_dir"] = sorted(os.listdir(DATA_DIR))
    info["snapshots_sample"] = sorted(os.listdir(DATA_DIR / "snapshots"))[:5]
    info["wheels_sample"] = sorted(os.listdir(DATA_DIR / "wheels"))[:5]
    info["repos"] = sorted({t.repo for t in tasks})
except Exception:
    info["error"] = traceback.format_exc()
(OUT / "info.json").write_text(json.dumps(info, indent=1, default=str))
print(json.dumps(info, indent=1, default=str)[:3000])

# Signatures of everything public in the evaluate / verification / sandbox / agent_runner modules.
import importlib
sig_lines = []
for name in mods:
    if not any(k in name for k in ("evaluate", "verif", "sandbox", "agent_runner", "harness", "config", "cli")):
        continue
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        sig_lines.append(f"## {name}: import failed {e}")
        continue
    sig_lines.append(f"## {name}")
    for k, v in vars(mod).items():
        if k.startswith("_") or getattr(v, "__module__", None) != name:
            continue
        try:
            if inspect.isclass(v):
                sig_lines.append(f"class {k}{inspect.signature(v) if callable(v) else ''}")
                for mk, mv in vars(v).items():
                    if callable(mv) and not mk.startswith("__"):
                        try:
                            sig_lines.append(f"    def {mk}{inspect.signature(mv)}")
                        except Exception:
                            sig_lines.append(f"    def {mk}(?)")
            elif callable(v):
                sig_lines.append(f"def {k}{inspect.signature(v)}")
        except Exception as e:
            sig_lines.append(f"{k}: {e}")
(OUT / "signatures.txt").write_text("\n".join(sig_lines))
print("\n".join(sig_lines)[:6000])
print("[FL envcheck] introspection written to", OUT)
