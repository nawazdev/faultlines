# FaultLines

**Where local Gemma 4 coding agents break, and how much of it is just the interface.**

Code for the paper *Fault Lines: Where Local Gemma 4 Coding Agents Break*, a Kaggle Gemma 4 Developer Agent **Paper Track** submission. It has two parts:

- **toolguard**: parses Gemma 4's native tool-call syntax (`<|tool_call>call:NAME{...}<tool_call|>`), classifies what is wrong, repairs only what is safe to repair, and detects loops. Pure Python, no dependencies.
- **atlas**: a trace schema, a failure taxonomy, an auto-labeler and the statistics the paper needs (Wilson CIs, Cohen's kappa, exact McNemar).

Status: **v0.2.** Tested on real traces from the official competition harness: a 16-task pilot of Gemma 4 31B and a paired prompt-only before/after run (RQ4). Results are summarized in `results/`. 30 tests pass.

## Install

```bash
pip install -e ".[dev]"
python -m pytest -q          # 30 tests (5 run a real end-to-end task)
```

## Commands

```bash
faultlines check '<|tool_call>call:bash{cmd:<|"|>ls<|"|>}<tool_call|>'   # one turn -> verdict JSON
faultlines ingest --results results/ --tasks tasks.jsonl --model gemma-4-31b -o runs.jsonl
faultlines label runs.jsonl -o labeled.jsonl      # step + run labels
faultlines summarize labeled.jsonl                # RQ1 / RQ2 tables
faultlines replay labeled.jsonl                   # RQ3 on real turns
faultlines agreement labeled.jsonl                # kappa vs your hand labels (set human_label)
faultlines compare before.jsonl after.jsonl       # RQ4 paired before/after
faultlines stress --from labeled.jsonl            # RQ3 stress set from real calls
faultlines infer-schemas labeled.jsonl -o tools.json   # tool declarations from observed calls
```

To try it without any real data, label the synthetic examples:

```bash
faultlines label examples/traces/synthetic_runs.jsonl -o demo.jsonl && faultlines summarize demo.jsonl
```

## Official harness run (primary path for the paper)

The official `swegemma` harness ships as wheels in the **Gemma 4 Developer Agent Wheelhouse** dataset, and the official starter notebook runs it with `gemma-4-31b-it-qat-w4a16-ct` on GPU L4 x4. `notebooks/faultlines_official_addon.py` is a single cell to paste at the end of a copy of that starter. It:

- reuses the starter's own `EvalConfig`, `Evaluator`, `run_sync` and vLLM server (found by type, not by variable name);
- raises the per-task budget, redirects output to `/kaggle/working/fl_official`, and prints every field it changed;
- runs a stratified, seeded sample (5 tasks per repo, round-robin), resumes if re-run, and stops starting new tasks near the 12 h session limit;
- ingests the ATIF v1.7 traces (decoding `completion_token_ids` to raw text when logged), labels every run, prints the RQ1/RQ2/RQ3 tables, and zips everything.

Rebuild it after changing the package: `python scripts/build_official_addon.py`.

## Mini-harness (fallback, for GPUs that can't run the 31B model)

`faultlines.harness` re-implements the harness from the published `HARNESS_README.md`:

- the 9 tools, with the exact signatures and JSON replies;
- the 7-section task prompt and the 3 exact nudge messages;
- the budgets and output limits;
- offline editable installs from `wheels/`;
- phase-2 verification (apply the patch with fallbacks, reset the test files, apply `test_patch`, run pytest with the documented flags).

On top of that, it records the model's **raw text** for every turn. It renders the Gemma 4 chat template locally and calls vLLM's `/v1/completions` with special tokens kept.

```bash
faultlines harness --data /path/to/competition_data --gold --limit 5 --out runs/gold           # environment check, no model
faultlines harness --data ... --endpoint http://127.0.0.1:8000 --model gemma --tokenizer MODEL_DIR \
                   --limit 5 --mode strict --out runs/e4b_strict                                  # strict = like the official parser
faultlines harness ... --mode guard --out runs/e4b_guard                                          # with toolguard repairs
```

`notebooks/faultlines_run.ipynb` does all of this on Kaggle: it starts vLLM, runs gold replay, runs N tasks, then reports timing and labels. Build it with `python scripts/build_run_notebook.py`.

**Known differences from the official harness** (state them in the paper):
- no Docker, and no 4 GiB / 2 vCPU limits;
- compaction is approximated (history is truncated past a token threshold) rather than ADK's summarising compaction;
- the "Standard Instructions" text is paraphrased;
- the harness's generated `pytest.ini`/`conftest.py` are replaced by the documented command-line flags.

## Layout

| Path | What it does |
| --- | --- |
| `src/faultlines/toolguard/gemma_syntax.py` | Extracts, parses and renders Gemma 4 tool-call syntax. Detects unclosed calls. |
| `src/faultlines/toolguard/validate.py` | Call error codes (`TRUNCATED_CALL`, `UNKNOWN_TOOL`, `MISSING_ARG`, …) |
| `src/faultlines/toolguard/repair.py` | Safe repairs (aliases, JSON arguments, type coercion, a missing close tag) and error feedback text |
| `src/faultlines/toolguard/loops.py` | Detects repeated calls and A-B-A-B cycles |
| `src/faultlines/toolguard/harness_tools.json` | The 9 harness tools, with the exact published signatures |
| `src/faultlines/atlas/taxonomy.py` | Primary labels, groups, definitions, secondary flags |
| `src/faultlines/atlas/autolabel.py` | The precedence rule and named thresholds |
| `src/faultlines/atlas/adapters.py` | Reads official-harness output (ATIF v1.7 traces + `task_results.jsonl`) into `Run` records |
| `src/faultlines/atlas/report.py` | Tables and statistics |
| `src/faultlines/stress.py` | Stress set with 10 corruption types |
| `src/faultlines/harness/` | Mini-harness: sandbox, 9 tools, prompt and nudges, agent loop, verification, vLLM client |
| `notebooks/faultlines_official_addon.py` | Paste-in cell for the official starter: 20 official-harness runs + labels |
| `notebooks/faultlines_run.ipynb` | Mini-harness Kaggle notebook: gold check, vLLM, N tasks, timing, labels |

## Design rules for repair

1. **Never invent content.** A call cut off inside a string, such as half of a `write_file` body, is refused, never completed.
2. **Rename only through an alias table, or a spelling match of 0.8 or higher**, and only to a name that exists and isn't already used.
3. **Dropping an argument loses information**, so it is off by default (`--lossy` turns it on). Report results both ways.
4. **Every repair is recorded by name**, so the paper can say exactly what was changed.

## Results so far

| File | What it holds |
| --- | --- |
| `results/official_run_2026-09-29.md` | Pilot: 16 dev tasks, official harness, Gemma 4 31B (QAT W4A16). Per-run auto and hand labels. |
| `results/official_run_labels.json` | The pilot's auto and adjudicated labels as JSON |
| `results/rq4_treatment_2026-10-04.md` | RQ4: the same 16 tasks with a prompt-only addendum, paired against the pilot |

Headline: tool-call syntax was almost always valid (2 of 520 turns), but argument encoding, exact loops and the 50-turn budget sank most runs. A prompt-only fix raised submissions from 4 to 11 of 16 (exact McNemar p = 0.04) but resolutions only from 2 to 3. Half of the pilot's verifications were confounded by the sandbox environment.

## Honest caveats

- **The stress set is circular.** It checks the rules that were written for these corruption types, so it scores 100% by construction. The paper's real evidence for repair is (a) replay on real traces and (b) a hand audit of 100 real repairs. Add corruption types taken from real failures you observe.
- **Auto-labels are heuristics.** Report kappa against hand labels, and publish the thresholds in `Thresholds`.
- **Synthetic example runs** are flagged `synthetic: true`. `summarize` warns when they are present. Never report them.

## License

Code: Apache-2.0 (see `LICENSE` and `NOTICE`). The trace dataset will be released separately on Kaggle Datasets under CC BY 4.0.
