# RQ4 treatment run (official harness, prompt-only addendum)

Kaggle notebook nawazs4265/getting-started-gemma-4-developer-agent, Version 2 (kernel session 355082808),
GPU L4 x4, 6477 s total, agent phase 1.61 h (6.0 min/task). Same 16 tasks and config as the pilot
(Version 1); only change: FL_PROMPT_ADDENDUM (1,001 chars) appended to prompts/system.md.

Outputs on Kaggle: fl_treatment/labeled.jsonl, fl_treatment/fl_task_results.jsonl, fl_treatment_results.zip.

| task | resolved | auto (v0.2) | adjudicated | submitted | turns | edits (fail) | invalid turns | patch files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rich_3454 | 1 | RESOLVED | RESOLVED | yes | 42 | 14 (1 + 4 rejected) | 4 rejected | rich/highlighter.py |
| httpx_3672 | 0 | BROKEN_PATCH | WRONG_FIX (hidden test needs httpx.Stream) | yes | 34 | 7 (0) | 0 | 4 src files |
| fastapi_14077 | 1 | RESOLVED | RESOLVED | yes | 19 | 7 (0) | 0 | exception_handlers.py, openapi/utils.py, docs_src, pyproject.toml |
| requests_6589 | 0 | ENV_UNVERIFIABLE | ENV_UNVERIFIABLE (exit 124) | yes | 33 | 1 (0) | 2 repaired | utils.py |
| rich_3518 | 1 | RESOLVED | RESOLVED | yes | 21 | 1 (0) | 2 repaired | rich/table.py |
| fastapi_14262 | 0 | BUDGET | BUDGET (50 turns) | no | 50 | 9 (1) | 1 repaired | 3 src files |
| requests_6757 | 0 | ENV_UNVERIFIABLE | ENV_UNVERIFIABLE (exit 124; patch only tox.ini) | yes | 13 | 2 (0) | 0 | tox.ini |
| rich_3675 | 0 | WRONG_FIX | WRONG_FIX | yes | 25 | 1 (0) | 2 repaired | repro.py, rich/console.py |
| fastapi_14303 | 0 | ENV_UNVERIFIABLE | ENV_UNVERIFIABLE (collection) | yes | 16 | 1 (0) | 0 | dependencies/utils.py |
| requests_7315 | 0 | WRONG_FIX | WRONG_FIX (rewrote existing test; hidden test patch failed to apply) | yes | 33 | 9 (7 rejected, identical x7) | 7 rejected | adapters.py, tests/test_adapters.py |
| rich_3930 | 0 | BUDGET | BUDGET (20 min) | no | 44 | 10 (2) | 1 rejected (write_file w/o filepath) | repro + 2 src |
| fastapi_14361 | 0 | ENV_UNVERIFIABLE | ENV_UNVERIFIABLE (collection) | yes | 18 | 2 (0) | 0 | _compat/v2.py |
| requests_7433 | 0 | BUDGET | BUDGET (50 turns) | no | 50 | 2 (0) | 0 | 2 scratch + models.py |
| rich_4079 | 0 | BUDGET | BUDGET (50 turns; scratch only) | no | 50 | 0 | 0 | 4 scratch scripts |
| fastapi_14485 | 0 | BUDGET | BUDGET (50 turns) | no | 50 | 2 (1) | 2 repaired, 1 no-call | utils.py + 8 scratch |
| requests_7502 | 0 | ENV_UNVERIFIABLE | ENV_UNVERIFIABLE (exit 124) | yes | 25 | 3 (0) | 0 | models.py |

Paired vs pilot (16 tasks): resolved 2 -> 3 (only-after: fastapi_14077; McNemar exact p = 1.0);
submitted 4 -> 11 (8 gained, fastapi_14485 lost; p = 0.039); edit_file failed 35/63 -> 16/71 (5 runtime + 11 rejected);
failed over-escaped edits 32 -> 0; max identical repeat 42 -> 7; schema-invalid turns 2/520 -> 21/523
(toolguard replay: 9 repaired, 12 rejected); 11.4 -> 6.0 min/task.

New defect: corrupted argument keys (e.g. `start_line"`, or code text containing `"` parsed as a key so
old_string is missing). Raw generations are not in the traces, so model vs parser cannot be separated.
