# Official-harness pilot run (2026-09-29)

Kaggle notebook: nawazs4265/getting-started-gemma-4-developer-agent (version 1, private), GPU L4 x4.
Harness: official `swegemma` Evaluator from the competition wheelhouse, subprocess sandbox, sample submission.
Model: gemma-4-31b-it-qat-w4a16-ct (vLLM, tp=4, gemma4 tool/reasoning parsers, thinking on).
Sampling (starter sampling.yaml): temperature 0.2, top_p 0.95, max_output_tokens 16384, thinking_budget 4096.
Budgets: max_time_minutes 20 (overridden), max_tool_calls 100 (overridden), max_turns 50 and command timeout 60 s (sample eval_config.yaml).
Sample: 5 tasks per repo, seed 0 (httpx has 1 dev task) -> 16 tasks. Session 11,594 s; agent phase 3.03 h; 11.4 min/task.

| task | resolved | test exit | turns | wall s | stop | auto v0.1 | manual | notes |
|---|---|---|---|---|---|---|---|---|
| rich_3454 | yes | 0 | 50 | 394 | turn cap | RESOLVED | RESOLVED | never called submit_patch; working tree scored; 6 scratch files in patch |
| httpx_3672 | no | 2 | 12 | 1203 | time cap | BUDGET | BUDGET | 896 completion tok/turn; incomplete (gold touches 7 files) |
| fastapi_14077 | no | 2 | 50 | 290 | turn cap | LOOP | INTERFACE | 25 over-escaped edits all failed, same call x27; final patch SyntaxError |
| requests_6589 | no | 124 | 50 | 487 | turn cap | LOOP | LOOP | same run_command x28; never edited library source |
| rich_3518 | yes | 0 | 11 | 168 | submitted | RESOLVED | RESOLVED | |
| fastapi_14262 | no | 2 | 50 | 786 | turn cap | BUDGET | BUDGET | partial API (Depends(scope=)); inline_snapshot missing |
| requests_6757 | no | 124 | 20 | 1262 | time cap | LOOP | INTERFACE | 7/8 edits over-escaped and failed; only tox.ini changed |
| rich_3675 | no | 1 | 36 | 1205 | time cap | BUDGET | BUDGET | never edited library source |
| fastapi_14303 | no | 2 | 8 | 96 | submitted | NO_PATCH | ENV_UNVERIFIABLE | dirty_equals missing; source patch on gold file |
| requests_7315 | no | 1 | 50 | 385 | turn cap | LOOP | LOOP | same check x18; wrong fix ('//v:h') |
| rich_3930 | no | -1 | 24 | 1203 | time cap | BUDGET | BUDGET | empty patch; gold regenerates 26 unicode tables |
| fastapi_14361 | no | 2 | 18 | 1207 | time cap | BUDGET | BUDGET | only scratch files; inline_snapshot missing |
| requests_7433 | no | 124 | 50 | 278 | turn cap | LOOP | LOOP | `grep ... \| grep -v "..."` x42 (always empty, exit 1) |
| rich_4079 | no | 1 | 25 | 1206 | time cap | BUDGET | BUDGET | only repro.py |
| fastapi_14485 | no | 2 | 43 | 480 | submitted | NO_PATCH | ENV_UNVERIFIABLE | inline_snapshot missing; source patch on gold file |
| requests_7502 | no | 124 | 23 | 260 | submitted | NO_PATCH | ENV_UNVERIFIABLE | test run timed out; source patch on gold file |

Auto v0.1 vs manual: 11/16 agree, Cohen's kappa 0.60. Disagreements: argument-encoding runs read as LOOP (2);
adapter did not read patch text from result records, so 3 submitted runs read as NO_PATCH. Both fixed in v0.2.

Step level (520 turns): run_command 243, read_file 170, edit_file 63, write_file 31, code-graph tools 5, submit_patch 4, no call 4 (all closing summaries after submit).
Schema-invalid calls: 2 (edit_file, missing required + undeclared argument, one run). Raw `<|tool_call>` text in traces: 0; token ids: 0.
edit_file failures: 35/63. Over-escaped (literal \n, no real newline): 32 calls, 32 failed. Other edits: 3/31 failed.
Completion tokens per turn: time-capped runs 710 mean, turn-capped runs 166 mean.
