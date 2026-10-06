# Environment check: reference-patch replay through the official verifier

Kaggle notebook nawazs4265/getting-started-gemma-4-developer-agent, Version 4 (CPU, pinned 2026-07-01 image), 53 min.
Each of the 129 dev tasks was verified twice with `swegemma.harness.verification.verify_task` in the
starter's subprocess sandbox (60 s command timeout): once with the reference ("gold") patch as the agent
patch, once with an empty patch. Gold runs that timed out were repeated with a 300 s timeout.
Code: `scripts/envcheck_gold_replay_cell.py`. Raw output on Kaggle: `fl_envcheck/gold_replay.jsonl`.

A task is **judgeable** when the gold patch resolves and the empty patch does not.

| Repo | Tasks | Gold resolves | Judgeable |
| --- | --- | --- | --- |
| Textualize/rich | 48 | 41 | 41 |
| fastapi/fastapi | 67 | 31 | 29 |
| psf/requests | 13 | 1 | 0 |
| encode/httpx | 1 | 0 | 0 |
| **All** | **129** | **73** | **70 (54%)** |

Not judgeable (59):
- Missing test dependency in the sandbox, gold and empty both fail at collection:
  - `inline_snapshot` (26): fastapi_11355, 13207, 14099, 14186, 14246, 14262, 14266, 14297, 14349, 14361, 14430, 14455, 14459, 14482, 14485, 14512, 14583, 14605, 14609, 14953, 14962, 14964, 14978, 15023, 15030, 15745
  - `dirty_equals` (8): fastapi_12942, 13713, 13786, 14303, 14356, 14360, 14371, 14791
- Test suite hangs, gold and empty, at 60 s and at 300 s (8): requests_6589, 6592, 6629, 6757, 7328, 7433, 7502, 7505
- Gold patch fails its own tests (12): requests_7205, 7309, 7315, 7427; rich_3064, 3130, 3296, 3469, 3782, 4070; fastapi_15763, 15785
- Gold patch errors at collection (1): httpx_3672
- Tests pass but the verifier does not count it resolved (1): rich_3486
- Empty patch also resolves, so the tests cannot detect a fix (3): fastapi_14077, fastapi_14851, requests_6644

A longer command timeout (300 s) fixed none of the timeouts.

Judgeable task ids (70):
fastapi_11194, fastapi_13537, fastapi_13920, fastapi_14258, fastapi_14301, fastapi_14306, fastapi_14372, fastapi_14419, fastapi_14448, fastapi_14458, fastapi_14463, fastapi_14479, fastapi_14487, fastapi_14492, fastapi_14616, fastapi_14786, fastapi_14794, fastapi_14873, fastapi_14986, fastapi_15280, fastapi_15588, fastapi_15589, fastapi_15661, fastapi_15800, fastapi_5077, fastapi_5624, fastapi_9425, fastapi_9555, fastapi_9753, rich_2725, rich_2943, rich_3006, rich_3043, rich_3052, rich_3061, rich_3063, rich_3067, rich_3105, rich_3180, rich_3278, rich_3454, rich_3468, rich_3470, rich_3471, rich_3472, rich_3480, rich_3506, rich_3518, rich_3521, rich_3535, rich_3675, rich_3676, rich_3718, rich_3772, rich_3777, rich_3882, rich_3894, rich_3905, rich_3930, rich_3934, rich_3935, rich_3938, rich_3942, rich_3944, rich_3953, rich_4006, rich_4075, rich_4076, rich_4077, rich_4079

## Effect on the pilot and RQ4 (same 16 tasks)

Only 5 of the 16 study tasks are judgeable: rich_3454, rich_3518, rich_3675, rich_3930, rich_4079.
On those 5, both the pilot and the RQ4 treatment resolved the same 2 (rich_3454, rich_3518).
The treatment's only extra resolution, fastapi_14077, is a task whose tests pass with an empty patch.
