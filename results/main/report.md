# Run main

## Overall

| Model | Episodes | Success | 95% CI | Mean steps | Mean latency (s) | Tokens in / out |
|---|---|---|---|---|---|---|
| gemini-flash-lite | 35 | 94.3% | 81.4% - 98.4% | 5.1 | 5.9 | 199,208 / 8,131 |

## Success rate by category

| Category | gemini-flash-lite |
|---|---|
| aggregation | 4/4 |
| cleaning | 5/5 |
| filtering | 3/3 |
| groupby | 4/4 |
| join | 4/4 |
| multi_step | 4/4 |
| statistics | 4/4 |
| time_series | 4/4 |
| unanswerable | 1/3 |

## Outcomes (failure taxonomy)

| Outcome | gemini-flash-lite |
|---|---|
| correct | 33 |
| max_steps | 2 |

## Behaviour flags (share of episodes)

| Flag | gemini-flash-lite |
|---|---|
| tool_hallucination | 0% |
| code_error | 20% |
| repeated_call | 0% |
| no_evidence | 0% |
| text_only_turn | 0% |

## Failed episodes

| Model | Task | Outcome | Prediction | Truth |
|---|---|---|---|---|
| gemini-flash-lite | un_002 | max_steps | None | None |
| gemini-flash-lite | un_003 | max_steps | None | None |
