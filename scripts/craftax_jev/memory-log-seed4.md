# Memory log — lessons-seed4
71 steps · 1 achievements (collect_sapling) · return 0.1 · seed 4 · jev-latest

**Admission:** 70 transitions judged · 26 labelled reinforce or avoid · 19 rejected as already covered · **7 kept**

**Coverage score:** min 0.10 · median 0.35 · max 0.90 · 33% at or above the 0.5 bar

## Every admission decision

`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected

| step | call | coverage | outcome | transition |
|---|---|---|---|---|
| 11 | + reinforce | 0.11 | **KEPT** | `do -> (31,25) cow->grass, (32,25) grass->cow, sapling 0->1, achievement collect_sapling` |
| 13 | - avoid | 0.38 | **KEPT** | `up -> nothing you can see changed` |
| 15 | - avoid | 0.60 | *covered* | `do -> nothing you can see changed` |
| 16 | - avoid | 0.45 | **KEPT** | `place_plant -> (32,23) cow->grass, (32,24) grass->cow` |
| 17 | + reinforce | 0.57 | *covered* | `do -> (32,24) cow->grass, (32,25) grass->cow` |
| 18 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 20 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 26 | + reinforce | 0.42 | **KEPT** | `do -> (34,24) grass->cow, (34,25) cow->grass, food 9->8` |
| 27 | - avoid | 0.47 | **KEPT** | `do -> nothing you can see changed` |
| 28 | - avoid | 0.90 | *covered* | `do -> nothing you can see changed` |
| 29 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 33 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 36 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 40 | + reinforce | 0.32 | **KEPT** | `do -> (35,22) cow->grass, (35,23) grass->cow, sapling 4->5` |
| 41 | + reinforce | 0.67 | *covered* | `do -> (35,23) cow->grass, (35,24) grass->cow, sapling 5->6` |
| 44 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 49 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 54 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 55 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 59 | - avoid | 0.17 | **KEPT** | `do -> health 7->5` |
| 60 | + reinforce | 0.75 | *covered* | `do -> (35,22) grass->cow, (35,23) cow->grass, sapling 7->8` |
| 61 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 65 | - avoid | 0.50 | *covered* | `do -> health 5->1` |
| 66 | - avoid | 0.88 | *covered* | `do -> nothing you can see changed` |
| 68 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 69 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |

## Final store — reinforce (3)

- `do -> (31,25) cow->grass, (32,25) grass->cow, sapling 0->1, achievement collect_sapling`
- `do -> (34,24) grass->cow, (34,25) cow->grass, food 9->8`
- `do -> (35,22) cow->grass, (35,23) grass->cow, sapling 4->5`

## Final store — avoid (4)

- `up -> nothing you can see changed`
- `place_plant -> (32,23) cow->grass, (32,24) grass->cow`
- `do -> nothing you can see changed`
- `do -> health 7->5`

## Store growth

step 11 → 1, step 13 → 2, step 16 → 3, step 26 → 4, step 27 → 5, step 40 → 6, step 59 → 7
