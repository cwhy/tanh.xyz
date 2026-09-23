# Memory log — lessons-seed1
82 steps · 2 achievements (collect_sapling, collect_wood) · return 1.1 · seed 1 · jev-latest

**Admission:** 81 transitions judged · 28 labelled reinforce or avoid · 22 rejected as already covered · **6 kept**

**Coverage score:** min 0.09 · median 0.19 · max 0.88 · 27% at or above the 0.5 bar

## Every admission decision

`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected

| step | call | coverage | outcome | transition |
|---|---|---|---|---|
| 5 | - avoid | 0.18 | **KEPT** | `up -> nothing you can see changed` |
| 18 | - avoid | 0.25 | **KEPT** | `down -> nothing you can see changed` |
| 19 | - avoid | 0.60 | *covered* | `noop -> nothing you can see changed` |
| 20 | + reinforce | 0.23 | **KEPT** | `do -> (32,41) tree->grass, wood 0->1, achievement collect_wood` |
| 22 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 23 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 24 | + reinforce | 0.16 | **KEPT** | `do -> sapling 0->1, achievement collect_sapling` |
| 25 | - avoid | 0.51 | *covered* | `do -> nothing you can see changed` |
| 27 | - avoid | 0.51 | *covered* | `do -> nothing you can see changed` |
| 28 | - avoid | 0.57 | *covered* | `do -> nothing you can see changed` |
| 32 | - avoid | 0.45 | **KEPT** | `do -> nothing you can see changed` |
| 33 | - avoid | 0.88 | *covered* | `do -> nothing you can see changed` |
| 34 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 37 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 40 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 43 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 50 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 53 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 59 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 60 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 61 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 65 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 66 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 67 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 69 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 70 | - avoid | 0.13 | **KEPT** | `do -> health 5->3` |
| 75 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 76 | - avoid | 0.61 | *covered* | `do -> health 3->1` |

## Final store — reinforce (2)

- `do -> (32,41) tree->grass, wood 0->1, achievement collect_wood`
- `do -> sapling 0->1, achievement collect_sapling`

## Final store — avoid (4)

- `up -> nothing you can see changed`
- `down -> nothing you can see changed`
- `do -> nothing you can see changed`
- `do -> health 5->3`

## Store growth

step 5 → 1, step 18 → 2, step 20 → 3, step 24 → 4, step 32 → 5, step 70 → 6
