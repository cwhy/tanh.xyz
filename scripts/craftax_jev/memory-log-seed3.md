# Memory log — lessons-seed3
190 steps · 4 achievements (collect_drink, collect_sapling, defeat_zombie, place_plant) · return 3.1 · seed 3 · jev-latest

**Admission:** 189 transitions judged · 109 labelled reinforce or avoid · 97 rejected as already covered · **12 kept**

**Coverage score:** min 0.12 · median 0.61 · max 0.91 · 55% at or above the 0.5 bar

## Every admission decision

`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected

| step | call | coverage | outcome | transition |
|---|---|---|---|---|
| 9 | - avoid | 0.22 | **KEPT** | `do -> nothing you can see changed` |
| 12 | - avoid | 0.60 | *covered* | `up -> nothing you can see changed` |
| 27 | - avoid | 0.90 | *covered* | `down -> nothing you can see changed` |
| 28 | + reinforce | 0.24 | **KEPT** | `do -> drink 8->9, achievement collect_drink` |
| 29 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 30 | - avoid | 0.75 | *covered* | `down -> nothing you can see changed` |
| 49 | + reinforce | 0.82 | *covered* | `down -> came into view: (28,30) grass, (29,30) grass, (30,30) grass, (31,30) grass, (32,30) grass, (33,30) grass, (34,30) cow, (35,30) grass, (36,30) …` |
| 50 | + reinforce | 0.26 | **KEPT** | `do -> (33,30) grass->cow, (34,30) cow->grass` |
| 51 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 53 | + reinforce | 0.38 | **KEPT** | `do -> (33,29) grass->cow, (33,30) cow->grass` |
| 54 | + reinforce | 0.66 | *covered* | `do -> (33,29) cow->grass, (33,30) grass->cow, sapling 0->1, achievement collect_sapling` |
| 62 | - avoid | 0.16 | **KEPT** | `do -> health 9->7, energy 8->7` |
| 63 | - avoid | 0.90 | *covered* | `do -> nothing you can see changed` |
| 65 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 66 | + reinforce | 0.27 | **KEPT** | `do -> (32,28) zombie->grass, (32,29) grass->cow, (32,30) cow->grass, achievement defeat_zombie` |
| 68 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 69 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 71 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 73 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 75 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 77 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 78 | + reinforce | 0.23 | **KEPT** | `do -> (36,29) grass->cow, health 7->8, food 7->6` |
| 80 | - avoid | 0.58 | *covered* | `do -> nothing you can see changed` |
| 81 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 82 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 83 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 85 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 86 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 87 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 90 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 92 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 93 | - avoid | 0.20 | **KEPT** | `do -> energy 7->6` |
| 94 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 95 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 96 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 97 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 98 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 99 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 100 | + reinforce | 0.87 | *covered* | `do -> (33,30) grass->cow` |
| 101 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 103 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 104 | + reinforce | 0.59 | *covered* | `do -> health 8->9, food 6->5` |
| 105 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 106 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 107 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 108 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 109 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 110 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 111 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 112 | + reinforce | 0.43 | **KEPT** | `do -> drink 6->5` |
| 113 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 114 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 115 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 116 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 117 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 118 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 119 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 120 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 121 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 122 | - avoid | 0.88 | *covered* | `do -> nothing you can see changed` |
| 123 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 124 | - avoid | 0.53 | *covered* | `do -> energy 6->5` |
| 125 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 126 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 127 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 128 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 129 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 131 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 132 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 133 | + reinforce | 0.82 | *covered* | `do -> drink 5->4` |
| 134 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 135 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 136 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 137 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 138 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 139 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 140 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 141 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 142 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 143 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 144 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 145 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 146 | - avoid | 0.41 | **KEPT** | `down -> nothing you can see changed` |
| 147 | - avoid | 0.90 | *covered* | `do -> nothing you can see changed` |
| 148 | - avoid | 0.90 | *covered* | `do -> nothing you can see changed` |
| 149 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 150 | - avoid | 0.91 | *covered* | `do -> nothing you can see changed` |
| 151 | - avoid | 0.91 | *covered* | `do -> nothing you can see changed` |
| 152 | - avoid | 0.91 | *covered* | `do -> nothing you can see changed` |
| 153 | - avoid | 0.91 | *covered* | `do -> nothing you can see changed` |
| 154 | + reinforce | 0.65 | *covered* | `do -> drink 4->3` |
| 157 | - avoid | 0.87 | *covered* | `do -> nothing you can see changed` |
| 158 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 159 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 160 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 166 | - avoid | 0.58 | *covered* | `do -> health 9->7` |
| 167 | - avoid | 0.88 | *covered* | `do -> nothing you can see changed` |
| 168 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 169 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 172 | - avoid | 0.28 | **KEPT** | `do -> (28,24) cow->grass, health 7->5` |
| 173 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 174 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 177 | - avoid | 0.90 | *covered* | `do -> nothing you can see changed` |
| 178 | - avoid | 0.61 | *covered* | `do -> health 5->3` |
| 180 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 183 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 184 | - avoid | 0.26 | **KEPT** | `do -> health 4->2` |
| 187 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 188 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |

## Final store — reinforce (6)

- `do -> drink 8->9, achievement collect_drink`
- `do -> (33,30) grass->cow, (34,30) cow->grass`
- `do -> (33,29) grass->cow, (33,30) cow->grass`
- `do -> (32,28) zombie->grass, (32,29) grass->cow, (32,30) cow->grass, achievement defeat_zombie`
- `do -> (36,29) grass->cow, health 7->8, food 7->6`
- `do -> drink 6->5`

## Final store — avoid (6)

- `do -> nothing you can see changed`
- `do -> health 9->7, energy 8->7`
- `do -> energy 7->6`
- `down -> nothing you can see changed`
- `do -> (28,24) cow->grass, health 7->5`
- `do -> health 4->2`

## Store growth

step 9 → 1, step 28 → 2, step 50 → 3, step 53 → 4, step 62 → 5, step 66 → 6, step 78 → 7, step 93 → 8, step 112 → 9, step 146 → 10, step 172 → 11, step 184 → 12
