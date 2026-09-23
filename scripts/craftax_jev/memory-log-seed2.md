# Memory log — lessons-seed2
204 steps · 0 achievements (none) · return -0.9 · seed 2 · jev-latest

**Admission:** 203 transitions judged · 177 labelled reinforce or avoid · 164 rejected as already covered · **13 kept**

**Coverage score:** min 0.19 · median 0.71 · max 0.89 · 81% at or above the 0.5 bar

## Every admission decision

`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected

| step | call | coverage | outcome | transition |
|---|---|---|---|---|
| 1 | - avoid | 0.20 | **KEPT** | `do -> nothing you can see changed` |
| 3 | - avoid | 0.34 | **KEPT** | `up -> nothing you can see changed` |
| 5 | + reinforce | 0.35 | **KEPT** | `down -> came into view: (28,36) grass, (29,36) stone, (30,36) path, (31,36) stone, (32,36) grass, (33,36) path, (34,36) stone, (35,36) stone, (36,36) …` |
| 6 | + reinforce | 0.28 | **KEPT** | `down -> came into view: (28,37) stone, (29,37) stone, (30,37) stone, (31,37) stone, (32,37) stone, (33,37) path, (34,37) iron, (35,37) stone, (36,37) …` |
| 8 | + reinforce | 0.38 | **KEPT** | `down -> came into view: (28,39) stone, (29,39) stone, (30,39) stone, (31,39) stone, (32,39) stone, (33,39) path, (34,39) lava, (35,39) stone, (36,39) …` |
| 9 | - avoid | 0.30 | **KEPT** | `down -> nothing you can see changed` |
| 10 | - avoid | 0.59 | *covered* | `do -> nothing you can see changed` |
| 11 | + reinforce | 0.22 | **KEPT** | `right -> came into view: (37,33) grass, (37,34) grass, (37,35) stone, (37,36) stone, (37,37) stone, (37,38) stone, (37,39) stone, left view: (28,33) s…` |
| 12 | + reinforce | 0.55 | *covered* | `down -> came into view: (29,40) stone, (30,40) stone, (31,40) stone, (32,40) stone, (33,40) path, (34,40) lava, (35,40) stone, (36,40) stone, (37,40) …` |
| 13 | + reinforce | 0.55 | *covered* | `down -> came into view: (29,41) stone, (30,41) stone, (31,41) stone, (32,41) stone, (33,41) stone, (34,41) stone, (35,41) stone, (36,41) stone, (37,41…` |
| 14 | + reinforce | 0.45 | **KEPT** | `down -> came into view: (29,42) stone, (30,42) stone, (31,42) stone, (32,42) stone, (33,42) stone, (34,42) stone, (35,42) stone, (36,42) stone, (37,42…` |
| 15 | + reinforce | 0.40 | **KEPT** | `down -> came into view: (29,43) stone, (30,43) stone, (31,43) iron, (32,43) stone, (33,43) coal, (34,43) stone, (35,43) stone, (36,43) stone, (37,43) …` |
| 16 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 17 | - avoid | 0.89 | *covered* | `do -> nothing you can see changed` |
| 18 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 19 | - avoid | 0.86 | *covered* | `do -> nothing you can see changed` |
| 20 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 22 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 23 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 24 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 25 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 26 | + reinforce | 0.52 | *covered* | `down -> food 9->8` |
| 27 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 28 | - avoid | 0.79 | *covered* | `down -> nothing you can see changed` |
| 29 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 30 | - avoid | 0.64 | *covered* | `do -> nothing you can see changed` |
| 31 | - avoid | 0.64 | *covered* | `down -> energy 9->8` |
| 32 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 33 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 34 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 35 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 36 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 37 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 38 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 39 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 40 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 41 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 43 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 44 | - avoid | 0.64 | *covered* | `do -> nothing you can see changed` |
| 45 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 46 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 47 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 48 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 49 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 50 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 51 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 53 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 54 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 55 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 56 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 57 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 58 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 59 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 60 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 61 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 64 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 65 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 66 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 67 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 68 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 69 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 70 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 71 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 72 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 73 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 74 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 75 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 76 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 77 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 79 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 80 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 81 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 82 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 83 | - avoid | 0.66 | *covered* | `do -> nothing you can see changed` |
| 85 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 86 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 87 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 88 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 89 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 90 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 91 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 92 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 94 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 95 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 96 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 97 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 98 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 99 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 100 | - avoid | 0.66 | *covered* | `do -> nothing you can see changed` |
| 101 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 102 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 103 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 106 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 107 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 108 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 109 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 110 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 111 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 112 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 113 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 114 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 115 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 116 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 117 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 118 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 119 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 120 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 121 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 122 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 123 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 125 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 127 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 128 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 129 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 131 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 132 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 133 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 134 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 135 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 136 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 137 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 138 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 139 | - avoid | 0.61 | *covered* | `do -> nothing you can see changed` |
| 140 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 141 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 142 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 143 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 144 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 145 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 146 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 148 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 149 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 150 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 151 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 152 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 153 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 154 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 157 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 158 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 159 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 160 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 161 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 162 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 163 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 164 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 165 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 166 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 167 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 169 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 170 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 171 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 172 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 173 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 174 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 175 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 176 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 179 | + reinforce | 0.33 | **KEPT** | `do -> (33,38) zombie->path, (33,39) path->zombie` |
| 181 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 183 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 184 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 185 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 187 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 188 | - avoid | 0.66 | *covered* | `do -> nothing you can see changed` |
| 190 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 191 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 192 | - avoid | 0.30 | **KEPT** | `do -> health 6->4` |
| 193 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 194 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 195 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 196 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 197 | + reinforce | 0.35 | **KEPT** | `do -> (33,37) path->cow` |
| 198 | - avoid | 0.37 | **KEPT** | `do -> health 4->2` |
| 199 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 200 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 201 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 202 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 203 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |

## Final store — reinforce (8)

- `down -> came into view: (28,36) grass, (29,36) stone, (30,36) path, (31,36) stone, (32,36) grass, (33,36) path, (34,36) stone, (35,36) stone, (36,36) stone, left view: (28,29) water, (29,29) water, (30,29) water, (31,29) grass, (32,29) grass, (33,29) grass, (34,29) grass, (35,29) grass, (36,29) grass, you are now at (32,33)`
- `down -> came into view: (28,37) stone, (29,37) stone, (30,37) stone, (31,37) stone, (32,37) stone, (33,37) path, (34,37) iron, (35,37) stone, (36,37) stone, left view: (28,30) water, (29,30) water, (30,30) grass, (31,30) grass, (32,30) tree, (33,30) grass, (34,30) grass, (35,30) grass, (36,30) grass, you are now at (32,34)`
- `down -> came into view: (28,39) stone, (29,39) stone, (30,39) stone, (31,39) stone, (32,39) stone, (33,39) path, (34,39) lava, (35,39) stone, (36,39) stone, left view: (28,32) grass, (29,32) grass, (30,32) grass, (31,32) grass, (32,32) grass, (33,32) tree, (34,32) grass, (35,32) grass, (36,32) grass, you are now at (32,36)`
- `right -> came into view: (37,33) grass, (37,34) grass, (37,35) stone, (37,36) stone, (37,37) stone, (37,38) stone, (37,39) stone, left view: (28,33) sand, (28,34) sand, (28,35) grass, (28,36) grass, (28,37) stone, (28,38) lava, (28,39) stone, facing down->right, you are now at (33,36)`
- `down -> came into view: (29,42) stone, (30,42) stone, (31,42) stone, (32,42) stone, (33,42) stone, (34,42) stone, (35,42) stone, (36,42) stone, (37,42) stone, left view: (29,35) tree, (30,35) grass, (31,35) grass, (32,35) grass, (33,35) grass, (34,35) stone, (35,35) stone, (36,35) stone, (37,35) stone, you are now at (33,39)`
- `down -> came into view: (29,43) stone, (30,43) stone, (31,43) iron, (32,43) stone, (33,43) coal, (34,43) stone, (35,43) stone, (36,43) stone, (37,43) stone, left view: (29,36) stone, (30,36) path, (31,36) stone, (32,36) grass, (33,36) path, (34,36) stone, (35,36) stone, (36,36) stone, (37,36) stone, you are now at (33,40)`
- `do -> (33,38) zombie->path, (33,39) path->zombie`
- `do -> (33,37) path->cow`

## Final store — avoid (5)

- `do -> nothing you can see changed`
- `up -> nothing you can see changed`
- `down -> nothing you can see changed`
- `do -> health 6->4`
- `do -> health 4->2`

## Store growth

step 1 → 1, step 3 → 2, step 5 → 3, step 6 → 4, step 8 → 5, step 9 → 6, step 11 → 7, step 14 → 8, step 15 → 9, step 179 → 10, step 192 → 11, step 197 → 12, step 198 → 13
