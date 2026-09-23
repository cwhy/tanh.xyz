# Memory log — lessons-seed0
370 steps · 4 achievements (collect_drink, collect_sapling, defeat_zombie, place_plant) · return 3.1 · seed 0 · jev-latest

**Admission:** 369 transitions judged · 162 labelled reinforce or avoid · 139 rejected as already covered · **23 kept**

**Coverage score:** min 0.11 · median 0.49 · max 0.92 · 48% at or above the 0.5 bar

## Every admission decision

`+` reinforce · `-` avoid · **KEPT** stored · *covered* rejected

| step | call | coverage | outcome | transition |
|---|---|---|---|---|
| 5 | - avoid | 0.24 | **KEPT** | `up -> nothing you can see changed` |
| 11 | - avoid | 0.24 | **KEPT** | `down -> nothing you can see changed` |
| 12 | - avoid | 0.56 | *covered* | `noop -> nothing you can see changed` |
| 13 | + reinforce | 0.21 | **KEPT** | `left -> (30,35) cow->grass, (30,36) grass->cow, came into view: (27,30) water, (27,31) water, (27,32) water, (27,33) sand, (27,34) sand, (27,35) sand,…` |
| 14 | + reinforce | 0.74 | *covered* | `left -> (30,36) cow->grass, (31,36) grass->cow, came into view: (26,30) water, (26,31) water, (26,32) water, (26,33) water, (26,34) water, (26,35) san…` |
| 15 | + reinforce | 0.57 | *covered* | `left -> (31,36) cow->grass, (32,36) grass->cow, came into view: (25,30) water, (25,31) water, (25,32) water, (25,33) water, (25,34) water, (25,35) wat…` |
| 18 | - avoid | 0.49 | **KEPT** | `left -> nothing you can see changed` |
| 19 | - avoid | 0.82 | *covered* | `noop -> nothing you can see changed` |
| 20 | - avoid | 0.91 | *covered* | `left -> nothing you can see changed` |
| 27 | - avoid | 0.49 | **KEPT** | `noop -> nothing you can see changed` |
| 28 | - avoid | 0.92 | *covered* | `noop -> nothing you can see changed` |
| 30 | + reinforce | 0.40 | **KEPT** | `right -> (32,36) grass->cow, came into view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) grass, (38,34) grass, (38,35) grass, (38,36) grass, …` |
| 31 | + reinforce | 0.68 | *covered* | `right -> (32,36) cow->grass, came into view: (39,30) grass, (39,31) grass, (39,32) grass, (39,33) grass, (39,34) grass, (39,35) tree, (39,36) grass, l…` |
| 41 | - avoid | 0.88 | *covered* | `noop -> nothing you can see changed` |
| 54 | + reinforce | 0.74 | *covered* | `left -> came into view: (27,30) water, (27,31) water, (27,32) water, (27,33) sand, (27,34) sand, (27,35) sand, (27,36) grass, left view: (36,30) tree,…` |
| 58 | - avoid | 0.54 | *covered* | `left -> came into view: (23,30) water, (23,31) water, (23,32) water, (23,33) water, (23,34) water, (23,35) water, (23,36) water, left view: (32,30) gr…` |
| 64 | + reinforce | 0.50 | *covered* | `right -> (34,35) cow->grass, (34,36) grass->cow, came into view: (37,30) grass, (37,31) tree, (37,32) grass, (37,33) grass, (37,34) grass, (37,35) gra…` |
| 65 | + reinforce | 0.70 | *covered* | `right -> came into view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) grass, (38,34) grass, (38,35) grass, (38,36) grass, left view: (29,30) w…` |
| 76 | - avoid | 0.85 | *covered* | `do -> nothing you can see changed` |
| 82 | + reinforce | 0.41 | **KEPT** | `left -> (35,34) cow->grass, (36,34) grass->cow, (37,36) grass->cow, came into view: (34,30) grass, (34,31) tree, (34,32) tree, (34,33) grass, (34,34) …` |
| 83 | + reinforce | 0.66 | *covered* | `left -> (36,34) cow->grass, (37,34) grass->cow, (37,35) grass->cow, (37,36) cow->grass, came into view: (33,30) grass, (33,31) tree, (33,32) tree, (33…` |
| 85 | + reinforce | 0.46 | **KEPT** | `left -> (36,35) grass->cow, (37,33) grass->cow, (37,34) cow->grass, (37,35) cow->grass, came into view: (31,30) grass, (31,31) grass, (31,32) grass, (…` |
| 86 | + reinforce | 0.57 | *covered* | `left -> (35,35) grass->cow, (36,35) cow->grass, (37,33) cow->grass, (38,33) grass->cow, came into view: (30,30) grass, (30,31) grass, (30,32) grass, (…` |
| 87 | + reinforce | 0.45 | **KEPT** | `left -> (37,33) grass->cow, came into view: (29,30) water, (29,31) grass, (29,32) grass, (29,33) grass, (29,34) grass, (29,35) grass, (29,36) grass, l…` |
| 89 | + reinforce | 0.65 | *covered* | `left -> (35,35) cow->grass, came into view: (27,30) water, (27,31) water, (27,32) water, (27,33) sand, (27,34) sand, (27,35) sand, (27,36) grass, left…` |
| 93 | - avoid | 0.49 | **KEPT** | `left -> came into view: (23,30) water, (23,31) water, (23,32) water, (23,33) water, (23,34) water, (23,35) water, (23,36) water, left view: (32,30) gr…` |
| 94 | - avoid | 0.88 | *covered* | `left -> nothing you can see changed` |
| 100 | + reinforce | 0.66 | *covered* | `right -> (36,34) grass->cow, came into view: (37,30) grass, (37,31) tree, (37,32) grass, (37,33) grass, (37,34) grass, (37,35) grass, (37,36) grass, l…` |
| 101 | + reinforce | 0.50 | *covered* | `right -> (36,34) cow->grass, (36,35) grass->cow, came into view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) grass, (38,34) grass, (38,35) gr…` |
| 115 | + reinforce | 0.53 | *covered* | `left -> (37,34) grass->cow, (38,34) cow->grass, came into view: (36,30) tree, (36,31) grass, (36,32) grass, (36,33) grass, (36,34) grass, (36,35) cow,…` |
| 117 | + reinforce | 0.60 | *covered* | `left -> (36,34) cow->grass, (36,35) grass->cow, came into view: (34,30) grass, (34,31) tree, (34,32) tree, (34,33) grass, (34,34) grass, (34,35) grass…` |
| 128 | - avoid | 0.58 | *covered* | `left -> came into view: (23,30) water, (23,31) water, (23,32) water, (23,33) water, (23,34) water, (23,35) water, (23,36) water, left view: (32,30) gr…` |
| 135 | + reinforce | 0.69 | *covered* | `right -> came into view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) grass, (38,34) grass, (38,35) grass, (38,36) grass, left view: (29,30) w…` |
| 137 | + reinforce | 0.25 | **KEPT** | `do -> sapling 0->1, achievement collect_sapling` |
| 138 | + reinforce | 0.25 | **KEPT** | `place_plant -> (36,33) grass->plant, sapling 1->0, achievement place_plant` |
| 139 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 140 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 141 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 142 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 143 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 145 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 148 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 149 | + reinforce | 0.60 | *covered* | `left -> came into view: (27,30) water, (27,31) water, (27,32) water, (27,33) sand, (27,34) sand, (27,35) sand, (27,36) grass, left view: (36,30) tree,…` |
| 152 | - avoid | 0.61 | *covered* | `do -> nothing you can see changed` |
| 154 | - avoid | 0.76 | *covered* | `left -> came into view: (23,30) water, (23,31) water, (23,32) water, (23,33) water, (23,34) water, (23,35) water, (23,36) water, left view: (32,30) gr…` |
| 155 | + reinforce | 0.23 | **KEPT** | `do -> drink 2->3, energy 5->4, achievement collect_drink` |
| 157 | + reinforce | 0.54 | *covered* | `do -> drink 4->5` |
| 158 | + reinforce | 0.53 | *covered* | `do -> drink 5->6` |
| 159 | + reinforce | 0.60 | *covered* | `do -> drink 6->7` |
| 160 | + reinforce | 0.51 | *covered* | `do -> drink 7->8` |
| 162 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 163 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 164 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 165 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 166 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 167 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 169 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 170 | - avoid | 0.67 | *covered* | `do -> nothing you can see changed` |
| 180 | - avoid | 0.21 | **KEPT** | `do -> (34,35) grass->zombie, (35,35) zombie->grass, health 9->7` |
| 181 | - avoid | 0.51 | *covered* | `do -> (34,34) grass->zombie, (34,35) zombie->grass` |
| 182 | + reinforce | 0.26 | **KEPT** | `do -> (34,33) grass->zombie, (34,34) zombie->grass, health 7->8, food 3->2` |
| 184 | + reinforce | 0.46 | **KEPT** | `do -> (33,33) zombie->grass, (34,33) grass->zombie, (35,33) zombie->grass, achievement defeat_zombie` |
| 186 | - avoid | 0.39 | **KEPT** | `do -> health 8->6, energy 4->3` |
| 188 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 189 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 191 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 192 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 193 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 194 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 195 | - avoid | 0.84 | *covered* | `do -> nothing you can see changed` |
| 196 | - avoid | 0.82 | *covered* | `do -> nothing you can see changed` |
| 197 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 198 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 199 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 200 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 201 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 202 | + reinforce | 0.87 | *covered* | `do -> sapling 0->1` |
| 203 | + reinforce | 0.59 | *covered* | `place_plant -> (33,33) grass->plant, sapling 1->0` |
| 204 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 205 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 206 | - avoid | 0.68 | *covered* | `do -> nothing you can see changed` |
| 207 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 208 | + reinforce | 0.44 | **KEPT** | `do -> (33,36) grass->zombie, health 6->7, food 2->1, drink 8->7` |
| 211 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 212 | - avoid | 0.63 | *covered* | `do -> nothing you can see changed` |
| 213 | + reinforce | 0.61 | *covered* | `do -> (36,36) grass->cow` |
| 215 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 222 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 223 | - avoid | 0.66 | *covered* | `do -> nothing you can see changed` |
| 224 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 225 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 226 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 227 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 232 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 233 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 235 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 236 | - avoid | 0.69 | *covered* | `do -> nothing you can see changed` |
| 237 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 238 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 239 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 240 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 241 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 242 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 243 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 244 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 245 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 246 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 247 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 249 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 254 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 256 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 257 | - avoid | 0.74 | *covered* | `do -> nothing you can see changed` |
| 259 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 260 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 261 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 264 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 265 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 266 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 267 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 268 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 272 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 273 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 274 | - avoid | 0.56 | *covered* | `do -> health 7->6` |
| 275 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 278 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 280 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 283 | - avoid | 0.78 | *covered* | `do -> nothing you can see changed` |
| 284 | - avoid | 0.80 | *covered* | `do -> nothing you can see changed` |
| 285 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 286 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 287 | - avoid | 0.73 | *covered* | `do -> nothing you can see changed` |
| 288 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 290 | - avoid | 0.54 | *covered* | `do -> (33,34) grass->zombie, (34,34) zombie->grass, health 6->5` |
| 291 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 293 | - avoid | 0.71 | *covered* | `do -> nothing you can see changed` |
| 294 | - avoid | 0.65 | *covered* | `do -> nothing you can see changed` |
| 295 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 298 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 299 | - avoid | 0.72 | *covered* | `do -> nothing you can see changed` |
| 300 | - avoid | 0.70 | *covered* | `do -> nothing you can see changed` |
| 306 | - avoid | 0.27 | **KEPT** | `do -> (34,35) zombie->grass, (34,36) grass->zombie, health 5->4` |
| 307 | - avoid | 0.58 | *covered* | `do -> (34,36) zombie->grass, (35,36) grass->zombie` |
| 309 | - avoid | 0.64 | *covered* | `do -> (34,35) grass->zombie, (35,35) zombie->grass` |
| 322 | - avoid | 0.52 | *covered* | `do -> (30,36) cow->grass, (31,36) grass->cow, (33,34) zombie->grass, (34,34) grass->zombie, health 4->3` |
| 328 | - avoid | 0.75 | *covered* | `do -> nothing you can see changed` |
| 329 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 330 | - avoid | 0.77 | *covered* | `do -> nothing you can see changed` |
| 331 | - avoid | 0.79 | *covered* | `do -> nothing you can see changed` |
| 333 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 335 | - avoid | 0.76 | *covered* | `do -> nothing you can see changed` |
| 338 | - avoid | 0.30 | **KEPT** | `do -> (30,36) zombie->grass, (33,34) zombie->grass, (33,35) grass->zombie, health 3->2` |
| 339 | - avoid | 0.33 | **KEPT** | `do -> (30,36) grass->zombie, (32,35) grass->zombie, (33,35) zombie->grass, (34,34) grass->zombie, (34,35) zombie->grass` |
| 340 | - avoid | 0.39 | **KEPT** | `do -> (30,35) grass->zombie, (30,36) zombie->grass, (33,34) grass->zombie, (34,34) zombie->grass` |
| 345 | - avoid | 0.74 | *covered* | `do -> (30,35) grass->zombie, (30,36) zombie->grass` |
| 351 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 354 | - avoid | 0.29 | **KEPT** | `do -> (28,35) cow->grass, (28,36) grass->cow, (31,36) grass->zombie, health 2->1` |
| 355 | - avoid | 0.50 | *covered* | `do -> (31,35) grass->zombie, (31,36) zombie->grass, drink 1->0` |
| 357 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |
| 360 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 365 | - avoid | 0.81 | *covered* | `do -> nothing you can see changed` |
| 368 | + reinforce | 0.85 | *covered* | `do -> (34,33) grass->zombie, (34,34) zombie->grass` |
| 369 | - avoid | 0.83 | *covered* | `do -> nothing you can see changed` |

## Final store — reinforce (11)

- `left -> (30,35) cow->grass, (30,36) grass->cow, came into view: (27,30) water, (27,31) water, (27,32) water, (27,33) sand, (27,34) sand, (27,35) sand, (27,36) grass, left view: (36,30) tree, (36,31) grass, (36,32) grass, (36,33) grass, (36,34) grass, (36,35) grass, (36,36) grass, facing down->left, you are now at (31,33)`
- `right -> (32,36) grass->cow, came into view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) grass, (38,34) grass, (38,35) grass, (38,36) grass, left view: (29,30) water, (29,31) grass, (29,32) grass, (29,33) grass, (29,34) grass, (29,35) grass, (29,36) grass, you are now at (34,33)`
- `left -> (35,34) cow->grass, (36,34) grass->cow, (37,36) grass->cow, came into view: (34,30) grass, (34,31) tree, (34,32) tree, (34,33) grass, (34,34) grass, (34,35) grass, (34,36) grass, left view: (43,30) sand, (43,31) tree, (43,32) grass, (43,33) grass, (43,34) grass, (43,35) grass, (43,36) grass, you are now at (38,33)`
- `left -> (36,35) grass->cow, (37,33) grass->cow, (37,34) cow->grass, (37,35) cow->grass, came into view: (31,30) grass, (31,31) grass, (31,32) grass, (31,33) grass, (31,34) tree, (31,35) grass, (31,36) grass, left view: (40,30) grass, (40,31) grass, (40,32) grass, (40,33) grass, (40,34) grass, (40,35) grass, (40,36) grass, you are now at (35,33)`
- `left -> (37,33) grass->cow, came into view: (29,30) water, (29,31) grass, (29,32) grass, (29,33) grass, (29,34) grass, (29,35) grass, (29,36) grass, left view: (38,30) grass, (38,31) grass, (38,32) grass, (38,33) cow, (38,34) grass, (38,35) grass, (38,36) grass, you are now at (33,33)`
- `do -> sapling 0->1, achievement collect_sapling`
- `place_plant -> (36,33) grass->plant, sapling 1->0, achievement place_plant`
- `do -> drink 2->3, energy 5->4, achievement collect_drink`
- `do -> (34,33) grass->zombie, (34,34) zombie->grass, health 7->8, food 3->2`
- `do -> (33,33) zombie->grass, (34,33) grass->zombie, (35,33) zombie->grass, achievement defeat_zombie`
- `do -> (33,36) grass->zombie, health 6->7, food 2->1, drink 8->7`

## Final store — avoid (12)

- `up -> nothing you can see changed`
- `down -> nothing you can see changed`
- `left -> nothing you can see changed`
- `noop -> nothing you can see changed`
- `left -> came into view: (23,30) water, (23,31) water, (23,32) water, (23,33) water, (23,34) water, (23,35) water, (23,36) water, left view: (32,30) grass, (32,31) grass, (32,32) grass, (32,33) grass, (32,34) tree, (32,35) grass, (32,36) grass, you are now at (27,33), energy 7->6`
- `do -> (34,35) grass->zombie, (35,35) zombie->grass, health 9->7`
- `do -> health 8->6, energy 4->3`
- `do -> (34,35) zombie->grass, (34,36) grass->zombie, health 5->4`
- `do -> (30,36) zombie->grass, (33,34) zombie->grass, (33,35) grass->zombie, health 3->2`
- `do -> (30,36) grass->zombie, (32,35) grass->zombie, (33,35) zombie->grass, (34,34) grass->zombie, (34,35) zombie->grass`
- `do -> (30,35) grass->zombie, (30,36) zombie->grass, (33,34) grass->zombie, (34,34) zombie->grass`
- `do -> (28,35) cow->grass, (28,36) grass->cow, (31,36) grass->zombie, health 2->1`

## Store growth

step 5 → 1, step 11 → 2, step 13 → 3, step 18 → 4, step 27 → 5, step 30 → 6, step 82 → 7, step 85 → 8, step 87 → 9, step 93 → 10, step 137 → 11, step 138 → 12, step 155 → 13, step 180 → 14, step 182 → 15, step 184 → 16, step 186 → 17, step 208 → 18, step 306 → 19, step 338 → 20, step 339 → 21, step 340 → 22, step 354 → 23
