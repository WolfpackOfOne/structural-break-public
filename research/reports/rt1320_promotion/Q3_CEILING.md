# Q3 — Is the existing legal OOF bank at its ensemble ceiling?

Analysis ID: `AN-Q3-D4-20260903`
Commit parent: `a535042bf70fe3c31f505fc774f4a34715ccd1ad`
Canonical development folds 0–4 only: 4,032,524 online rows.

## Decision rule (frozen before computation)

- **CEILING BINDING** if no combination beats the current ensemble by more than `0.0011` with fold consistency; then the existing bank is saturated and the program is finished.
- **CEILING DEAD** if any combination clears that bar with fold consistency; report the exact vectors and margin.
- Otherwise **INCONCLUSIVE**.

Fold consistency is fixed as at least **4/5 positive canonical outer-fold deltas**, matching the existing program convention. The already pre-registered four-partition RT1320 replication retains its frozen **at least 3/4 positive partitions** rule.

## Verdict: CEILING DEAD

An existing-vector combination beats RT1257 by >0.0011 with the fixed consistency requirement.

- `RT-1255 + RT-410 + RT-411 + RT-412 + RT-1254 + RT-414 + RT-415 + RT-1320` — canonical fold-pure RT1257 equal-weight addition; mean margin **+0.0014160**; vector `[-0.0001495, +0.0025516, +0.0008956, +0.0037165, +0.0000658]`.
- `RT-1255 + RT-410 + RT-411 + RT-412 + RT-1254 + RT-414 + RT-415 + RT-994` — canonical fold-pure RT1257 equal-weight addition; mean margin **+0.0011039**; vector `[+0.0003884, +0.0016733, -0.0000522, +0.0029383, +0.0005716]`.
- `outer-fold-specific greedy paths` — nested fold-pure greedy selection; mean margin **+0.0022082**; vector `[+0.0003640, +0.0047521, +0.0030138, +0.0051407, -0.0022297]`.
- `RT-1255 + RT-410 + RT-411 + RT-412 + RT-1254 + RT-414 + RT-415 + RT-1320` — pre-registered RT1320 addition contract, four partitions; mean margin **+0.0015751**; vector `[+0.0014161, +0.0018707, +0.0012849, +0.0017287]`.

The decisive fixed combination is `RT-1257 + RT-1320`. Its pre-registered canonical/alt1/alt2/alt3 E2−E0 mean is **+0.0015751**, positive on 4/4 partitions and above `0.0011`. The legal bank is therefore not saturated.

## D4 protocol

The bank contains **72** registered, legal, full-fold, one-dimensional OOF vectors: 46 legacy, 17 leaderboard-alpha, RT-1251, seven deep-ensemble vectors, and RT-1320.

For outer fold `k`, its SCDF_NSEEN maps are fit on folds `!=k`; selection-time fold `g` maps are fit excluding both `k` and `g`. Greedy selection starts at RT1257, chooses the best equal-weight addition using only folds `!=k`, and stops at no training-fold improvement. Held-out `k` never chooses a vector or weight.

Leave-one-in adds each vector at `1/8` to RT600 versus RT401 and to RT1257 versus RT403 under the same outer-fold calibration.

### Greedy outer-fold results

| outer | training-selected additions | held-out RT1257 | held-out final | delta |
|---:|---|---:|---:|---:|
| 0 | RT-1320, RT-1246_RT-414, RT-1246_RT-411, RT-1248, RT-813, RT-1261, RT-971, RT-1245_RT-414, RT-994, RT-961, RT-1262, RT-812, RT-433, RT-1249, RT-1246_RT-412 | 0.640430252 | 0.640794217 | +0.0003640 |
| 1 | RT-1320, RT-1246_RT-411, RT-1248, RT-813, RT-961, RT-1261, RT-1246_RT-414, RT-812, RT-1249, RT-1246_RT-412, RT-994, RT-433, RT-1260 | 0.620692590 | 0.625444663 | +0.0047521 |
| 2 | RT-1320, RT-1248, RT-812, RT-1246_RT-414, RT-994, RT-813, RT-1261, RT-1246_RT-411, RT-961, RT-1262, RT-1249 | 0.635138075 | 0.638151842 | +0.0030138 |
| 3 | RT-1320, RT-1261, RT-1249, RT-1246_RT-411, RT-1245_RT-414, RT-961, RT-812, RT-1246_RT-414, RT-1248, RT-711, RT-813, RT-1251, RT-994, RT-1262, RT-760 | 0.620745947 | 0.625886680 | +0.0051407 |
| 4 | RT-1320, RT-1246_RT-411, RT-812, RT-961, RT-1261, RT-994, RT-1249, RT-751, RT-1263 | 0.622181288 | 0.619951600 | -0.0022297 |

Final greedy delta vector `[+0.0003640, +0.0047521, +0.0030138, +0.0051407, -0.0022297]`; mean **+0.0022082**, positive 4/5. Paths remain outer-fold-specific; no post-heldout consensus path was manufactured.

## Effective rank

The 72×72 within-t rank-correlation matrix has participation effective rank **2.730** and entropy effective rank **9.040**. It needs 33/50/66 components for 90%/95%/99% spectral mass; the top eigenvalue holds 60.08%, and mean off-diagonal correlation is 0.5690.

The bank is strongly redundant, but redundancy is not a binding ceiling: RT1320 is a direct fold- and partition-consistent counterexample.

## Leave-one-in marginals

Sorted by mean candidate-minus-RT403 margin on RT1257. Full per-fold vectors are in the JSON.

| vector | standalone | +RT600 | vs RT401 | +folds | +RT1257 | vs RT403 | +folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-1320 | 0.615606059 | +0.0019793 | +0.0019523 | 5/5 | +0.0014160 | +0.0015157 | 4/5 |
| RT-994 | 0.615340923 | +0.0015586 | +0.0015316 | 5/5 | +0.0011039 | +0.0012036 | 4/5 |
| RT-1246_RT-411 | 0.608562224 | +0.0012787 | +0.0012517 | 4/5 | +0.0009925 | +0.0010922 | 4/5 |
| RT-812 | 0.609750813 | +0.0011777 | +0.0011507 | 4/5 | +0.0009302 | +0.0010299 | 4/5 |
| RT-1261 | 0.623508392 | +0.0014209 | +0.0013940 | 5/5 | +0.0008263 | +0.0009260 | 4/5 |
| RT-1248 | 0.564739962 | +0.0010585 | +0.0010316 | 4/5 | +0.0007289 | +0.0008287 | 3/5 |
| RT-813 | 0.615207562 | +0.0006957 | +0.0006687 | 5/5 | +0.0005786 | +0.0006783 | 4/5 |
| RT-995 | 0.621275121 | +0.0008381 | +0.0008112 | 5/5 | +0.0005736 | +0.0006733 | 4/5 |
| RT-1260 | 0.609377678 | +0.0009382 | +0.0009112 | 5/5 | +0.0005387 | +0.0006384 | 4/5 |
| RT-1246_RT-414 | 0.610484372 | +0.0008643 | +0.0008373 | 3/5 | +0.0004551 | +0.0005548 | 3/5 |
| RT-1262 | 0.615326421 | +0.0010363 | +0.0010093 | 4/5 | +0.0003851 | +0.0004848 | 4/5 |
| RT-1246_RT-412 | 0.612632561 | +0.0005250 | +0.0004980 | 3/5 | +0.0003474 | +0.0004471 | 3/5 |
| RT-1245_RT-411 | 0.602596210 | +0.0006059 | +0.0005789 | 4/5 | +0.0003212 | +0.0004209 | 3/5 |
| RT-750 | 0.617355430 | +0.0004980 | +0.0004711 | 5/5 | +0.0003085 | +0.0004082 | 4/5 |
| RT-1251 | 0.620406479 | +0.0010028 | +0.0009759 | 5/5 | +0.0002868 | +0.0003865 | 4/5 |
| RT-1246_RT-300 | 0.614705729 | +0.0005493 | +0.0005223 | 4/5 | +0.0002712 | +0.0003710 | 2/5 |
| RT-751 | 0.617255826 | +0.0004820 | +0.0004550 | 3/5 | +0.0002687 | +0.0003685 | 4/5 |
| RT-1263 | 0.619833221 | +0.0009247 | +0.0008977 | 5/5 | +0.0002231 | +0.0003228 | 2/5 |
| RT-431 | 0.604880951 | +0.0003846 | +0.0003576 | 3/5 | +0.0002203 | +0.0003200 | 3/5 |
| RT-760 | 0.614680362 | +0.0004492 | +0.0004223 | 3/5 | +0.0002062 | +0.0003060 | 3/5 |
| RT-432 | 0.613395853 | +0.0003468 | +0.0003198 | 4/5 | +0.0001986 | +0.0002983 | 4/5 |
| RT-961 | 0.580603692 | +0.0001604 | +0.0001334 | 3/5 | +0.0001563 | +0.0002560 | 2/5 |
| RT-1245_RT-414 | 0.608091166 | +0.0006088 | +0.0005818 | 3/5 | +0.0001537 | +0.0002534 | 3/5 |
| RT-1249 | 0.543249146 | +0.0002759 | +0.0002490 | 3/5 | +0.0000961 | +0.0001959 | 2/5 |
| RT-816 | 0.615755086 | +0.0002696 | +0.0002426 | 3/5 | +0.0000949 | +0.0001946 | 3/5 |
| RT-1254 | 0.620438803 | +0.0009930 | +0.0009660 | 5/5 | +0.0000732 | +0.0001730 | 3/5 |
| RT-433 | 0.610523379 | +0.0003971 | +0.0003702 | 4/5 | +0.0000319 | +0.0001317 | 3/5 |
| RT-411 | 0.607812663 | +0.0002722 | +0.0002453 | 3/5 | +0.0000310 | +0.0001307 | 3/5 |
| RT-1255 | 0.620239534 | +0.0009580 | +0.0009310 | 5/5 | +0.0000279 | +0.0001276 | 2/5 |
| RT-971 | 0.543179485 | +0.0002215 | +0.0001945 | 3/5 | +0.0000119 | +0.0001117 | 4/5 |
| RT-730 | 0.615744372 | +0.0002005 | +0.0001735 | 4/5 | +0.0000087 | +0.0001084 | 3/5 |
| RT-731 | 0.614822125 | +0.0001507 | +0.0001238 | 3/5 | -0.0000222 | +0.0000775 | 3/5 |
| RT-815 | 0.607883009 | +0.0003249 | +0.0002979 | 3/5 | -0.0000698 | +0.0000300 | 2/5 |
| RT-970 | 0.541519919 | +0.0001260 | +0.0000990 | 3/5 | -0.0000827 | +0.0000171 | 4/5 |
| RT-1245_RT-412 | 0.609552070 | +0.0001396 | +0.0001127 | 2/5 | -0.0000996 | +0.0000001 | 3/5 |
| RT-403 | 0.616869329 | +0.0000624 | +0.0000354 | 2/5 | -0.0000997 | +0.0000000 | 0/5 |
| RT-401 | 0.616609041 | +0.0000270 | +0.0000000 | 0/5 | -0.0001525 | -0.0000527 | 3/5 |
| RT-434 | 0.615793130 | -0.0001174 | -0.0001444 | 2/5 | -0.0002174 | -0.0001177 | 3/5 |
| RT-740 | 0.614583978 | -0.0000569 | -0.0000838 | 3/5 | -0.0002383 | -0.0001386 | 2/5 |
| RT-300 | 0.616051915 | -0.0003932 | -0.0004202 | 1/5 | -0.0002391 | -0.0001393 | 2/5 |
| RT-412 | 0.613933176 | -0.0001293 | -0.0001562 | 2/5 | -0.0002445 | -0.0001448 | 2/5 |
| RT-1246_RT-415 | 0.611443249 | -0.0000051 | -0.0000321 | 2/5 | -0.0002532 | -0.0001535 | 2/5 |
| RT-413 | 0.614808947 | -0.0005875 | -0.0006144 | 0/5 | -0.0002713 | -0.0001716 | 2/5 |
| RT-702 | 0.614808947 | -0.0005875 | -0.0006144 | 0/5 | -0.0002713 | -0.0001716 | 2/5 |
| RT-1256 | 0.610328372 | +0.0003947 | +0.0003677 | 3/5 | -0.0002796 | -0.0001798 | 3/5 |
| RT-414 | 0.611040577 | +0.0000892 | +0.0000623 | 2/5 | -0.0003058 | -0.0002061 | 2/5 |
| RT-711 | 0.607700822 | -0.0002855 | -0.0003125 | 3/5 | -0.0003172 | -0.0002174 | 3/5 |
| RT-303 | 0.614874154 | -0.0001834 | -0.0002104 | 2/5 | -0.0003417 | -0.0002420 | 3/5 |
| RT-404 | 0.614845633 | -0.0001979 | -0.0002248 | 1/5 | -0.0003623 | -0.0002626 | 2/5 |
| RT-405 | 0.614979543 | -0.0002260 | -0.0002530 | 1/5 | -0.0003918 | -0.0002921 | 2/5 |
| RT-406 | 0.615168667 | -0.0002226 | -0.0002495 | 2/5 | -0.0003960 | -0.0002963 | 2/5 |
| RT-710 | 0.614713323 | -0.0002893 | -0.0003163 | 1/5 | -0.0004331 | -0.0003334 | 2/5 |
| RT-700 | 0.613339280 | -0.0003700 | -0.0003970 | 1/5 | -0.0004436 | -0.0003439 | 1/5 |
| RT-1246_RT-410 | 0.605086912 | -0.0001428 | -0.0001698 | 2/5 | -0.0004501 | -0.0003504 | 2/5 |
| RT-415 | 0.617080983 | -0.0002659 | -0.0002928 | 0/5 | -0.0004555 | -0.0003558 | 2/5 |
| RT-1245_RT-300 | 0.608992010 | -0.0001592 | -0.0001862 | 3/5 | -0.0004832 | -0.0003835 | 3/5 |
| RT-990 | 0.611851846 | -0.0004554 | -0.0004823 | 1/5 | -0.0004863 | -0.0003866 | 1/5 |
| RT-1246_RT-413 | 0.609279621 | -0.0003023 | -0.0003293 | 2/5 | -0.0005447 | -0.0004450 | 2/5 |
| RT-302 | 0.614122008 | -0.0003717 | -0.0003987 | 0/5 | -0.0005544 | -0.0004547 | 1/5 |
| RT-430 | 0.609146210 | -0.0002909 | -0.0003179 | 1/5 | -0.0005609 | -0.0004612 | 2/5 |
| RT-960 | 0.570587580 | -0.0005711 | -0.0005981 | 1/5 | -0.0005838 | -0.0004841 | 2/5 |
| RT-701 | 0.610044744 | -0.0005056 | -0.0005326 | 1/5 | -0.0005952 | -0.0004955 | 1/5 |
| RT-402 | 0.613567518 | -0.0005085 | -0.0005355 | 0/5 | -0.0006930 | -0.0005932 | 1/5 |
| RT-1245_RT-410 | 0.602576478 | -0.0003828 | -0.0004097 | 2/5 | -0.0007330 | -0.0006332 | 3/5 |
| RT-811 | 0.603792697 | -0.0004859 | -0.0005129 | 1/5 | -0.0007683 | -0.0006685 | 1/5 |
| RT-301 | 0.612567689 | -0.0005939 | -0.0006209 | 0/5 | -0.0007792 | -0.0006795 | 1/5 |
| RT-1245_RT-415 | 0.607675605 | -0.0004907 | -0.0005176 | 2/5 | -0.0007954 | -0.0006957 | 3/5 |
| RT-1245_RT-413 | 0.606833157 | -0.0005542 | -0.0005811 | 1/5 | -0.0008261 | -0.0007264 | 2/5 |
| RT-814 | 0.608705624 | -0.0007736 | -0.0008006 | 1/5 | -0.0009306 | -0.0008309 | 0/5 |
| RT-712 | 0.601320657 | -0.0011356 | -0.0011626 | 0/5 | -0.0012291 | -0.0011293 | 1/5 |
| RT-410 | 0.605112820 | -0.0009947 | -0.0010217 | 1/5 | -0.0012565 | -0.0011567 | 1/5 |
| RT-1247 | 0.528172231 | -0.0014001 | -0.0014271 | 1/5 | -0.0015589 | -0.0014591 | 1/5 |

## Required reference points

- W5-NULLTEST: **+0.0000270** (reference `+0.00003`).
- W4-E6 13-stream union minus RT600: **-0.0009519**; worse, as reported.
- `m09_back`: within-t correlation **0.8219** versus clone **0.7846**; candidate-minus-clone **-0.0012587**.
- `m12_rdep`: candidate-minus-clone **+0.0014052** (reference `+0.00141`).
- Noise floor: **0.0011**.
- RT1320 four-partition mean E2−E0: **+0.0015751** (reference approximately `+0.0016`).

**Reference reproduction: PASS — all six checks reproduced.**

Legacy raw-logit m09/m12 and RT1320 partition values use their exact frozen contracts; W4/W5 are recomputed from frozen OOF under fold-pure SCDF_NSEEN.

## Scope and exclusions

- RT-500..RT-506: final-10k calibration vectors fitted with the spent fold -1 lockbox.
- RT-900/RT-991: void/oracle tau or label leakage; RT-992/RT-993: contaminated fold-0 pilots.
- RT-1000-series and RT-1200..RT-1242 screens: fold-0 experiments or later folds generated only for calibration support, not registered full-fold candidates.
- RT-1234/1235 and RT-1237..1242 local arrays: non-finite outside fold 0.
- CATSLOT-* and M2-RTLF scratch arrays: unregistered/provenance-ambiguous; registered exact duplicates are represented by their RT vector.
- Alternate partitions, nested teachers, precalibrated vectors, ensemble caches: validation surfaces, fit auxiliaries, or deterministic derivatives rather than independent candidates.

No detector was trained. No fold -1 row, test/lockbox artifact, `folds_final10k`, `RESULTS.csv`, setup, Crunch push, or RT allocation was read, scored, or changed. Whole-file hashes below are byte-level provenance only.

## Input provenance

| vector | bytes | SHA-256 | source |
|---|---:|---|---|
| RT-1245_RT-300 | 20146196 | `0350a20c015df48c32691b87d6299e4df3bc746cb1fe83450ca4c27a718aa9e8` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-300.npy` |
| RT-1245_RT-410 | 20146196 | `6894f546695167bb2a85eab5e2795c8a136a70c196717bb3b5b2849900f98f0c` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-410.npy` |
| RT-1245_RT-411 | 20146196 | `8fc8fbce0967f54d4bdeea8f1a1965e1f9388bf6008932438ac50e344a4ccf95` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-411.npy` |
| RT-1245_RT-412 | 20146196 | `0c56912d91d3d6844882cd9a77794daa288658a4655f1daae25cbe271a2010b6` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-412.npy` |
| RT-1245_RT-413 | 20146196 | `7b7e48f089cb13b08ab40c0554be3e86c8eb63bcc5cdf99781b8fdb0a21c7a90` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-413.npy` |
| RT-1245_RT-414 | 20146196 | `03001314df55b1635ca82c659fcb5a975944c73e51c1f9d08a2db179d2a72fde` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-414.npy` |
| RT-1245_RT-415 | 20146196 | `c9322312e255d7ef0916d42b9a3bd5fb443da291ca1f952416a572536197fb4d` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1245_RT-415.npy` |
| RT-1246_RT-300 | 20146196 | `f401a894384289a5315cade78dbe1e70c90baa02121251cac007fde698dbb4a8` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-300.npy` |
| RT-1246_RT-410 | 20146196 | `c8e0bb663f754a6a0dccca063656d4796c31ed8f44ecd6e685c0e14a59af50bb` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-410.npy` |
| RT-1246_RT-411 | 20146196 | `23f5c8be3274dc1a3f46363770a3c84af59a7e0ec7ef218f148b6bab1c04aae1` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-411.npy` |
| RT-1246_RT-412 | 20146196 | `faf7a935921ef8d4e114fb358129325b3d63be017442704e683ea1a3cef40968` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-412.npy` |
| RT-1246_RT-413 | 20146196 | `e917922deb0cc1562a4e4deaa4ead2a3bd0625166c7c03071c7b3d2dd88936f8` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-413.npy` |
| RT-1246_RT-414 | 20146196 | `adffdc48c9f6809a6f62b47621453e1ac3262db25d90b5355bdf9ffae7562b9d` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-414.npy` |
| RT-1246_RT-415 | 20146196 | `f2c9f6a207f0ee0a335d40c73c7836ff772aaa76b4c61f24ae893a6bbfc310dc` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1246_RT-415.npy` |
| RT-1247 | 20146196 | `85be05e3ab0cdcf851ceca6519efb309284fb474a697b9199ed1f627c54d38da` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1247.npy` |
| RT-1248 | 20146196 | `e295e1da694e071f673c399542023907d37abbb46fcd174ff0c393ae5cc631b5` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1248.npy` |
| RT-1249 | 20146196 | `5e97ee5ded1949b83f4d8d8be1d444468dfcfd35bd12af2d24fa3cfbc0756582` | `/path/to/workspace/structural-break-leaderboard-alpha-2026/research/oof/RT-1249.npy` |
| RT-1251 | 20146196 | `82805c2d46e6b8e2046c878bd341479d25b59b2bf3fee545e6bba7d3891b83c5` | `/path/to/workspace/structural-break-learner-diversity-2026/research/oof/RT-1251.npy` |
| RT-1254 | 20146196 | `3025da0531aab9af2c65235106e4c426328c808aaea8a3d53c00375a842ad5cf` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1254.npy` |
| RT-1255 | 20146196 | `26a6d8474f4edfe5df9cd83b5d835cb3ba13cd39a2917473f95ec881df1ca8ac` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1255.npy` |
| RT-1256 | 20146196 | `66b59333a78fd0048aec717af43c402e3848371ce0a29d20796a347a1807f913` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1256.npy` |
| RT-1260 | 20146196 | `68d8614c07118df42fdc19070b0d6ed7103991af90a66c384497e497c466458c` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1260.npy` |
| RT-1261 | 20146196 | `536d6f4732ea6c3cda71649bd6fca7da9f3950d5145aee47ebbbcd0910f5fb62` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1261.npy` |
| RT-1262 | 20146196 | `8d563ef82d98bbc9af41e35dda01fd75100c590fed66b929235c26fe1a31c079` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1262.npy` |
| RT-1263 | 20146196 | `62a1e8405b57fb2a77d3f818f20920767b17c5ad4d35e92dbc2d775b65b45f99` | `/path/to/workspace/structural-break-deep-ensemble-frontier-local-2026/research/oof/RT-1263.npy` |
| RT-1320 | 20146196 | `53170d390fbb5f666c7764a19cc5f01853091148df3ad9f28460487f9c89fd7a` | `/path/to/workspace/structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student_confirm_s20260901/armc_residual_student_oof.npy` |
| RT-300 | 20146196 | `bd3e6456fef300a7d0fefacec92095724ab2401fbe046c28733c9600b96879d3` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-300.npy` |
| RT-301 | 20146196 | `c9c2ccc90ec0d76a5ef31c2aa236dc3f65d1ecb63d477dfe59d49e229872d3c9` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-301.npy` |
| RT-302 | 20146196 | `a22c2b487ede13fdcb6e8189856cdbec9225324ec18f966cc542bec3f4eb719b` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-302.npy` |
| RT-303 | 20146196 | `290396b542c13d14ae518b091d13ce3d821da369886e1262849482236df7ac15` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-303.npy` |
| RT-401 | 20146196 | `16e1a10e75f3a2adf12d0efdfe8f4dd9391cfb53f871909a3f7392d2f175f57e` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-401.npy` |
| RT-402 | 20146196 | `dd2901750916a5ad4ba66372a3c974fb7a4dc4f9f53fdb75da1637c1bc1ae730` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-402.npy` |
| RT-403 | 20146196 | `81f5fb6a3004adb4672b386c50e33fc97e6d08d6ab694fa2e929ba3fd2a4f4e6` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-403.npy` |
| RT-404 | 20146196 | `512a8f4c3201939372bdcf2d861e6dbac49e80e2246214972759478acc677fc2` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-404.npy` |
| RT-405 | 20146196 | `5fb87cfd0373e045f0857b92da669dce56973a424df1f6f84de766b637a83a5f` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-405.npy` |
| RT-406 | 20146196 | `f9207bd78dab266c8ef8b0c1cc13b5961a4a916fe8968a4264172e9572b7fc0b` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-406.npy` |
| RT-410 | 20146196 | `cbe1282006585cdcceb020b8c67ba62e2cc8be551bb93b81d9a19bac8ec628cb` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-410.npy` |
| RT-411 | 20146196 | `f08248b3c47683046e4531637179a198df5cdcb024539ddc098eb88ca1158612` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-411.npy` |
| RT-412 | 20146196 | `2684cec6917c22b44af0b37f5e5b055e6159d7d452b0989bdc21cee0f73a3ae0` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-412.npy` |
| RT-413 | 20146196 | `f101827affa4394e83b007c22cadaa06f1f4021771615a6db02c17c50b6ba4ac` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-413.npy` |
| RT-414 | 20146196 | `31e98d95f7453d2a9bdf22ad877df6c1e8a032a9b7d12464c76d77885921972f` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-414.npy` |
| RT-415 | 20146196 | `04e6cc609a4791021d164f8037a1f2df8f8bca9618d408a97cdd7253ee0127b6` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-415.npy` |
| RT-430 | 20146196 | `06d50de9abcabd16998b707a9b73bf7ef66b321c4670bee8534274691b813ec6` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-430.npy` |
| RT-431 | 20146196 | `f3cb0b28160ed893e8653e097f08e01883b927fa4de90a0d07a7725f03122cd1` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-431.npy` |
| RT-432 | 20146196 | `ebd14f72530ed214f367cd7664e7450f53f61d47dc3962628b284b01d0e7197e` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-432.npy` |
| RT-433 | 20146196 | `559bef47e4af8549b90b00963a3d8bf14a6df78328e1264dd68c78d65ba1685a` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-433.npy` |
| RT-434 | 20146196 | `732115680c464fec64e66c25512d9bd459036a24dc552e1a055e79dba20b138b` | `/path/to/workspace/structural-break-claude-wave3/research/oof/RT-434.npy` |
| RT-700 | 20146196 | `1de92831ebf36cfd5f4da3fbe23c64faae64a4fca769288774692390e960c9dc` | `/path/to/workspace/structural-break-wave5/research/oof/RT-700.npy` |
| RT-701 | 20146196 | `43c9206b9022f27c38fb260457e6f81b3b53f0f19b36154e538bfd6be3f734fb` | `/path/to/workspace/structural-break-wave5/research/oof/RT-701.npy` |
| RT-702 | 20146196 | `f101827affa4394e83b007c22cadaa06f1f4021771615a6db02c17c50b6ba4ac` | `/path/to/workspace/structural-break-wave5/research/oof/RT-702.npy` |
| RT-710 | 20146196 | `4a99bfc6b715b3cf0cb97ea663ae0376a480346ff8fcdcb4483e3b05a9e4e3cd` | `/path/to/workspace/structural-break-wave5/research/oof/RT-710.npy` |
| RT-711 | 20146196 | `087abe25cad76a24a6979413b9be1e10b86e58b79735861ee7e76cf41f3ae607` | `/path/to/workspace/structural-break-wave5/research/oof/RT-711.npy` |
| RT-712 | 20146196 | `e6c16bd1d2585b192cd1259a8cf42306dc44aa7ea7f26691008e16977667abd9` | `/path/to/workspace/structural-break-wave5/research/oof/RT-712.npy` |
| RT-730 | 20146196 | `674fe0c6d6146bb1e8fe684db785459e1da055b116782a028f202eaaa4caa96b` | `/path/to/workspace/structural-break-wave5/research/oof/RT-730.npy` |
| RT-731 | 20146196 | `e9a0699cf1edf6c6e6441fb57232360cb6f50d222604e6943f91cab35a23d2cd` | `/path/to/workspace/structural-break-wave5/research/oof/RT-731.npy` |
| RT-740 | 20146196 | `9f8dcd364172e99382f60373054156894768a9fdc96ae0f04d578a40d65d703b` | `/path/to/workspace/structural-break-wave5/research/oof/RT-740.npy` |
| RT-750 | 20146196 | `be82b2e8675b6de7cd64403360f86828f267682ad5ac71f47b06844e6553a900` | `/path/to/workspace/structural-break-wave5/research/oof/RT-750.npy` |
| RT-751 | 20146196 | `df5566f1c241ac79175e29fdf5ef1ec99101642038eddb2ce760679748bf2493` | `/path/to/workspace/structural-break-wave5/research/oof/RT-751.npy` |
| RT-760 | 20146196 | `5bdb6de3ad405bb925a060d0116efafe0b86b1fb8f6737899aabc09f499d8d60` | `/path/to/workspace/structural-break-wave5/research/oof/RT-760.npy` |
| RT-811 | 20146196 | `e2ca0370d8b1dd6ff26961aa95d521fab3a8e199923a1ac7f7d324c91dfff2f2` | `/path/to/workspace/structural-break-wave5/research/oof/RT-811.npy` |
| RT-812 | 20146196 | `f3511d17f33d5303be2f77a3e368bac546a91aabe0feb6e4b19098cdb065d7f8` | `/path/to/workspace/structural-break-wave5/research/oof/RT-812.npy` |
| RT-813 | 20146196 | `7fa6a889e3282d2ed460a27faf88ce2f41fe3316c2b26592f17549022167a79e` | `/path/to/workspace/structural-break-wave5/research/oof/RT-813.npy` |
| RT-814 | 20146196 | `fd1ea10735ee0f8d6ae279f5c4793ac892059d79bd52db4a2815f75cbd0815e8` | `/path/to/workspace/structural-break-wave5/research/oof/RT-814.npy` |
| RT-815 | 20146196 | `0596313c6d4c617c45e589f6ece4b0419df3047ca2a42dd99339eef802261b27` | `/path/to/workspace/structural-break-wave5/research/oof/RT-815.npy` |
| RT-816 | 20146196 | `0c220da2f6d6ed14d7a4db07ad2ad6c2de0288851a877deaf8fa8efa3a9b778a` | `/path/to/workspace/structural-break-wave5/research/oof/RT-816.npy` |
| RT-960 | 20146196 | `f3c33e9720296fecbe837490770d7c75fd0afbf82a7c8543b5cd25c49314d439` | `/path/to/workspace/structural-break-wave8/research/oof/RT-960.npy` |
| RT-961 | 20146196 | `c7c253b7a53deff00f8ceaa748dab10a76a56a176a09fea34e099b0609ad2486` | `/path/to/workspace/structural-break-wave8/research/oof/RT-961.npy` |
| RT-970 | 20146196 | `4ff163a2db65d2e8fcd687c6c2337f4d3aa8736c220359bee1a4f45bc0fc30e7` | `/path/to/workspace/structural-break-wave8/research/oof/RT-970.npy` |
| RT-971 | 20146196 | `56bfe494bef8481e730fc487135d4229c67dfc665622073280bf0a1ffe70a07e` | `/path/to/workspace/structural-break-wave8/research/oof/RT-971.npy` |
| RT-990 | 20146196 | `872902e7ac6525a98787997607348d8d54f31d1d7e3002aad971cec2ac1b7c41` | `/path/to/workspace/structural-break-wave8/research/oof/RT-990.npy` |
| RT-994 | 20146196 | `28cc1174def755fa83624fa038f767d42fdad0573b508e2cebeb1b5770746407` | `/path/to/workspace/structural-break-wave8/research/oof/RT-994.npy` |
| RT-995 | 20146196 | `988eb7c4665088a74fcc4617527d6289df8b969163bb04af900c73745090f144` | `/path/to/workspace/structural-break-wave8/research/oof/RT-995.npy` |

Reference files:

- `synthesis` — `1822df5b0b9d44e55fc6999001b7e23438e0896074625198b3d2576a7f43fcb9` — `/path/to/workspace/structural-break-main-current/research/ai/investigations/20260829-173245-unresolved-quant-questions/05_SYNTHESIS.md`
- `primary_research` — `2003304917a9ca91863e3ad794f1d4a3d8b03d36d21a3847549e4005db75448d` — `/path/to/workspace/structural-break-main-current/research/ai/investigations/20260829-173245-unresolved-quant-questions/02_PRIMARY_RESEARCH.md`
- `wave4_union` — `495cce6f007905721cc03a4feb4e342e210ef45e1386a2162fafa37acc515652` — `/path/to/workspace/structural-break-main-current/research/reports/wave4_union_ensemble.json`
- `rt1320_stage2` — `468c66364a405c4c3938a4fda3479a799a52a1c4a15c5bf5676a83919f1b91e4` — `/path/to/workspace/structural-break-main-current/research/reports/rt1320_promotion/PHASE1_STAGE2.md`
- `canonical` — `84b2341cbd9e610fa8517310380b9644b64237f032330ecac5f5809b50fd0741` — `/path/to/workspace/structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student_confirm_s20260901/E2_E1_addition_contract.json`
- `alt1` — `7d45a8a797c97c870a308399f06cbe6cbbf44773673f86da8074195d5be96db5` — `/path/to/workspace/structural-break-main-current/research/reports/rt1320_promotion/PHASE1_ALT1_E2_E1_addition_contract.json`
- `alt2` — `dc7055fbacbc972cb8d4967d5f90e0dbf55d7a9eb69ba8c4916368207cb86569` — `/path/to/workspace/structural-break-main-current/research/reports/rt1320_promotion/PHASE1_ALT2_E2_E1_addition_contract.json`
- `alt3` — `7538d844c56b858edfa7058c4bbb094628e6dbe1795f7e8a6eb2f88b3eca1f8a` — `/path/to/workspace/structural-break-main-current/research/reports/rt1320_promotion/PHASE1_ALT3_E2_E1_addition_contract.json`

## Interpretation

Most incumbent-like additions fail their matched-clone test, so the local conventional-stream ceiling is high. But Q3 asks the stronger legal-causal question. RT1320 uses the same information set and clears the frozen noise floor with fold and partition consistency. Q3 is closed as **CEILING DEAD**.
