# Design choices

Why the system has its current shape. Each choice lists the reason and, where one exists, the measured evidence. Alternatives that were tried and dropped are in [tried_and_dropped.md](tried_and_dropped.md); how the recordings were made and used is in [own_data.md](own_data.md).

## Question and tasks

| Choice | Reason | Evidence |
|---|---|---|
| Infer *what* was done to the object, not *how* | A robot with a different body cannot copy a human's motion, but it can reproduce the outcome | The same *z* drives the floating hand and the arm-mounted hand ([arm.md](arm.md)) |
| 9 tasks = move {left, stay, right} × roll {left, none, right} | Two independent components make composition testable: remove one, add another | Intent arithmetic, README |
| Roll about the block's initial forward axis; movement in the table frame, from the demonstrator's point of view | A roll then means the same turn wherever the block starts, and left/right match how the demonstrator describes the take | Labels in `config/` |

## Embodiment

| Choice | Reason | Evidence |
|---|---|---|
| Floating 5-finger Inspire hand, no forearm | A human-like hand without arm reach limits or inverse kinematics; failures then come from the intent, not the arm | Mounting it on a UR5e gives the same outcome in 536/536 runs ([arm.md](arm.md)) |
| The hand grips from the demonstrator's side, fingers pointing away from the person; roll left uses its own human-like grasp | The masked silhouette should look like a person doing the same task. The earlier roll-left grip was the roll-right grip turned 180°, so its silhouette mirrored a human roll | With the mirrored grip, a sim-only model called 20 of 21 real head-phone rolls "roll right" |
| Grasps searched under randomised block mass and friction (cross-entropy method, `handsoff/sim/grip_robust.py`) | A grasp tuned on one nominal block can rely on that block's exact weight and friction | Nominal-block roll-right grasp: 32% on held-out light or slippery blocks; searched grasp: 100%, and the full system holds its nominal success under every tested change (README) |

## Input video

| Choice | Reason | Evidence |
|---|---|---|
| Two phones: one over the demonstrator's shoulder, one anywhere else except behind the person | A cheap multi-view setup; the head view sees the hand's side, the second phone a different angle | — |
| Mask the whole agent with a grey silhouette (person, arm and hand in real; robot hand and a person proxy in sim) | Human and robot clips then differ only in the silhouette's shape, which closes most of the embodiment gap | The grip still leaks roll direction in sim (probe 79%); no fill tried removed it (tried_and_dropped.md) |
| Sim cameras placed like the phones: free cameras at any azimuth outside 150–210° behind the person, spread over that range; the head camera 50–75° down, in front of the head | Real phones stand at varied sides and the head phone looks steeply down; one fixed camera would not cover them | Enforced by `handsoff/checks.py` |
| One crop code for sim and real (848 × 478 phone frame), centred on the block's start | Identical preprocessing on both sides. An off-centre start gave the answer away: the crop offset alone predicted the movement | Enforced by `handsoff/checks.py` |
| Photoreal sim (RTX rendering: random room, table, block and floor materials) | Real clips show wood on a coloured table in a room; flat MuJoCo colours leave a large appearance gap | Not ablated in isolation. After the head, a clip's nearest neighbour shares its scene in 3% of cases (100% before) |

## Model

| Choice | Reason | Evidence |
|---|---|---|
| Frozen Cosmos-Embed1 video encoder | A video-text encoder from NVIDIA's Cosmos world-foundation-model platform, trained for physical-AI video; frozen, so only a small head trains | DINOv2-large tied (development set 61% vs 65%); Cosmos was kept by a rule fixed before that run |
| A 2D block track as a second input | Cosmos features lose where the block is | Position error 6.7 cm → 2.6 cm |
| A small head (0.84 M parameters) to a 64-D unit vector *z*, from the first, last and mean frame features | Few parameters for a small dataset; frame order fitted sim-specific motion | 1D convolution over frames: development set 68% → 52% |
| Contrastive loss plus an arithmetic loss z(a) − z(b) + z(c) → target task | Contrastive alone made *z* a lookup table that could not compose | Arithmetic queries: 3% with contrastive alone, 33/36 with both |
| Fixed schedule: the final epoch of 100, chosen before any test clip | Sim validation saturates at 100%, so picking an epoch would fit noise | 100 epochs chosen by leave-one-take-out on the development set |

## Training data

| Choice | Reason | Evidence |
|---|---|---|
| Head and policy train on sim only; real takes are for evaluation | Tests whether a sim-trained intent transfers to real people. Real training is kept as a control | Adding the labelled development takes: real test set 47% → 35% (nearest-clip readout) |
| Cosmos-transferred real takes only as a control, each generated video approved after watching it in full | Generated videos sometimes changed the block's shape or texture, which could change the task | With them: 40%, below sim-only |

## Policy

| Choice | Reason | Evidence |
|---|---|---|
| *z* and the block start choose one of 16 motion bins (k-means on teacher plans), output 14 hand waypoints | Choosing first avoids averaging between tasks | Direct regression: real test set 33% |
| The fine offset comes from the block start only | An offset from *z* made results depend on the policy seed | Development set 38% → 62.5% (5 policy seeds) |
| Open-loop execution in MuJoCo physics; success = the block's final state matches the task | Simple and repeatable; the outcome is judged on the object, as in the question | Limitations, README |

## Evaluation

| Choice | Reason | Evidence |
|---|---|---|
| Real test set recorded after the system was frozen; its result reported once, beside the pre-registered system | Keeps the test set out of every design decision | Pre-registered 39%, final 47% (post hoc) |
| Every method compared on the same clips and settings; 3 head seeds with ranges | Seed spread is large on 36 real clips | Results summary, README |
| Report per question (moved, which way, rolled, which way) | Shows where transfer fails | Roll direction near chance on real (54%) |
