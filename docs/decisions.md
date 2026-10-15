# User rules register

Every rule or decision the user has given about data, simulation, training or evaluation, in one place.
**This file is the source of truth.** Code comments, memory notes and old scripts are not.

How it works:
- **When the user states a rule, add it here in the same turn**: ID, date, the rule in plain words, why, and how it is enforced.
- If a rule can be checked by code, it gets an `auto:` check in `scripts/check_rules.py`. The checker refuses to run if any `auto:` rule here has no matching check.
- Every chain runs `scripts/check_rules.py` before GPU-heavy or training stages (for example `scripts/run_sim_v2.sh`). A failed rule stops the chain.
- When building a new dataset or script, take settings from **this file**, not from another script. The 12 Oct grid copied a benchmark script's fixed camera and silently dropped R03.
- `manual:` rules can't be checked by code. Read them before each step and confirm them in the checkpoint.

| ID | Date | Rule | Why | Enforced by |
|---|---|---|---|---|
| R01 | 11 Oct | Every Cosmos-generated video needs the user's full-video approval before any training/evaluation use, including cached features. Pending clips are excluded; automatic checks are not approval. | Generated videos can silently change the task | auto: real_cosmos_approved |
| R02 | 11 Oct | Roll is about the block's initial local forward axis; translation stays in the table/person frame. Never relabel legacy world-frame data. | Agreed task definition | auto: sim_roll_frame |
| R03 | 10 Oct | Free sim cameras go anywhere around the table **except 150–210° behind the person** (MuJoCo azimuth 30–330), **spread over that range**, not one fixed spot. | Real phones stand at varied sides, including across the table | auto: sim_free_cameras |
| R04 | 12 Oct | Sim cameras behave like the phones: an 848×478 landscape frame, then **the same crop code as the real clips** (`capture/clip_framing.py`). Block tracks use the same `features()`. | The same preprocessing on both sides | auto: same_crop_code |
| R05 | 12 Oct | The head camera leans over the table, 50–75° down (the sim reaches 64°; see R06), always in front of the person's head. | Real head takes look steeply down | auto: sim_head_camera |
| R06 | 12 Oct | The block stays fully in view through all 9 motions of each start, and the hand hides ≤10% of it in each motion's first frame. | The user framed the real phones that way; the crop centres on the first-frame block | auto: sim_block_in_view (first-frame visibility is enforced when cameras are drawn) |
| R07 | 12 Oct | The new training embodiment is a floating Inspire hand **without a forearm**. Mounting it on an arm is secondary. | Decided core embodiment | auto: sim_no_forearm |
| R08 | 11–12 Oct, changed 12 Oct | Real data is split by whole take; both phones and all Cosmos versions inherit the take's fold. **Every approved recorded take (D01–D17) and its approved Cosmos clips train** (`results/real_split_v2/real_training_split.json`). **The test set is the new recordings of 14 Oct** (T101–T118), labels user-confirmed, covering every task type; they never train and never pick a model. Takes numbered 100+ that are in no fold (new recordings, e.g. someone else's) pass: the final head is sim-only. | The user wants all approved data in training and a fresh test set (12 Oct) | auto: real_split |
| R09 | 12 Oct | Every real label is confirmed by the user before training or evaluation. | Labels are the ground truth | auto: real_labels_confirmed |
| R10 | 10 Oct | Real crops are centred on the block's start (the sim has the block start at the centre). | An off-centre start leaked the answer (E16) | auto: same_crop_code |
| R11 | 9 Oct | One GPU job of ours at a time (EGL rendering, ovrtx, embedding and training each count). Other users' light jobs (<12 GB) may run alongside. | Two crashes under overlapping load | manual: chains run stages sequentially |
| R12 | 10–11 Oct | Keep the user involved: before long runs, show exact frames (originals, generated, masked inputs) as plain JPG/MP4 paths, never HTML, and get an OK. Every step gets a concrete visual or numeric example under `results/`. | User reviews everything | manual |
| R13 | 11 Oct | Intent arithmetic means task composition: translation and roll are separate; exact distance is secondary. | Agreed research question | manual |
| R14 | 12 Oct | No hacks: one clean, controlled experiment, with the same clips, settings and seeds for every compared method, and failures reported. | User priority | manual |
| R15 | 9 Oct | Masks cover the whole person (real: person + hand + arm; sim: hand and person proxy). | The body is always in view | manual |
| R16 | 12 Oct | **Sim training images are photoreal** (ovrtx: random HDRI room, table, block and floor materials, as in v4/v5), never the flat MuJoCo looks. Flat renders are only a diagnostic or baseline. | Real videos are wood on a coloured table in a room; flat red-on-white leaves a large appearance gap. The 11–12 Oct grids switched silently | auto: sim_photoreal |
| R17 | 12 Oct | **Train on every sim camera.** The trainer's old held-out angle sectors (`train_intent.HELD_OUT_AZ/MID`, 280–330° and 190–215°) are off for new runs (`train_composition_head.py --held-out-views none`, the default); `sectors` is only for reproducing runs before 12 Oct. Unfamiliar viewpoints are tested by the real test clips. | The sectors hid 32 of 108 v2 free phones (most of the person's near-left side) from training; they were set for the old 110–330° camera range (`results/held_out_cameras/camera_map.jpg`) | auto: trainer_all_views |
| R18 | 12 Oct | **Real training takes teach with their user-confirmed task labels** (`train_composition_head.py --real-loss labelled`, the default): each real clip, and in the Cosmos stage each approved Cosmos version, sits with same-task sim clips and apart from other tasks. Only training-fold takes are used; validation/test takes never. `paired_views` is only for reproducing runs before 12 Oct. A sim-only run stays as the baseline. **15 Oct: the final head is sim-only** (training on the labelled takes lowered accuracy on new recordings, 47% → 35%), so the trainer has no real-video loss. | The unlabelled "two phones must agree" loss collapsed every real clip to "roll left" (E40) | auto: trainer_real_labels (now: no real-video loss) |
| R19 | 12 Oct | **The model used is the final epoch of a fixed training schedule** (`train_composition_head.py --select last`, the default), set before seeing any test clip. `best` (pick the epoch by validation score) is only for reproducing runs before 12 Oct. | Sim validation saturates at 100% and no real validation takes remain (R08), so picking an epoch would be noise (E40 picked epoch 2–10 from 4 clips) | auto: trainer_fixed_epochs |
| R20 | 12 Oct | **The robot hand grips like the user, from the user's side**: for every task the wrist is on the person's side and the fingers point away from the person (never reaching like someone sitting opposite). Roll left uses its own human-like grasp (`clearance_side_human` motion, `left_params` in the grip record), warm-started from the user's idea: the roll-right grasp turned 180° about the forward axis. | The old roll-left grip was the roll-right grip turned 180° about vertical; its grey silhouette mirrors a human roll, so a sim-only model said "roll right" for 20/21 real head-phone rolls (`results/roll_direction_check/`) | auto: sim_human_like_grip |

## Open decisions (not rules yet)

- None. (The fixed schedule is 100 epochs for the final head, R19.)


## Deferred (agreed to do later)

- **Head camera up to 75° down** (user, 12 Oct). It needs the robot hand to wait somewhere other than above the block, which means new robot motions and new physics checks. Not before the 16 Oct deadline unless time allows.

## Changed rules (history)

- **R08 before 12 Oct evening:** 7 train / 2 validation / 7 test takes (`results/hand_only_v1/real_training_split.json`). New translated-roll recordings were test-only, human left-upright takes never trained (so "move left" on a person was learned from the robot sim only), and the test set covered every task type. The user changed it: all recorded takes train, and the 13 Oct recordings are the test set. E40 and earlier used the old split.
