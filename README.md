# Hands Off: Learning Task Intent from Multi-View Video

**I record myself moving a wooden block with two phones. A robot hand in simulation then does the same *task*.** It does not copy my hand motion. The person is masked out of every frame, so the model has to read the task from what happens to the block.

![You do a task, the model reads it, the robot hand does it](docs/media/hero.gif)

<sub>A training take (D15) read by a model trained only on simulation, which never saw a real video. The held-out test on new recordings is below.</sub>

| | |
|---|---|
| **Input** | Two ordinary phone videos (head-mounted + handheld) of a person moving a block |
| **Output** | One of 9 tasks (move left / stay / right × roll left / none / right), executed by a floating 5-finger Inspire hand in MuJoCo |
| **Model** | Frozen [Cosmos-Embed1](https://huggingface.co/nvidia/Cosmos-Embed1-448p) video features + a 1.2 M-parameter head |
| **Training data** | 2,592 photoreal sim clips + 16 of my own takes (32 phone clips + 54 approved Cosmos-Transfer restylings) |
| **Test** | 18 new takes recorded after the models were frozen: **[pending, 13 Oct]** |

## The idea

A human hand and a robot hand look nothing alike, and neither do my kitchen and a simulator. So I **mask the agent out of every frame** (grey silhouette, both for me and for the robot). The model sees only the block, the table and a grey blob. What is left to learn is *what happened to the object*, and that is the same for a human and a robot. Hence "hands off".

The model maps each clip to a 64-d **intent vector** and is trained so that tasks compose:

![z(A) − z(B) + z(C) lands on the right task, in a different room](docs/media/task_arithmetic.jpg)

<sub>"move right + roll right" − "move right" + "move left" → "move left + roll right". The answer is a different start, in a different room and material. Held-out sim starts; `sim_seed0`.</sub>

The 9 tasks the robot hand can execute, seen from the person's viewpoint:

<img src="docs/media/robot_tasks.gif" width="560" alt="The 9 tasks executed by the Inspire hand">

## Results

**Held-out test on new recordings** (18 takes, 2 per task type, never trained on and never used to choose anything): **[pending: recorded 13 Oct]**

**Fair test on my own takes** (each of the 16 takes held out in turn; 32 clips; seed 0; % correct):

| model trained on | move + roll both right | move right | roll right |
|---|---|---|---|
| sim only (never saw a real video) | 69 | 91 | 78 |
| sim + my takes | 66 | 88 | 78 |
| sim + my takes + Cosmos restylings | 62 | 84 | 78 |
| chance | 11 | 33 | 33 |

**Task arithmetic** (held-out sim starts, answer from a different look; 12 queries per family × 3 seeds):

| family | example | sim only | sim + takes | + Cosmos | C alone |
|---|---|---|---|---|---|
| remove the move, keep the roll | right+roll − right + stay → roll in place | 97 | 67 | 58 | 0 |
| remove the roll, keep the move | right+roll − roll + stay → right | 100 | 100 | 100 | 0 |
| swap the move direction | right+roll − right + left → left+roll | 100 | 94 | 89 | 0 |

**What these say:**
- **Move left / stay / right transfers from sim to real well:** 84–91% on real clips, from a model that never saw one.
- **On real video, the roll direction of an in-place roll is close to a coin flip.** "Rolled vs upright" is reliable; "left vs right" is not (about 7/14 clips).
- **Adding my real takes did not help on held-out takes,** and it weakens the sim arithmetic. 16 takes is a small dataset; the test set decides.
- The head is *trained* with arithmetic on the training starts. The table shows that it generalises to new starts and looks, not that it emerges on its own.

## How it works

```
two phones ──► sync by audio ──► SAM3: mask person/hand/arm grey, find the block ──► crop centred on the block
          ──► 8 frames ──► Cosmos-Embed1 (frozen) + 2D block track ──► head ──► z (64-d)
          ──► nearest simulated clip ──► its task ──► Inspire hand picks, carries, turns and places the block
```

1. **Simulation.** 36 random block starts × 9 tasks: a floating Inspire hand does each task with one fixed pick-carry-turn-place motion and a human-like grip. Every episode passes physics checks (no table penetration, gentle release, no twist). Each episode is filmed by 4 phone-like cameras (one where my head is, three random) and rendered photoreal with NVIDIA's RTX renderer (ovrtx), in 72 random rooms and materials:
   ![Photoreal sim looks](docs/media/sim_looks.jpg)
2. **My data.** 16 takes × 2 phones; I confirmed every label by watching both videos. Cosmos-Transfer2.5 restyles each clip (new table colour and sleeve). I watched all 64 generated videos in full and approved 55; 54 of those also pass an automatic check that the block still does the same task.
3. **Training.** Same-task clips from different rooms pull together, two small readouts predict the move and the roll, and A − B + C must land nearest a clip of the composed task. In the real stages, my labelled clips sit with same-task sim clips. Each stage has a fixed epoch count, chosen by leave-one-take-out before the test. There are 3 stages (sim / + takes / + Cosmos) × 3 seeds.
4. **Acting.** The robot's motion is a scripted primitive per task. What is learned is *which task*, from the video.

## What didn't work

- **The robot gripped like someone sitting opposite.** The roll-left grip came from the far side, so in sim the hand's position gave the roll direction away. The sim-only model called 20 of 21 real in-place rolls "roll right". A human-like grip from the person's side removed that bias, but roll direction on real video is still near chance (above).
- **Unlabelled real-data loss.** Teaching "both phones of a take must agree" collapsed every real clip to "roll left". Labelled takes replaced it.
- **Flat MuJoCo renders** (red block, white table) left a large gap to real video, so all training images are photoreal now.
- **3D block pose tracking** (FoundationPose) failed on the head view, where the hand covers the block. The model uses a 2D block track from SAM3 instead.
- **RL for the robot motion** (a short SAC pilot on CPU): 0/45 complete motions against 30/45 for the scripted reference. Dropped; the motions stay scripted.
- **A rigid robot forearm** hit the table in many motions. It was removed from the physics and the masks; the robot is a floating hand.
- **Cosmos restylings are not always faithful.** In an earlier batch, 3 of 13 takes failed in every look: the block shrank, a second block appeared, or the block was erased. One video passed the automatic end-frame checks but morphed mid-clip. Hence the full-video human review of every generated clip.

## Run it

Tested on an aarch64 NVIDIA DGX Spark (GB10, CUDA 13), Python 3.12, [uv](https://docs.astral.sh/uv/).

| step | command | time |
|---|---|---|
| setup (3 venvs, hand model, HDRIs) | `bash scripts/setup.sh` | downloads |
| build the sim training set | `bash scripts/build_sim.sh` | ~2.5 h |
| sync phone takes for review | `.venv-spider/bin/python -m handsoff.real.prepare sync --raw RAW --review DIR` | ~5 s per take |
| mask, crop, embed the takes | `bash scripts/prepare_real.sh ...` (see header) | ~1 min per take |
| train 9 heads + arithmetic | `bash scripts/train.sh` | ~1.5 h |
| held-out test → robot | `bash scripts/run_test.sh RAW` | ~1 min per take |
| unit tests | `.venv-spider/bin/python -m unittest discover -s tests -t .` | 2 s |

Each script runs detached, writes a log, and stops at the first failed step. `python -m handsoff.rules` checks a dataset against the project's rules (`docs/decisions.md`) before any training.

**[pending: data and model download, recording layout]**

## Repo map

```
handsoff/
  sim/     scene.py (MuJoCo scene) · primitive.py (the 9 robot motions) · grip.py (grasp search)
           dataset.py (sim training set) · render.py, geometry.py (cameras, masks) · ovrtx.py, photoreal.py (RTX looks)
  real/    sync.py (two-phone audio sync) · sam3.py (masks, block) · framing.py (crop, shared with sim)
           prepare.py (takes → model inputs) · cosmos.py (restylings) · review_gate.py (human approval gate)
  model/   embed.py (Cosmos-Embed1) · head.py (intent head) · tasks.py (9 tasks, composition) · data.py
           train.py · choose_epochs.py · evaluate.py (clip → task → robot) · arithmetic.py
  rules.py automatic checks of the design rules
scripts/   one shell script per stage (table above)
config/    grip parameters, confirmed labels, take split, Cosmos review log, track-noise models
tests/     28 unit tests: robot geometry and grip, task composition, trainer leakage and resume, approval gate
docs/      decisions.md (every design rule and why), media/
```
