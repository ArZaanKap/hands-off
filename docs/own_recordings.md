# Running the trained system on new recordings

Two phones film a person moving a wooden block. The system masks the person out, infers the task and the simulated hand performs it. The run compares that outcome with the task that was actually demonstrated.

- **Needed:** an NVIDIA GPU (tested on a DGX Spark, GB10, CUDA 13), Linux, [uv](https://docs.astral.sh/uv/), `ffmpeg`, a Hugging Face account.
- **Not needed:** the sim training set or any of the original recordings. The trained weights ship with the repository in `weights/` (head seed 0, 9 MB).
- **Time:** the one-off setup is mostly downloads (several GB), then about 1 min per take.

All commands run from the repository folder (`hands-off/`).

---

## Step 1. Install (once)

```bash
git clone https://github.com/ArZaanKap/hands-off.git
cd hands-off
bash scripts/setup.sh
```
- Clones the code, then builds the three Python environments, the hand model and the backgrounds.
- `setup.sh` stops at the SAM3 download (3.3 GB) on the first run. SAM3, the masking model, is gated:
  1. Request access at https://huggingface.co/facebook/sam3 (usually approved within minutes).
  2. Log in, then run setup again:

```bash
.venv-mask/bin/hf auth login
bash scripts/setup.sh
```
- `hf auth login` asks for a Hugging Face access token (Settings → Access Tokens, "read" is enough).
- Re-running `setup.sh` skips everything already installed.
- Cosmos-Embed1, the video model, downloads by itself the first time it runs.

## Step 2. Record

**Kit**
- A wooden block, about 8.6 × 6.2 × 3.8 cm, standing on its small end.
- A plain table.
- Two phones, filming in landscape at 30 fps.

**Phones**

| Phone | Position | Example |
|---|---|---|
| Head phone | Behind the demonstrator's shoulder, looking down at the block (about 45–60 cm away, 50–75° down) | roughly what the demonstrator's own eyes see |
| Second phone | Low, near table height (about 12–28 cm up, 30–45 cm from the block), anywhere except behind the demonstrator | across or beside the table |

- Keep the block near the centre of both views and fully visible throughout.
- Place the block square to the table edge at the start.

**One take = one task**
1. Start both phones.
2. Clap once. The clap synchronises the two videos.
3. Do one of the 9 tasks below, then take the hand away.
4. Stop both phones.

| Movement (about 15 cm) | Roll (a quarter turn) |
|---|---|
| move left / stay / move right | roll left / no roll / roll right |

- Left and right are from the demonstrator's point of view.
- "Roll left" means the top of the block ends up pointing to the demonstrator's left.
- Example: "move left + roll right" = pick the block up, carry it about 15 cm to the left, and set it down lying on its side with the top pointing right.

## Step 3. Put the videos in place

Create a folder inside `data/raw/` and copy each take's two videos into it, numbered from 1:

```
hands-off/
  data/raw/my_takes/
    arz1.mp4    take 1, head phone
    del1.mp4    take 1, second phone
    arz2.mp4    take 2, head phone
    del2.mp4    take 2, second phone
```
- `arz` = head phone and `del` = second phone (the prefixes of the original recordings). The same number means the same take.
- The folder must be inside `hands-off/`. `data/` is listed in `.gitignore`, so the recordings are never committed.

## Step 4. Synchronise the phones

```bash
.venv-spider/bin/python -m handsoff.real.prepare sync --raw data/raw/my_takes --review results/test_inputs
```
- Finds the time offset between the two phones from the clap (about 5 s per take). Take numbers start at 101, so take 1 becomes T101.
- Writes, per take:
  - `results/test_inputs/T101_both_phones.mp4`: both phones side by side, aligned.
  - `results/test_inputs/T101_motion.jpg`: 24 frames from each phone.
- Writes `data/raw/my_takes/labels.json`, a form with one entry per take.
- Example output: `T101: offset -0.40 s, envelope score 0.80, window 12.1 s`.

## Step 5. Watch each video and fill in the labels

Open every `results/test_inputs/T*_both_phones.mp4` and check the two phones show the same take. Then edit `data/raw/my_takes/labels.json` and fill in four fields per take.

Before:
```json
{ "pair": 101, "head_source": "data/raw/my_takes/arz1.mp4", "second_source": "data/raw/my_takes/del1.mp4",
  "translation": null, "roll": null, "label": null, "user_confirmed": false, "notes": "" }
```
After:
```json
{ "pair": 101, "head_source": "data/raw/my_takes/arz1.mp4", "second_source": "data/raw/my_takes/del1.mp4",
  "translation": "left", "roll": "left", "label": "move left + roll left", "user_confirmed": true, "notes": "" }
```

| Field | Allowed values |
|---|---|
| `translation` | `"left"`, `"stay"`, `"right"` |
| `roll` | `"left"`, `"upright"` (no roll), `"right"` |
| `label` | any short description |
| `user_confirmed` | `true` once the video has been checked |

- The label is the ground truth the robot is scored against. It is never shown to the model.
- A take with `user_confirmed: false` is refused in the next step.

## Step 6. Run

```bash
bash scripts/run_test.sh data/raw/my_takes
tail -f results/test/chain.log
```
- The first command starts the run in the background. The second follows its progress (Ctrl+C stops following, not the run).
- The run does, in order:
  1. **Rules:** checks the code and labels against the design rules (`docs/decisions.md`).
  2. **Inputs:** for each phone, 8 frames are taken, cropped around the block, and the person is masked grey (SAM3). This is the step that uses the GPU.
  3. **Embed:** Cosmos-Embed1 turns each masked clip into features.
  4. **Intent:** the trained head turns the features into the 64-D intent vector *z*.
  5. **Robot:** the policy turns *z* into a hand motion, which runs in MuJoCo physics. The block's measured movement and roll are compared with the label.
- The log ends with `DONE` when finished.

## Step 7. Read the result

At the end of `results/test/chain.log`, one line per clip (example from two real test takes):
```
  take 101 head    demonstrated: move left + roll left  robot did: move left + roll left  success
  take 101 second  demonstrated: move left + roll left  robot did: move right + roll right fail
  take 102 head    demonstrated: move left + roll right robot did: move left + roll right success
  take 102 second  demonstrated: move left + roll right robot did: stay + roll right      fail
```
- **success** = the right movement and the right roll, with a clean, gentle placement.
- Per-clip details are in `results/test/seed0/runs_real.json`, and totals in `results/test/seed0/report.json`.
- **Reference:** the shipped weights (seed 0) scored 36% on the real test set (3 seeds: 36–53%; chance 11%). "Did it move" and "did it roll" are usually right. Roll *direction* is near chance on real video (README, Failure case).

---

## Running again

- **New takes:** put each new batch in a new folder and give it its own output folders. Adding takes to a folder that has already run is refused.
```bash
.venv-spider/bin/python -m handsoff.real.prepare sync --raw data/raw/batch2 --review results/batch2_inputs
bash scripts/run_test.sh data/raw/batch2 results/batch2 results/batch2_inputs data/real/batch2
```
- **Own trained weights:** after `bash scripts/train.sh` (needs the sim training set, README: Reproduce), run `WEIGHTS=results/final bash scripts/run_test.sh data/raw/my_takes`. This uses all 3 seeds.

## Common errors

| Message | Fix |
|---|---|
| `Unpaired files: head only [3] ...` | `arz3.mp4` has no `del3.mp4` (or the reverse). Every take needs both. |
| `T101: label not confirmed by the user (R09)` | Set `"user_confirmed": true` for that take in `labels.json` (Step 5). |
| `... changed since sync; rerun sync` | A video was replaced after Step 4. Repeat Step 4. |
| `Inputs/settings changed; use a fresh output directory` | The takes changed since the last run. Use new output folders (see Running again). |
| SAM3 fails to load (`facebook/sam3` not found) | SAM3 was not downloaded. Repeat the login and setup in Step 1. |
| `Initial block not detected; inspect before proceeding` | SAM3 found no wooden block at the start of a clip. Re-record with the block nearer the centre and fully visible. |
