# Own data: how it was recorded and how it is used

All real data in this project was recorded by the author. The system trains on sim clips only; the recordings measure how well it transfers to real human demonstrations, and they set how the sim was built. The recordings are not published; the trained weights and the recording guide ([own_recordings.md](own_recordings.md)) let others run the system on their own takes.

![Three uncropped phone frames: head phone and second phone at the same moment on the green table, and the head phone in the outdoor scene](media/own_data_frames.jpg)

## Recording

| | |
|---|---|
| Object | One light-wood block, 86 × 62 × 38 mm, standing on its small end |
| Scenes | An indoor lime-green table; an outdoor dark table with a brick wall (real test set only) |
| Head phone | Behind the demonstrator's shoulder, looking steeply down at the block |
| Second phone | Low, near table height, anywhere except behind the demonstrator; moved between sessions |
| Video | Landscape, 848 × 478, 30 fps |
| Sync | One clap per take; the two videos are aligned by audio |
| Take | One task: move left / stay / move right, combined with roll left / no roll / roll right (a quarter turn) |
| Labels | Movement and roll are judged from the demonstrator's point of view and confirmed by the demonstrator for every take |

## Takes

| Set | Date | Takes | Tasks covered | Used for |
|---|---|---|---|---|
| Early takes | 9–10 Oct | 14 | Moves, some rolls, angled starts | Debugging the pipeline only; not in any reported result |
| Real development set | 11 Oct | 16 (17 recorded, 1 deleted: the block moved during a "stay") | 5 of 9: move left, move right, stay, roll left, roll right | Design decisions (below) |
| Real test set | 14 Oct | 18 (5 outdoor) | All 9 | One final evaluation |

## How the recordings are used

### 1. They drive the simulated hand

Each real clip goes through the same steps as a sim clip: SAM3 masks the person, the frame is cropped around the block's start, Cosmos-Embed1 and the intent head turn 8 frames into *z*, and the policy turns *z* into waypoints that the simulated hand executes in MuJoCo. The hero GIF in the README shows one real take end to end.

### 2. Real test set

- Recorded on 14 Oct, after the pre-registered system had been frozen (13 Oct).
- Labels were confirmed before any score was computed.
- The pre-registered system scored 39%; the final system, chosen on non-test data only, scores 44% (chance 11%).
- The outdoor scene does not lower the score.

### 3. Real development set

The head's sim validation scores saturate near 100%, so they cannot separate design options. These 16 takes did. Each change had an adoption rule written before it ran, typically: at least 6 points better on these takes, and no more than 3 points worse on any sim measure.

| Decision | Score on these takes | Result |
|---|---|---|
| Training length | Fixed at 100 epochs (leave-one-take-out over the 16 takes) | Adopted |
| Policy: the block start alone sets the fine offset | Robot success 38% → 62.5% (mean of 5 policy seeds) | Adopted |
| Remove the two auxiliary classifiers and the clip vector | No loss with the new policy | Adopted |
| Frame order in the head (1D convolution) | Task 68% → 52% | Rejected |
| DINOv2-large instead of Cosmos-Embed1 | 61% vs 65% | Rejected (Cosmos kept) |
| Real *z* in policy training | −19 points (policy seed 0) | Rejected |
| LDA to 8 numbers before the policy | +2.5 points (needed +6) | Rejected |

Because these choices were made on them, the development-set score (64%) is optimistic; the test set gives the transfer number.

### 4. They shaped the simulation

| Sim setting | Taken from the recordings | Why it matters |
|---|---|---|
| Block size | The measured real block | The grip and roll physics depend on the block's proportions |
| Cameras | Head camera 50–75° down in front of the person; free cameras anywhere except behind the person | Sim viewpoints cover where the phones stood |
| Frame and crop | The same 848 × 478 frame and the same crop code for sim and real | Identical preprocessing on both sides |
| Crop centre | Centred on the block's start | In the early takes the block started off-centre and the offset gave away the direction |
| Agent mask | Covers the whole person: hand, arm and body | The body is visible in every real clip |
| Robot grip | The hand grips from the demonstrator's side, as the demonstrator does | With the earlier grip, the masked roll-left silhouette looked like a human roll right: a sim-only model said "roll right" for 20 of 21 real head-phone rolls |

### 5. Control: training on the real takes

A control head was also trained on the development takes, to check whether real training data would help.

- **Sim only (the system):** 47% task accuracy on the real test set (nearest-clip readout).
- **Sim + the 16 development takes:** 35%.
- **Sim + the takes + 54 Cosmos-transferred versions** (each approved by the author after watching the full video): 40%.
- **Cause:** the second phone stood on the other side of the table for 8 of the test takes. Heads trained on real takes learned the training phone's position and flipped left/right on those takes (movement correct 17% vs 67% for the sim-only head).
