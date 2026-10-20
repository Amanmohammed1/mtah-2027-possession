# Frame-level possession from automatic tracking — technical note

*More than a Hack 2027, Phase 1. Draft — updated as the method changes.*

## 1. Summary

We split the problem in two, because the two halves fail differently:

1. **Is the ball alive or dead?** A two-state hidden Markov model over the whole match, fed by
   four weak but independent signals: whether the **TV broadcast** is showing live play, what the
   tracked ball is doing, how fast the players move, and the Smart Tagging set-piece windows. Its
   parameters are learned by EM on the full match, initialised from the public five minutes.
   Stoppage edges are then sharpened: onsets snap to the frame the ball crossed a line, restarts
   to the frame the ball is struck.
2. **While alive, which team has it?** The last team to *hold* the ball: possession passes when a
   player of the other team stays within 1.25 m of the ball for 8 consecutive frames (0.32 s).
   Loose balls, passes in flight and brief contests leave possession where it was, as the
   challenge definition asks.

Public window: spellF1 **0.573**, macroF1 **0.884** (baseline 0.326 / 0.621 on the same window).
We do not tune on the public window beyond sanity checks: it holds 17 possessions, and one of
them is worth ~0.06 spellF1.

## 2. What the data told us

- **Most of the missing ball is dead time, not hidden play.** The tactical ball is missing on 41%
  of graded frames, in long gaps (median 10.7 s). Where it is missing, the broadcast has usually
  cut away and the players are walking: the blocks of the match with the worst ball visibility
  are long stoppages. In-play blackouts longer than 2 s add up to about 5 minutes over the match.
- **The broadcast director is a dead-ball detector.** Once aligned, the broadcast shows live play
  on 96–97% of in-play frames but on only 24% of dead-ball frames in the public window: the
  director cuts to close-ups and replays when play stops.
- **The two recordings align with one offset per half.** Matching player layouts (both tracking
  files share the pitch frame) gives tactical = broadcast + 4014 frames in the first half and
  + 26919 in the second, stable to a frame. Replays replace live footage rather than shifting
  the clock.
- **Possession changes at the receiving touch**, and brief possessions are real (0.8 s and 1.3 s
  spells in the public ground truth), so heavy smoothing cannot be the answer.

## 3. Method

### 3.1 Tracking preprocessing (`possession/features.py`)
Positions to metres from the pitch centre. Ball: remove points that jump faster than 40 m/s away
from both neighbours, drop detections shorter than 3 frames, 5-frame smoothing, linear fill of
gaps up to 12 frames. Per frame: nearest home / away player distance to the ball, player speeds
from positions (provided speed channel is mostly empty for the ball), team shape in each team's
attacking frame.

### 3.2 Broadcast alignment (`possession/broadcast.py`)
For broadcast anchor frames every 2 s, a numba chamfer search over every 5th tactical frame (same-
team nearest-neighbour distance plus the ball), refined to the exact frame, then propagated
frame by frame with a local search. The offset is taken as a rolling median of confident matches.

### 3.3 Alive / dead HMM (`possession/alivehmm.py`)
- Observations at 5 Hz: broadcast live / cut / absent; ball invisible / out / still / slow /
  fast; player speed (3 s mean, 7 bins); speed trend (next 3 s minus last 3 s, 5 bins); inside a
  Smart Tagging set-piece window.
- States: alive, and a 15-step chain of dead sub-states (minimum stoppage 3 s).
- Baum–Welch on both halves; the public labels enter as fixed counts.
- Posterior > 0.5 is dead.

Validated as a holdout first (EM without the public labels): the public window gives the same
score, so the public labels are not doing the work.

### 3.4 Boundary refinement (`possession/deadball.py`)
- Onset: the earliest frame in the 10 s before the predicted onset where the ball crosses a line
  and stays out. Without a crossing (fouls, offsides), the onset is moved back 60 frames, the
  measured lag of the evidence behind the whistle.
- Restart: the strongest ball kick from rest within ±2 s.

### 3.5 Team in possession (`possession/team.py`)
Control = nearest player within 1.25 m of the observed ball and nearer than any opponent.
Possession switches when the other team holds control for 8 consecutive frames, dated to the
start of that control. After a restart the first controlling team is back-filled to the restart.

## 4. What we tried and dropped

- **Nearest player + majority vote** (the reference baseline): flickers; the window that fixes the
  flicker destroys short possessions.
- **Viterbi over per-frame touch evidence**, with and without a ball-free team-shape model:
  under-segments, because the ground truth flips on brief control that any switching penalty
  strong enough to suppress noise also removes.
- **Dead-ball classifier on rule-based pseudo-labels:** learned the labelling rule back (e.g.
  "ball invisible ⇒ dead") instead of the stoppage. The rule itself was also wrong on throw-ins
  and balls skimming the line, where the HMM was right.
- **Broadcast tracking for control:** its near-ball positions agree with the ground truth less
  often than the tactical ones (0.87 vs 0.91).
- **Video:** not used yet.

## 5. Limitations

- Team errors come from tracking errors near the ball (wrong positions or team for the nearest
  player); without the video they cannot be corrected.
- The 60-frame foul-onset correction is a single calibration from three examples.
- The full-match statistics (≈51% dead ball, 325 possessions) cannot be checked without labels.

## 6. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s /path/to/data/bundle data
.venv/bin/python run.py                      # ~40 s on an M1 laptop from scratch, deterministic
.venv/bin/python evaluation/validate_submission.py out/submission.csv
```
