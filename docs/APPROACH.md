# Frame-level possession from automatic tracking — technical note

*More than a Hack 2027, Phase 1. Draft — updated as the method changes.*

## 1. Summary

We split the problem in two, because the two halves fail differently:

1. **Is the ball alive or dead?** A hidden Markov model over the whole match, fed by four weak
   but independent signals: whether the **TV broadcast** is showing live play, what the tracked
   ball is doing, how fast the players move, and the Smart Tagging set-piece windows. Its
   parameters are learned by EM on the full match. Stoppage edges are then sharpened: onsets
   snap to the frame the ball crossed a line, restarts to the frame the ball is struck.
2. **While alive, which team has it?** The team that took the restart holds the ball until the
   other team *holds* it: a player within 1.25 m of the ball, nearer than any opponent, for
   8 consecutive frames (0.32 s). Loose balls, passes in flight and brief contests leave
   possession where it was, as the challenge definition asks.

Public window: spellF1 **0.573**, macroF1 **0.893** (reference baseline 0.326 / 0.621 on the same
window). The public window holds 17 possessions, so one possession moves spellF1 by ~0.06: it is
used as a sanity check, never to choose between variants. Variants were chosen on a validation
set we labelled from the tactical video (section 4).

## 2. What the data told us

- **Most of the missing ball is dead time, not hidden play.** The tactical ball is missing on 41%
  of graded frames, in long gaps (median 10.7 s). Where it is missing, the broadcast has usually
  cut away and the players are walking. In-play blackouts longer than 2 s add up to about
  5 minutes over the match.
- **The broadcast director is a dead-ball detector.** Once aligned, the broadcast shows live play
  on 96–97% of in-play frames but on only 24% of dead-ball frames in the public window: the
  director cuts to close-ups and replays when play stops. After a restart the replay often runs
  on for a few seconds, so the cut-away marks the stoppage well but lags its end.
- **The two recordings share one clock per half.** Matching player layouts (both tracking files
  use the same pitch frame) gives tactical = broadcast + 4014 frames in the first half and
  + 26920 in the second. Replays replace live footage; they do not shift the clock.
- **Possession changes at the receiving touch**, and brief possessions are real (0.8 s and 1.3 s
  spells in the public ground truth), so heavy smoothing cannot be the answer.
- **Effective playing time is about half the graded time** (we predict 49.5% dead ball), in line
  with LaLiga's low effective playing time; the match includes two cooling / medical breaks of
  3–4 minutes and long post-goal stoppages.

## 3. Method

### 3.1 Tracking preprocessing (`possession/features.py`)
Positions to metres from the pitch centre. Ball: remove points that jump faster than 40 m/s away
from both neighbours, drop detections shorter than 3 frames, 5-frame smoothing, linear fill of
gaps up to 12 frames. Per frame: nearest home / away player distance to the ball, player speeds
from positions (the provided speed channel is empty for the ball 91% of the time), team shape.

### 3.2 Broadcast alignment (`possession/broadcast.py`)
For broadcast anchor frames every 2 s, a numba chamfer search over every 5th tactical frame
(same-team nearest-neighbour distance plus the ball), refined to the exact frame. 88% of
confident matches fall within ±2 frames of two offsets; the rest are replays, which match the
earlier moment they show. The offset is therefore one constant per half (median of the matches
near each mode). An earlier rolling-median estimate was dragged by replays during a 3-minute
break and left the broadcast unmapped there; the constant fixed it.

### 3.3 Alive / dead HMM (`possession/alivehmm.py`)
- Observations at 5 Hz: broadcast live / cut / absent; ball invisible / out / still / slow /
  fast; mean player speed (3 s, 7 bins); speed trend (next 3 s minus last 3 s, 5 bins); inside a
  Smart Tagging set-piece window.
- States: alive, and a 15-step chain of dead sub-states (minimum stoppage 3 s).
- Baum–Welch on both halves; the public labels enter as fixed counts.
- Each channel's dead-vs-alive log-ratio is capped at 2 nats per step, so that no single signal
  can veto the others (slow players alone made slow restarts look dead). Lower caps start to
  drop stoppages we verified on video.
- Posterior > 0.5 is dead. Live slivers shorter than 3 s in which nobody ever has the ball are
  folded back into the surrounding stoppage.
- Dead ball before the published second-half kick-off (frame 113742) is enforced.

Validated as a holdout first (EM without the public labels): the public window gives the same
score, so the public labels are not doing the work.

### 3.4 Boundary refinement (`possession/deadball.py`)
The HMM's onset trails the real one, because its evidence (players slowing, the broadcast
cutting away) follows the whistle. The onset is moved to, in order:
1. the earliest frame in the 10 s before where the ball is seen crossing a line and staying out;
2. the frame the ball left the pitch unseen: last sighting within 3 m of a line and moving
   towards it (or already on it), then not seen for 10 frames; the crossing is extrapolated from
   its outward speed. Only while play was live: the fastest tenth of the players must still have
   been running (> 2.5 m/s) in the two seconds before, because after a whistle the ball often
   rolls out while everybody has stopped, and that exit is a consequence, not the cause;
3. otherwise 60 frames earlier, the median lag of the evidence behind the whistle (measured on
   video: 5–150 frames, median ~50).

On the six onsets we could time on video, this puts every one within 17 frames (three were off
by 39–56 frames with the fixed lag alone). The restart moves to the strongest ball kick from rest
within ±2 s of the predicted one.

### 3.5 Team taking each restart (`possession/team.py: infer_restart`)
In order:
1. ball, seen on the pitch just before the restart, at a corner: the team attacking that goal;
   in a penalty area: goal kick or defensive free kick, the team defending that goal;
2. the team of the player within 3 m of the ball (opponents must stand back);
3. ball unseen and a penalty area crowded with ≥ 9 players: a set piece by the team attacking
   that goal;
4. otherwise the first team to control the ball.

Kick-offs follow the Laws of the Game: the team that kicks off the first half does not kick off
the second.

### 3.6 Team in possession during play (`possession/team.py: hysteresis`)
Control = nearest player within 1.25 m of the observed ball and nearer than any opponent.
Possession switches when the other team holds control for 8 consecutive frames, dated to the
start of that control.

## 4. Validation on the video

The ground truth for anything beyond the public five minutes is private, and the rules allow
hand-labelled data for validation. We made a frame-exact 720p copy of the tactical video
(`tools/colab_make_720p.ipynb`, checked frame by frame against the tracking with
`tools/check_alignment.py`) and labelled from it:

| what | how | result |
|---|---|---|
| team taking each restart | 64 restarts labelled (`validation/restarts_visual.csv`) | 52/64 → **61/64** after §3.5 |
| predicted stoppages | middle frame of all 95 covered by the video | all long ones genuine; the short doubtful ones are what the evidence cap removes |
| short stoppages | 11 sequences | throw-ins and fouls kept; slow build-up no longer called dead |
| restart timing | 6 restarts followed for 8 s | play live from the predicted frame in all 6 |
| set pieces | 58 Smart Tagging set pieces | 57 have a predicted restart nearby |
| possession switches | 36 sampled H↔A switches | ≥ 2 s: reliable; < 2 s: about half are duels, scrambles or aerial balls |

The labels never enter the pipeline; they decide between variants.

## 5. What we tried and dropped

- **Nearest player + majority vote** (the reference baseline): flickers; the window that fixes the
  flicker destroys short possessions.
- **Viterbi over per-frame touch evidence**, with and without a ball-free team-shape model:
  under-segments, because the ground truth flips on brief control that any switching penalty
  strong enough to suppress noise also removes.
- **Dead-ball classifier on rule-based pseudo-labels:** learned the labelling rule back (e.g.
  "ball invisible ⇒ dead") instead of the stoppage. The rule itself was wrong on throw-ins and on
  balls skimming the line, where the HMM was right.
- **Broadcast tracking for control:** its near-ball positions agree with the ground truth less
  often than the tactical ones (0.87 vs 0.91).
- **A wider (6 m) nearest-player pass for restarts:** decided few restarts and got one of the
  labelled ones wrong.

## 6. Limitations

- Team errors come from tracking errors near the ball: wrong positions, and occasional identity
  switches that put a player (we saw Barcelona's goalkeeper) in a slot of the other team.
  Jersey colour from the video would fix them; not done.
- Very short possessions in duels and aerial balls are unreliable.
- A restart after a long break can be called late when the broadcast keeps showing replays and
  play resumes slowly (seen once, ~7 s).
- The 60-frame foul-onset correction is a single calibration from three examples.

## 7. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s /path/to/data/bundle data
.venv/bin/python run.py                      # ~40 s on an M1 laptop from scratch, deterministic
.venv/bin/python evaluation/validate_submission.py out/submission.csv
```

The video is needed only for the validation tools, not to produce the submission.
