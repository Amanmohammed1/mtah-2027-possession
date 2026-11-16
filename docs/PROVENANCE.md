# Parameter provenance and decision rules

Written on 2026-11-16, after the organisers' clarification of the rules, and committed **before**
any of the re-tuning below was run. It fixes where every number comes from, so that nothing is
chosen by its score on the public ground truth, by comparison with the video, or by closeness to an
earlier submission.

## Rules we follow

- The pipeline never reads the ground truth (`evaluation/gt/`). `run.py` must work with that folder
  removed.
- No hand labels, no data created manually on the provided game, no checks of our output against
  the video.
- Every tuned parameter is tuned on external data. Everything else is fixed a priori from the Laws
  of the Game or physics, or takes its parameter-free option.
- The public five-minute window is scored once, at the end, and reported as it comes out. The only
  change it may trigger is a correctness fix of the kind the FAQ names (frame offset, H/A swap,
  file format), and any such fix is reported.

## Sources

| parameter | value | source |
|---|---|---|
| ground truth in the dead-ball model | none | removed: the HMM is initialised from tracking-only pseudo-labels (ball seen outside the lines = dead; ball seen moving fast well inside the lines, away from those episodes = alive) and fitted by unsupervised Baum-Welch on the match's own automatic observations; no labels enter |
| player-speed bins | regular 0.5 m/s grid, missing speed its own bin | a priori |
| speed-trend bins | regular 0.5 m/s grid, symmetric | a priori |
| per-channel evidence cap | none | parameter-free option (the earlier cap of 2 was set on the public window and the video) |
| minimum stoppage | 5th percentile of ball-dead run lengths in IDSSE | external data |
| stoppage-onset pull-back | 0 | parameter-free option (the evidence lag cannot be measured on external data, which has no broadcast) |
| seen line crossing | ball ≥ 0.3 m beyond a line for frames f+3…f+7 | Law 9 (ball wholly over the line): ball radius 0.11 m plus tracking error |
| unseen exit | last sighting within 3 m of a line, moving out faster than 2 m/s, then unseen for 10 frames; crossing extrapolated from the outward speed (≤ 1 s) | physics: a ball rolling out at ≥ 2 m/s covers 3 m in ≤ 1.5 s |
| live-play guard for exits | players' 90th-percentile speed in the 2 s before an exit above T, with T the 10th percentile of that statistic over IDSSE ball-alive frames | external data |
| restart kick | ball below 2 m/s, then above 5 m/s, within ±2 s of the predicted restart | Laws 13/16/17 (the ball is stationary at a restart); 2 m/s = tracking jitter of a stationary ball, 5 m/s = the slowest pass |
| restart team: zones | penalty area from Law 1 (16.5 m × 40.32 m); corner = within 3 m of the goal line and 5 m of the touchline (corner arc 1 m plus tracking error and roll) | Law 1 / Law 17 |
| restart team: radius | 2 m | Law 15 (opponents ≥ 2 m from a throw-in; ≥ 9.15 m at free kicks, corners and kick-offs, Laws 13/17/8) |
| restart team: windows | ball position over the 2 s before the restart; nearest player over the 2 s before to 0.4 s after | a priori |
| restart team: crowded box | ≥ 9 players in a penalty area | a priori (more than any team's outfield line at open play) |
| kick-offs | first half: team of the player nearest the centre spot; second half: the other team | Law 8 |
| graded halves, second-half kick-off (frame 113742) | as published | DATA.md, FAQ.md |
| possession control radius R and frames k | grid R ∈ {0.75, 1.0, 1.25, 1.5, 2.0} m × k ∈ {3, 5, 8, 12, 16, 25} frames | external data, see below |
| short live slivers with no team | < 75 frames → dead | a priori (equal to the 3 s used as the a priori minimum stoppage) |
| ball cleaning (40 m/s jumps, < 3-frame runs, 12-frame gap fill) | unchanged | a priori physics |
| Smart Tagging window (−50/+150 frames around a set-piece start) | unchanged | a priori |
| broadcast offset | computed | automatic, from the two tracking files |

## External tuning protocol (fixed in advance)

Data: IDSSE (Bassek, Rein, Weber & Memmert 2025, *Scientific Data* 12:195,
doi:10.6084/m9.figshare.28196177, CC BY 4.0; data owner DFL). Matches J03WMX, J03WN1, J03WPY.
Per-frame labels: `D` where BallStatus = 0, otherwise `H`/`A` from BallPossession.

For R and k the alive/dead mask and the team at each restart are taken from the IDSSE labels, so only
the in-play possession rule is scored. Two conditions: the clean tracking, and a degraded copy
(ball unseen on 10% of alive frames in 1–3 s blocks, Gaussian position noise σ = 0.3 m on players and
ball, fixed random seed). Score: the official `spell_f1_mean` from `evaluation/possession_metrics.py`.
**Selection: the (R, k) with the highest mean spellF1 over the 3 matches × 2 conditions; ties broken by
macroF1.**
