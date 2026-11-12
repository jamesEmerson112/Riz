Arm data from the Naive Nous YC hackathon, Sun Nov 8 2026 (copied from the arm-sim worktree's gitignored runs/).

gentest/                 the chips generate-and-test runs on the real SO-101 (red bag into the box)
  hermes-trials.jsonl    one row per try (Astra-guided): recording, pan, speed, effort, grip, jaw tip and joints
                         at close/open, Astra's diagnosis, bag estimate and verdict, probe results, re-grab attempts,
                         frame sequences
  trials.jsonl           the first fixed run (6 tries)
  live-1211.log          the loop's full log (every plan and result, 11:54 to 14:38)
  astra/                 the exact prompts sent to Astra and its raw replies (from 13:55)
  *.jpg                  camera frames: before, jaws close, probe, after, sampled every 4 s; *-grid* = what Astra saw
  *-sheet.jpg            timed frame sheets of whole tries
  gallery/               one GIF per try, index.html to browse them
  context.json           the last structured context Astra was given
runs-other/              the rest of runs/: earlier run records, dispatcher and event logs (Nov 7-8)

Best try: 13:50:27, recording_2, pan -4, speed 0.5: carried three bags 20.7 s (never opened its jaws then).
Code: github.com/jamesEmerson112/Naive-Nous-YC-Hackathon, prototypes/arm-sim/armsim/gentest.py (main at 644b748).
