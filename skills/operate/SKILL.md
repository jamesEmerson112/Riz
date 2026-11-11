---
name: operate
description: How to run the SO-101 arm with the so101-arm tools: the pick sequence, reading results, and the healing playbook when the watchdog parks the arm.
version: 0.1.0
metadata:
  hermes:
    tags: [robotics, so-101, safety]
---

# Operating the SO-101 arm

## When to use
The user asks you to move the arm, pick something up, play a taught motion, or find out why the arm stopped.

## Rules
- You choose skills; you never send joint angles. If a skill can't do what the user wants, say so.
- Before moving, call `arm_status`. Don't move while an incident is open, a servo reports an alarm, or the arm is frozen: the guard will block you anyway, and its message says the way out. An overload (status bit 32) usually means the jaws pressed on the block: `arm_park` clears it; check `arm_status`, then carry on. Any other alarm: park and report it.
- No tool unfreezes the arm: a person does, with `/arm unfreeze`. Never ask the user to unfreeze it so that you can keep going, and never reach the arm host another way (a terminal, its socket, its command line) to get around a freeze, an incident or the guard. Tell the user why it stopped.
- Report what the tools returned, not what you hoped. "arrived: true" means the joints reached the pose; it does not mean the task succeeded. Check `grasp.holding` after a grip or a lift.

## Picking up the block
1. `arm_go_home`
2. `arm_pick_up` (it runs approach, grip and lift, and stops at the first step that fails), or the three steps yourself: `arm_approach`, `arm_grip`, `arm_lift`.
3. To put it down: `arm_release`, then `arm_go_home`.

If the grip closes on nothing, the block isn't where the arm was taught to find it. In the sim, `arm_pick_up` measures where the block went and its `hint` names the retry. Otherwise:
1. `arm_release`, then `arm_look`.
2. Read its result:
   - `block_mm` is where plain code measured the block's centre on the map.
   - `to_pick_it` holds the `arm_pick_up` call that aims there.
   - Check both against the image. The grid's numbers are the `dx_mm`/`dy_mm` to pass: up is forward (+dx), left is +dy, and the red **+** is where the arm aims now.
3. Rehearse (sim): `arm_rehearse` with a few aims around `block_mm` (or none, to let it choose). Simulated copies of the arm try each one; the main arm doesn't move. When the arm is noisy, each aim is tried several times (`trials`) and `best` is the most reliable aim; otherwise it's the aim that held nearest the block.
4. Retry: `arm_pick_up` with that `dx_mm`/`dy_mm`. Each is limited to 40 mm, and the host refuses more.
   - If `arm_look` says `reachable: false`, don't try. Tell the user to move the block back near the marked spot.
   - If it says `holding: true`, the block is already in the gripper: `arm_release` puts it down.

After two failed corrections, stop and tell the user what you saw.

## When the watchdog parks the arm
The arm host checks every reading in plain code. On a fault it parks the arm and opens an incident; motion stays blocked until a repair passes a check at home.

| Incident kind | What it means | What to do |
|---|---|---|
| `us_frozen`, `us_stuck` | the distance reading stopped changing | `arm_restart_sensor`, then retry the task |
| `us_scale`, `us_offset`, `us_residual` | the distance disagrees with the arm's pose | `arm_restart_sensor`; if it comes back, tell the user the sensor needs a look |
| `us_noisy`, `us_missing_echo` | the sensor jitters or gets no echo | `arm_restart_sensor`; then `arm_handoff` if it persists |
| `joint_stall` | a joint stopped short of its target | `arm_check` once. If it still fails, the watchdog hands it to a person: tell the user which joint, and stop |

After a repair, read the result: `verified: true` means the incident closed, and its `next` line says what to do. When a move is stopped, the motion tool's own `next` names the one repair to try. Then go home and retry the task once.

## Handing off
Use `arm_handoff` with a short reason when the tools can't fix it. The arm stays parked until the person fixes it and runs `/arm check`. Tell the user exactly what failed and what you tried.
