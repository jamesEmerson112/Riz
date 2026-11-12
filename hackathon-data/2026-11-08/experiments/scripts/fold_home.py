import time
from armsim.real import RealArm
REST = [4.66, -110.07, 93.98, -101.05, -3.38, 1.35]
arm = RealArm().connect()                       # holds the pose it is in, from this moment
for _ in range(arm.fps * 3):                    # 3 s: let go and step back
    arm.send(arm._goal); arm.tick()
cur = [float(v) for v in arm._goal]
print("took over at", [round(v, 1) for v in cur], flush=True)
folded = cur[1] < -95 and cur[2] > 80 and cur[3] < -80
stages = ([("small move onto the exact rest pose", REST, 2.0)] if folded else
          [("up: elbow and wrist straighten, jaws rise", [cur[0], cur[1], 18.0, 54.0, cur[4], cur[5]], 2.5),
           ("base to centre, staying high", [4.7, cur[1], 18.0, 54.0, -3.4, cur[5]], 3.0),
           ("fold onto the rest stops", REST, 3.0)])
for name, pose, secs in stages:
    r = arm.goto(pose, duration=secs)
    print(f"{name}: arrived {r.get('arrived')} (max err {r.get('max_err_deg')} deg)", flush=True)
print("now:", [round(v, 1) for v in arm._q], "| alarms", arm.status(), flush=True)
arm.disconnect()                                # torque off, folded on its stops
