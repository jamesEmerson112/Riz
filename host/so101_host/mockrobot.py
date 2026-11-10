"""A stand-in for LeRobot's SO101Follower, enough to exercise RealArm without hardware (R1 tests).

Joints chase their goal with a ~80 ms lag while torque is on. The gripper stops on a block at
`block_at` (0-100 scale) and its load grows with how far the goal is past the block, saturating at the
servo's limit. A joint in `stuck` never moves (a dead servo). Register names follow LeRobot's Feetech
table; `Present_Load` is sign-magnitude raw 0-1000, like the STS3215.
"""
import time

import numpy as np

from .arm import JOINTS
from .real import STALL_NM


class MockBus:
    def __init__(self, robot, *, known=("Present_Load", "Status", "Present_Velocity", "Present_Position")):
        self.robot, self.known = robot, set(known)

    def sync_read(self, name, normalize=True):
        if name not in self.known:
            raise KeyError(name)
        r = self.robot
        r._step()
        if name == "Present_Load":
            return {j: int(round(min(1.0, abs(v) / STALL_NM) * 1000)) for j, v in zip(JOINTS, r.load)}
        if name == "Status":
            return {j: 0 for j in JOINTS}
        if name == "Present_Velocity":
            return {j: 0 for j in JOINTS}
        return {j: float(v) for j, v in zip(JOINTS, r.q)}

    def enable_torque(self):
        self.robot.torque = True

    def disable_torque(self):
        self.robot.torque = False


class MockRobot:
    def __init__(self, start=(0.0, -95.0, 90.0, 55.0, 90.8, 0.0), *, block_at=None, known=None):
        self.q = np.array(start, dtype=float)
        self.goal = self.q.copy()
        self.load = np.zeros(6)
        self.torque = False
        self.stuck = set()
        self.block_at = block_at
        self.is_connected = False
        self._t = time.monotonic()
        self.bus = MockBus(self, **({"known": known} if known is not None else {}))

    def connect(self):
        self.is_connected, self.torque = True, True

    def disconnect(self):
        self.is_connected = False

    def _step(self):
        now = time.monotonic()
        dt, self._t = min(0.1, now - self._t), now
        if not self.torque:
            self.load[:] = 0.0
            return
        for i in range(6):
            if i in self.stuck:
                self.load[i] = 0.0
                continue
            target = self.goal[i]
            if i == 5 and self.block_at is not None and target < self.block_at:
                target = self.block_at                       # the jaws stop on the block
                self.load[i] = min(STALL_NM, 0.5 * (self.block_at - self.goal[i]))
            else:
                self.load[i] = min(STALL_NM, 0.02 * abs(self.goal[i] - self.q[i]))
            self.q[i] += (target - self.q[i]) * min(1.0, dt * 12.0)

    def get_observation(self):
        self._step()
        return {f"{j}.pos": float(v) for j, v in zip(JOINTS, self.q)}

    def send_action(self, action):
        self._step()
        self.goal = np.array([float(action.get(f"{j}.pos", g)) for j, g in zip(JOINTS, self.goal)])
        return action
