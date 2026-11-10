"""The real-arm driver against a mock servo bus (no hardware): python -m pytest host/tests"""
import numpy as np
import pytest

from so101_host import level1
from so101_host.arm import JOINTS
from so101_host.host import Skills, lift
from so101_host.mockrobot import MockRobot
from so101_host.real import STALL_NM, RealArm, taught_library

POSES = {   # what `python -m so101_host teach` writes: [5 joints in degrees, gripper 0-100] (made-up values)
    "rest": [0, -95, 90, 55, 90, 0],
    "home": [5, -30, 30, 90, 90, 60],
    "above_block": [4, -11, 6, 95, 90, 60],
    "pick": [4, -5, 33, 63, 90, 60],
}


def arm_with(poses=POSES, **kw):
    arm = RealArm(robot=MockRobot(**kw)).connect()
    arm.taught_poses = poses
    return arm, Skills(arm, poses=poses, motions={})


def test_moves_arrive_and_a_dead_servo_stalls():
    arm, skills = arm_with()
    assert all(arm.registers.values()), arm.registers
    assert skills.go_home()["arrived"]
    arm.robot.stuck = {2}                                   # elbow servo dead
    r = skills.reach()
    assert not r["arrived"] and r["worst_joint"] == "elbow_flex", r
    found = level1.check_arrival(r, load_nm=arm.load_nm("elbow_flex"))
    assert found and found[0]["kind"] == "joint_stall", found


def test_grip_backs_off_below_the_overload_and_lifts_in_joint_space():
    arm, skills = arm_with(block_at=18.0)
    skills.go_home()
    arm.goto(POSES["pick"], duration=0.6)
    g = skills.grip()
    assert g["grasp"]["holding"], g
    assert 0.4 * STALL_NM <= g["grasp"]["load_nm"] < 0.9 * STALL_NM, g
    assert not any(g["status"]), g
    r = lift(arm, POSES)
    assert r["arrived"] and r["mode"] == "joint" and r["grasp"]["holding"], r


def test_overload_bit_when_held_at_the_limit():
    arm, _ = arm_with(block_at=18.0)
    arm.goto(POSES["pick"][:5] + [0], duration=0.3, settle_timeout=0.1)
    arm.hold(0.8)
    assert arm.status()[5] == 32, arm.status()


def test_unknown_registers_are_skipped():
    arm = RealArm(robot=MockRobot(known=("Present_Position",))).connect()
    assert arm.registers == {"Present_Load": False, "Status": False, "Present_Velocity": False}
    arm.hold(0.1)
    assert arm.status() == [0] * 6 and arm.load_nm("gripper") == 0.0


def test_approach_without_a_sensor_goes_to_the_taught_pick():
    arm, skills = arm_with()
    skills.go_home()
    r = skills.approach()
    assert r["arrived"] and r["mode"] == "joint", r
    assert np.allclose(arm._q[:5], POSES["pick"][:5], atol=2.0)


def test_an_untaught_pose_is_refused():
    arm, skills = arm_with(poses={**POSES, "pick": None})
    skills.go_home()
    with pytest.raises(RuntimeError, match="isn't taught"):
        skills.approach()


def test_no_taught_poses_means_no_motion_at_all():
    empty = {name: None for name in POSES}
    arm, _ = arm_with(poses=empty)
    with pytest.raises(RuntimeError, match="no taught poses"):
        arm.send([0] * 6)


def test_taught_library_reads_poses_and_motions(tmp_path):
    (tmp_path / "motions").mkdir()
    (tmp_path / "poses.json").write_text('{"home": [5, -30, 30, 90, 90, 60]}')
    (tmp_path / "motions" / "pack.json").write_text('{"name": "pack", "fps": 50, "joints": %s, "frames": [[0,0,0,0,0,0]]}'
                                                    % list(JOINTS).__repr__().replace("'", '"'))
    poses, motions = taught_library(tmp_path)
    assert poses["home"] == [5, -30, 30, 90, 90, 60] and poses["pick"] is None
    assert list(motions) == ["pack"]
