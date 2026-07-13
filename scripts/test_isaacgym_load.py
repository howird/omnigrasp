"""Minimal smoke test: load humanoid_gym-generated MJCF into IsaacGym.

Verifies:
- gym.load_asset() succeeds (no parse errors)
- DOF count == 69
- Body count == 24
- All 23 SMPL joint stems are present
- Sim steps 10 frames without explosion

Run from the omnigrasp root:
    .venv/bin/python scripts/test_isaacgym_load.py [--xml PATH]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from isaacgym import gymapi

SMPL_JOINT_STEMS = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]  # 23 bodies (excludes Pelvis/root)


def main() -> None:
    parser = argparse.ArgumentParser(description="IsaacGym MJCF load test")
    parser.add_argument(
        "--xml",
        default="/home/howard/humanoid_gym/outputs/smpl_humanoid.xml",
        help="Path to the MJCF file to test",
    )
    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    asset_root = os.path.dirname(xml_path)
    asset_file = os.path.basename(xml_path)

    # --- Init IsaacGym ---
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # --- Load asset ---
    print(f"\nLoading: {xml_path}")
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    print("gym.load_asset() succeeded\n")

    # --- Inspect DOFs ---
    num_dof = gym.get_asset_dof_count(asset)
    num_bodies = gym.get_asset_rigid_body_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)

    print(f"DOFs:   {num_dof}  (expected 69)  {'OK' if num_dof == 69 else 'MISMATCH'}")
    print(f"Bodies: {num_bodies}  (expected 24)  {'OK' if num_bodies == 24 else 'MISMATCH'}")

    print("\nDOF names:")
    for i, name in enumerate(dof_names):
        print(f"  [{i:2d}] {name}")

    print("\nBody names:")
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")

    # --- Check all joint stems present ---
    dof_name_set = set(dof_names)
    missing_stems = [
        stem for stem in SMPL_JOINT_STEMS
        if not any(stem in n for n in dof_name_set)
    ]
    if missing_stems:
        print(f"\nMISSING joint stems: {missing_stems}")
    else:
        print(f"\nAll {len(SMPL_JOINT_STEMS)} SMPL joint stems present  OK")

    # --- Create env + actor and step sim ---
    spacing = 2.0
    env = gym.create_env(
        sim,
        gymapi.Vec3(-spacing, 0, -spacing),
        gymapi.Vec3(spacing, spacing, spacing),
        1,
    )
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    gym.create_actor(env, asset, pose, "humanoid", 0, 0)

    print("\nStepping sim 10 frames...")
    for step in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    # Check DOF state via per-actor API (no GPU tensor needed)
    actor = gym.get_actor_handle(env, 0)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    positions = np.array([s[0] for s in dof_states])
    velocities = np.array([s[1] for s in dof_states])
    has_nan = np.isnan(positions).any() or np.isnan(velocities).any()
    has_inf = np.isinf(positions).any() or np.isinf(velocities).any()
    print(f"NaN/Inf in DOF state after 10 steps: {has_nan or has_inf}  {'FAIL' if has_nan or has_inf else 'OK'}")

    gym.destroy_sim(sim)

    print("\n--- Summary ---")
    ok = (num_dof == 69) and (num_bodies == 24) and not missing_stems and not has_nan
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
