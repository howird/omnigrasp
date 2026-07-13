"""Registry of known objects for the OmniGrasp pkl-synthesis and trajectory
authoring scripts (`scripts/data_process/create_object_pkl.py`,
`scripts/vis/create_object_traj_viser.py`).

Adding a new object means adding one `ObjectAsset` and one `OBJECT_RECIPES`
entry here — no new script or dataclass required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phc.utils.grasp_pkl import (
    AbsoluteShaftHeight,
    HandGraspParams,
    LinearApproach,
    ObjectTrajectory,
    RandomSmoothApproach,
    RelativeShaftOffset,
)


@dataclass
class ObjectAsset:
    name: str
    """Entry-dict key and urdf/obj_name stem, e.g. "floorlamp"."""
    mesh_path: Path
    """Mesh used for BPS encoding and viser display (trimesh-loadable: .stl or .obj)."""
    mesh_scale: float
    """Uniform scale applied to mesh vertices before use (mm meshes -> 0.001)."""
    obj_info_path: str
    """String written to entry["obj_data"]["obj_info"][0]. Distinct from mesh_path:
    phc/env/tasks/humanoid_omnigrasp.py parses this path's *stem* as obj_name and
    substring-matches "omomo"/"oakink" in its *directory* to choose the urdf asset
    root (phc/data/assets/urdf/omomo/, .../oakink/, else phc/data/assets/urdf/grab/
    by default). It need not point at a real file."""

    def __post_init__(self) -> None:
        assert Path(self.obj_info_path).stem == self.name, (
            f"ObjectAsset.obj_info_path stem must match name: "
            f"{self.obj_info_path!r} vs {self.name!r}"
        )


# FLOORLAMP.obj_info_path intentionally points at urdf/omomo/ rather than the
# mesh's actual source directory (phc/data/assets/mesh/omomo/) -- only the
# "omomo" substring + "floorlamp" stem matter, and they must resolve to the
# directory that actually contains floorlamp.urdf. Do not "fix" this to point
# at the mesh source dir; that directory has no .urdf files.
FLOORLAMP = ObjectAsset(
    name="floorlamp",
    mesh_path=Path("phc/data/assets/mesh/omomo/floorlamp.stl"),
    mesh_scale=1.0,
    obj_info_path="phc/data/assets/urdf/omomo/floorlamp.stl",
)

# HOCKEY_STICK.obj_info_path contains neither "omomo" nor "oakink", so it
# correctly falls through to the default urdf asset root
# (phc/data/assets/urdf/grab/), where hockey_stick.urdf actually lives. The
# exact path string otherwise only matters for its filename stem, so it's
# resolved to an absolute path here (rather than left CWD-relative) so the
# scripts that use it work regardless of invocation directory.
_HOCKEY_STICK_OBJ = Path(__file__).resolve().parents[3] / "humanoid_gym" / "assets" / "hockey_stick.obj"

HOCKEY_STICK = ObjectAsset(
    name="hockey_stick",
    mesh_path=_HOCKEY_STICK_OBJ,
    mesh_scale=0.001,
    obj_info_path=str(_HOCKEY_STICK_OBJ),
)


@dataclass
class ObjectRecipe:
    """Full default parameterization for synthesizing one object's inference pkl."""
    asset: ObjectAsset
    trajectory: ObjectTrajectory
    grasp: HandGraspParams
    entry_key: str
    output_path: Path


OBJECT_RECIPES: dict[str, ObjectRecipe] = {
    "floorlamp": ObjectRecipe(
        asset=FLOORLAMP,
        trajectory=LinearApproach(),
        grasp=HandGraspParams(z_mode=AbsoluteShaftHeight(), jitter_std=0.02, x_offset=0.06),
        entry_key="floorlamp",
        output_path=Path("sample_data/floorlamp_sample.pkl"),
    ),
    "hockey_stick": ObjectRecipe(
        asset=HOCKEY_STICK,
        trajectory=RandomSmoothApproach(),
        grasp=HandGraspParams(z_mode=RelativeShaftOffset(), jitter_std=0.015, x_offset=0.05),
        entry_key="hockey_stick",
        output_path=Path("sample_data/hockey_stick_sample.pkl"),
    ),
}

ObjectName = Literal[tuple(sorted(OBJECT_RECIPES.keys()))]
"""Valid --object-name choices, derived from OBJECT_RECIPES so it can't drift out of sync."""
