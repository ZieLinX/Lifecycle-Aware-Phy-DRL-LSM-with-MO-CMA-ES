from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticCompileTest(unittest.TestCase):
    def test_sources_compile(self) -> None:
        files = [
            ROOT / "train.py",
            ROOT / "train_rl.py",
            ROOT / "optimize_hybrid.py",
            ROOT / "optimize_3d.py",
            ROOT / "envs" / "cylinder_env.py",
            ROOT / "envs" / "cylinder_vec_env.py",
            ROOT / "config" / "cylinder_cfg.py",
            ROOT / "utils" / "exporter.py",
            ROOT / "utils" / "hybrid_optimizer.py",
            ROOT / "utils" / "hybrid_optimizer_3d.py",
            ROOT / "utils" / "planner.py",
            ROOT / "utils" / "rated_condition.py",
            ROOT / "utils" / "feasibility.py",
            ROOT / "utils" / "thermo_mech.py",
            ROOT / "utils" / "animation.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
