from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticCompileTest(unittest.TestCase):
    def test_sources_compile(self) -> None:
        files = [
            ROOT / "optimize_3d.py",
            ROOT / "train_policy.py",
            ROOT / "train_surrogate.py",
            ROOT / "config" / "cylinder_cfg.py",
            ROOT / "utils" / "exporter.py",
            ROOT / "utils" / "full3d_optimizer.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
