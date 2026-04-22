from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "RL smoke test requires CUDA")
class TrainRLSmokeTest(unittest.TestCase):
    def test_train_rl_smoke_end_to_end(self) -> None:
        cmd = [
            sys.executable,
            "train_rl.py",
            "--smoke",
            "--num-actors",
            "2",
            "--max-epochs",
            "1",
            "--max-steps",
            "2",
            "--experiment-name",
            "mcga_test_smoke",
            "--train-dir",
            "outputs/test_rl_runs",
            "--final-eval-dir",
            "outputs/test_final_eval",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)


if __name__ == "__main__":
    unittest.main()
