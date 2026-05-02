from __future__ import annotations

import unittest

from utils.animation import _metric_float, _metric_value


class AnimationMetricsTest(unittest.TestCase):
    def test_metric_float_prefers_voltage_v_over_legacy_key(self) -> None:
        metrics = {"voltage_v": 12.5, "rated_voltage_v": 0.0}
        self.assertEqual(_metric_float(metrics, "voltage_v", "rated_voltage_v"), 12.5)

    def test_metric_value_falls_back_to_3d_feasibility_key(self) -> None:
        metrics = {"constraint_feasible_3d": False}
        self.assertFalse(_metric_value(metrics, "feasible", "constraint_feasible_3d"))


if __name__ == "__main__":
    unittest.main()
