from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock

from utils.exporter import _resolve_freecad_command, _run_freecad_stl_to_step


class ExporterResolutionTest(unittest.TestCase):
    @mock.patch("utils.exporter.shutil.which", return_value=None)
    def test_freecad_env_override_is_used(self, _mock_which) -> None:
        fake_path = r"C:\Tools\FreeCADCmd.exe"

        def fake_isfile(path: str) -> bool:
            return path == fake_path

        with mock.patch.dict(os.environ, {"FREECAD_CMD": fake_path}, clear=False):
            with mock.patch("utils.exporter.os.path.isfile", side_effect=fake_isfile):
                resolved = _resolve_freecad_command("")
        self.assertEqual(resolved, fake_path)

    @mock.patch("utils.exporter.os.remove")
    @mock.patch("utils.exporter.tempfile.NamedTemporaryFile")
    @mock.patch("utils.exporter._resolve_freecad_command", return_value="FreeCADCmd")
    @mock.patch("utils.exporter.subprocess.run", side_effect=subprocess.TimeoutExpired(["FreeCADCmd"], 1.0))
    def test_freecad_step_conversion_timeout_returns_false(
        self,
        _mock_run,
        _mock_resolve,
        mock_temp,
        _mock_remove,
    ) -> None:
        temp_file = mock.Mock()
        temp_file.name = "convert.py"
        mock_temp.return_value.__enter__.return_value = temp_file

        written = _run_freecad_stl_to_step("input.stl", "output.stp", timeout_s=1.0)

        self.assertFalse(written)


if __name__ == "__main__":
    unittest.main()
