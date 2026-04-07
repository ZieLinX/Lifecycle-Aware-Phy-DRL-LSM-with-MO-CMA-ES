from __future__ import annotations

import os
import unittest
from unittest import mock

from utils.exporter import _resolve_freecad_command


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


if __name__ == "__main__":
    unittest.main()
