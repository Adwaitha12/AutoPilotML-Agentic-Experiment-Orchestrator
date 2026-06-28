import tempfile
import unittest
from pathlib import Path

from core.context import ExperimentContext


class VisualizationArtifactSanitizationTests(unittest.TestCase):
    def test_sanitize_visualization_artifacts_filters_invalid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            valid_png = tmp_path / "valid.png"
            valid_png.write_bytes(b"not-a-real-png-but-it-exists")

            empty_png = tmp_path / "empty.png"
            empty_png.write_bytes(b"")

            html_file = tmp_path / "chart.html"
            html_file.write_text("<html></html>", encoding="utf-8")

            artifacts = {
                "valid": valid_png,
                "empty": empty_png,
                "html": html_file,
                "invalid_object": object(),
                "missing": tmp_path / "missing.png",
            }

            context = ExperimentContext()
            sanitized = context.sanitize_visualization_artifacts(artifacts)

            self.assertEqual(sanitized, {"valid": str(valid_png)})


if __name__ == "__main__":
    unittest.main()
