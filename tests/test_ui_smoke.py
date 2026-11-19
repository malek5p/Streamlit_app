import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_empty_upload_state_renders_without_exception(self):
        app = Path(__file__).resolve().parents[1] / "ThermoSEEG_RC_UI.py"
        result = AppTest.from_file(str(app), default_timeout=45).run()
        self.assertEqual(len(result.exception), 0, [item.message for item in result.exception])
        self.assertTrue(any("Upload MRI/CT Slice" in str(item.label) for item in result.get("file_uploader")))


if __name__ == "__main__":
    unittest.main()
