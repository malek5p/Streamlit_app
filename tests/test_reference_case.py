import json, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import make_reference_case as ref


class ReferenceCaseTest(unittest.TestCase):
    def test_engine_reproduces_reference(self):
        saved = json.loads((ROOT / "docs" / "reference_case_phantom.json").read_text())
        now = ref.build()
        self.assertEqual(saved["input"]["volume_shape"], now["input"]["volume_shape"])
        self.assertEqual(saved["input"]["entry_mm"], now["input"]["entry_mm"])
        for preset, res in saved["results"].items():
            for key, val in res.items():
                cur = now["results"][preset][key]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    self.assertAlmostEqual(val, cur, delta=max(1e-6, 1e-4 * abs(val)), msg=f"{preset}/{key}")
                else:
                    self.assertEqual(val, cur, msg=f"{preset}/{key}")


if __name__ == "__main__":
    unittest.main()