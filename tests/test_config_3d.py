import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermoseeg_config import RunConfigError, parse_exported_run_config_3d, widget_values_3d


def _export(tool="ThermoSEEG_3D_v3", **over):
    cfg = {"procedure": "LITT Thermal Ablation", "laser_power_w": 5.5, "duration_s": 45, "cooldown_s": 30,
           "active_length_mm": 10.0, "source_radius_mm": 4.0, "contact_count": 8, "contact_spacing_mm": 3.5,
           "use_percentile": True, "risk_percentile": 98, "manual_threshold": 140, "strip_mm": 6,
           "min_risk_volume_mm3": 30, "safety_buffer_mm": 5.0, "target_mm": [-22, 4, 10]}
    cfg.update(over)
    return json.dumps({"metadata": {"tool_version": tool}, "volume": {"sha256": "abc"},
                       "reproducibility": {"configuration": cfg}}).encode()


class Config3D(unittest.TestCase):
    def test_roundtrip(self):
        v = widget_values_3d(parse_exported_run_config_3d(_export()))
        self.assertEqual(v["t3d_laser_power"], 5.5)
        self.assertEqual(v["t3d_risk_percentile"], 98)
        self.assertEqual(v["_target_mm"], [-22.0, 4.0, 10.0])

    def test_rejects_2d_export_and_garbage(self):
        with self.assertRaises(RunConfigError):
            parse_exported_run_config_3d(_export(tool="ThermoSEEG_RC_v5"))
        for raw in (b"not json", b"[]", b"{}", b"\xff\xfe"):
            with self.assertRaises(RunConfigError):
                parse_exported_run_config_3d(raw)

    def test_rejects_out_of_range_and_bad_types(self):
        for over in ({"laser_power_w": 20}, {"laser_power_w": True}, {"duration_s": 45.5}, {"risk_percentile": 50},
                     {"use_percentile": "yes"}, {"target_mm": [1, 2]}, {"procedure": "other"}, {"safety_buffer_mm": -1}):
            with self.subTest(over=over), self.assertRaises(RunConfigError):
                widget_values_3d(parse_exported_run_config_3d(_export(**over)))


if __name__ == "__main__":
    unittest.main()