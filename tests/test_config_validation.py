import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermoseeg_config import RunConfigError, configuration_widget_values, parse_exported_run_config


def _export(**cfg_overrides):
    cfg = {
        "procedure": "LITT Thermal Ablation", "use_otsu": True,
        "target_px": [10, 20], "entry_px": [5, 5],
        "laser_power_w": 10.0, "duration_s": 30.0, "risk_percentile": 97,
        "roi_strip_radius_px": 7, "minimum_risk_area_px2": 30, "risk_buffer_px": 5,
        "pixel_spacing_mm_xy": [0.5, 0.5],
    }
    cfg.update(cfg_overrides)
    return {
        "metadata": {"tool_version": "t", "image_sha256": "a" * 64,
                     "pixel_spacing_source": "manual isotropic assumption"},
        "reproducibility": {"configuration": cfg},
    }


def _load(payload):
    parsed = parse_exported_run_config(json.dumps(payload).encode())
    return configuration_widget_values(parsed)


class ConfigValidation(unittest.TestCase):
    def test_valid_roundtrip(self):
        v = _load(_export())
        self.assertEqual(v["manual_pixel_spacing"], 0.5)
        self.assertEqual(v["target_point"], (10, 20))

    def test_bad_spacing_values_raise_runconfigerror(self):
        for bad in (None, "bad", True, False, [None, None], ["bad", "bad"],
                    [True, True], [0.5], [0.5, 0.5, 0.5], "0.5", 10 ** 400):
            with self.subTest(bad=bad):
                with self.assertRaises(RunConfigError):
                    _load(_export(pixel_spacing_mm_xy=bad if isinstance(bad, list) else [bad, bad]) if not isinstance(bad, list) else _export(pixel_spacing_mm_xy=bad))

    def test_spacing_not_list(self):
        for bad in (None, "0.5", 0.5, {"x": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(RunConfigError):
                    _load(_export(pixel_spacing_mm_xy=bad))

    def test_non_finite_json_constants_rejected(self):
        raw = json.dumps(_export()).replace("[0.5, 0.5]", "[NaN, Infinity]").encode()
        with self.assertRaises(RunConfigError):
            parse_exported_run_config(raw)

    def test_non_isotropic_spacing_rejected(self):
        with self.assertRaises(RunConfigError):
            _load(_export(pixel_spacing_mm_xy=[0.5, 0.6]))

    def test_bad_numeric_fields(self):
        for field in ("laser_power_w", "duration_s", "risk_percentile", "roi_strip_radius_px"):
            for bad in (None, "bad", True, False, float("nan"), float("inf"), 10 ** 400, -1):
                with self.subTest(field=field, bad=bad):
                    payload = _export(**{field: bad})
                    raw = json.dumps(payload, allow_nan=True).encode()
                    with self.assertRaises(RunConfigError):
                        configuration_widget_values(parse_exported_run_config(raw))

    def test_bad_points(self):
        for bad in (None, "x", [1], [1, 2, 3], [1.5, 2], [True, 2], ["a", "b"]):
            with self.subTest(bad=bad):
                with self.assertRaises(RunConfigError):
                    parse_exported_run_config(json.dumps(_export(target_px=bad)).encode())

    def test_malformed_documents(self):
        for raw in (b"not json", b"[]", b"null", b"{}", b"\xff\xfe", b'{"metadata": 1}'):
            with self.subTest(raw=raw):
                with self.assertRaises(RunConfigError):
                    parse_exported_run_config(raw)

    def test_missing_configuration_section(self):
        with self.assertRaises(RunConfigError):
            configuration_widget_values({})
        with self.assertRaises(RunConfigError):
            configuration_widget_values({"configuration": None})


if __name__ == "__main__":
    unittest.main()
