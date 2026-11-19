from __future__ import annotations

import ast
import json
import logging
import time
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

APP = Path(__file__).resolve().parents[1] / "ThermoSEEG_RC_UI.py"
FUNCTIONS = {
    "bresenham_line", "calculate_geometry", "check_roi_containment", "_brain_roi",
    "_derive_threshold", "_compute_masks_core", "validate_target", "compute_seeg_contacts",
    "simulate_litt_thermal_field", "_build_pdf_report", "_distance_map", "check_collision", "_in_bounds_xy",
    "min_clearance_along_line", "_entry_is_admissible", "_evaluate_entry",
    "_roi_edge_candidates", "find_best_entry_candidate", "clamp", "get_status_color", "_roi_heuristic_modality_description",
    "load_and_preprocess_image", "compute_distance_map_cached",
}

def load_pure_functions():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    for node in body:
        node.decorator_list = []  # avoid Streamlit's cache runtime in isolated unit tests
    module = ast.Module(body=body, type_ignores=[])
    env: Dict[str, Any] = {
        "np": np, "cv2": cv2, "distance_transform_edt": distance_transform_edt,
        "logger": logging.getLogger("test-thermoseeg"),
        "gaussian_filter": gaussian_filter, "time": time, "Tuple": Tuple, "List": List, "Dict": Dict,
        "Any": Any, "Optional": Optional, "PointTuple": Tuple[int, int],
        "MaskArray": np.ndarray, "FloatArray": np.ndarray,
        "ROI_STRIP_RADIUS_DEFAULT_PX": 7, "ROI_CLOSE_KERNEL_PX": 9,
        "RISK_PERCENTILE_DEFAULT": 97, "BASELINE_BODY_TEMP_C": 37.0,
        "CEM43_THRESHOLD_TEMP_C": 43.0, "CEM43_R_HIGH": 0.5, "CEM43_R_LOW": 0.25,
        "CEM43_ABLATION_THRESHOLD_MIN": 240.0, "LITT_GAUSSIAN_SIGMA_BASE_PX": 8.0,
        "LITT_POWER_TO_DELTA_T_FACTOR": 4.8, "RISK_ATTENUATION_FACTOR": 0.30,
        "RISK_ATTENUATION_SIGMA_PX": 4.0, "THERMAL_NUMERICAL_GUARD_C": 100.0,
        "TEMP_DISPLAY_MAX_C": 80.0, "ROI_ENTRY_EDGE_TOL_PX": 6.0,
        "HEURISTIC_MIN_CLEARANCE_PX": 10.0, "HEURISTIC_WARNING_CLEARANCE_PX": 25.0,
        "HEURISTIC_CLEARANCE_SCORE_CAP_PX": 60.0, "RISK_MASK_MAX_FRACTION": 0.15,
        "MAX_UPLOAD_BYTES": 50 * 1024 * 1024, "MAX_IMAGE_SIDE": 4096, "MAX_IMAGE_DIM_PX": 4096, "MAX_IMAGE_PIXELS": 16_000_000,
        "SCIENTIFIC_LIMITATIONS": ("2D limitation",),
        "Image": Image,
        "ROI_ENTRY_EDGE_TOL_PX": 6.0,
        "ROI_PATH_TOLERANCE_PX": 2, "st": SimpleNamespace(error=lambda message: None),
    }
    from thermoseeg_dicom import decode_dicom_slice
    env["decode_dicom_slice"] = decode_dicom_slice
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, LongTable, Table, TableStyle
    env.update({"PDF_EXPORT_AVAILABLE": True, "BytesIO": BytesIO, "colors": colors, "letter": letter,
                "getSampleStyleSheet": getSampleStyleSheet, "Paragraph": Paragraph,
                "SimpleDocTemplate": SimpleDocTemplate, "Spacer": Spacer, "Table": Table, "LongTable": LongTable,
                "TableStyle": TableStyle, "json": json})
    exec(compile(module, str(APP), "exec"), env)
    return env


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = load_pure_functions()

    def test_anisotropic_length(self):
        result = self.f["calculate_geometry"]((0, 0), (3, 4), (2.0, 1.0))
        self.assertAlmostEqual(result["length_mm"], np.hypot(6, 4))
        self.assertAlmostEqual(result["length"], 5.0)

    def test_bresenham_includes_endpoints(self):
        points = self.f["bresenham_line"]((1, 2), (8, 6))
        self.assertEqual(points[0], (1, 2))
        self.assertEqual(points[-1], (8, 6))
        self.assertLessEqual(max(abs(b[0]-a[0]) for a, b in zip(points, points[1:])), 1)

    def test_roi_containment_counts_out_of_bounds(self):
        mask = np.zeros((5, 5), np.uint8)
        mask[1:4, 1:4] = 255
        self.assertEqual(self.f["check_roi_containment"](mask, [(0, 0), (1, 1), (8, 9)]), 2)

    def test_pixel_spacing_distance_transform_respects_row_column_order(self):
        mask = np.zeros((5, 5), np.uint8)
        mask[2, 2] = 255
        distance_mm = self.f["compute_distance_map_cached"](mask, sampling=(0.8, 0.6))
        self.assertAlmostEqual(distance_mm[2, 3], 0.6)
        self.assertAlmostEqual(distance_mm[3, 2], 0.8)

    def test_target_bounds_and_roi(self):
        roi = np.zeros((10, 10), np.uint8)
        roi[2:8, 2:8] = 255
        risk = np.zeros_like(roi)
        self.assertFalse(self.f["validate_target"]((-1, 3), risk, roi, 10, 10)[0])
        self.assertFalse(self.f["validate_target"]((1, 1), risk, roi, 10, 10)[0])
        self.assertTrue(self.f["validate_target"]((3, 3), risk, roi, 10, 10)[0])

    def test_seeg_geometric_span_and_placement(self):
        result = self.f["compute_seeg_contacts"]((0, 0), (0, 35), 35.0, 4, 10.0)
        self.assertTrue(result["fits_within_trajectory"])
        self.assertEqual(result["num_contacts_placed"], 4)
        self.assertEqual(result["required_span_px"], 30.0)

    def test_candidate_search_is_bounded_and_audited(self):
        roi = np.zeros((101, 101), np.uint8)
        cv2.circle(roi, (50, 50), 40, 255, -1)
        risk = np.zeros_like(roi)
        dist = self.f["_distance_map"](risk)
        depth = distance_transform_edt(roi > 0)
        result = self.f["find_best_entry_candidate"](risk, (50, 50), dist, roi, roi, depth, 24)
        self.assertIsNotNone(result["best"])
        self.assertEqual(result["initial_candidates_generated"], 24)
        self.assertGreater(result["total_path_evaluations"], 0)
        self.assertEqual(result["initial_path_evaluations"], result["initial_candidates_unique"])
        self.assertEqual(result["total_path_evaluations"], result["initial_path_evaluations"] + result["refinement_path_evaluations"])
        self.assertLessEqual(result["initial_candidates_accepted"], result["initial_path_evaluations"])
        self.assertTrue(any(row["accepted"] for row in result["candidate_audit"]))


class ImageAndThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = load_pure_functions()

    def test_roi_fills_internal_hole_by_design(self):
        # _brain_roi deliberately fills internal dark regions (ventricles/sulci) so the ALLOWED region is solid
        # (documented in its docstring; the obstacle/risk mask is NOT filled). This is a regression test of that choice.
        gray = np.zeros((151, 151), np.uint8)
        cv2.circle(gray, (75, 75), 55, 220, -1)
        cv2.circle(gray, (75, 75), 22, 0, -1)
        roi = self.f["_brain_roi"](gray, strip_radius_px=1)
        self.assertEqual(int(roi[75, 75]), 255)   # hole filled
        self.assertEqual(int(roi[5, 5]), 0)       # background stays outside

    def test_roi_handles_two_sizes(self):
        for size in (64, 128):
            gray = np.zeros((size, size), np.uint8)
            cv2.circle(gray, (size//2, size//2), size//3, 180, -1)
            roi = self.f["_brain_roi"](gray, strip_radius_px=0)
            self.assertEqual(roi.shape, gray.shape)
            self.assertGreater(int(np.count_nonzero(roi)), 0)

    def test_raster_ingestion_and_invalid_input(self):
        image = np.zeros((512, 512, 3), np.uint8)
        cv2.circle(image, (256, 256), 190, (180, 180, 180), -1)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        result = self.f["load_and_preprocess_image"](encoded.tobytes(), 0, True, 97.0, 1, 2, 1)
        self.assertEqual(result["original"].shape, image.shape)
        self.assertIn("roi", result)
        self.assertGreaterEqual(result["processing_seconds"], 0.0)
        invalid = self.f["load_and_preprocess_image"](b"not an image", 0, True, 97.0, 1, 2, 1)
        self.assertEqual(invalid, {})
        oversized = np.zeros((2, 4097, 3), np.uint8)
        ok, oversized_png = cv2.imencode(".png", oversized)
        self.assertTrue(ok)
        self.assertEqual(self.f["load_and_preprocess_image"](oversized_png.tobytes(), 0, True, 97.0, 1, 2, 1), {})

    def test_dose_area_is_restricted_to_analysis_mask(self):
        raw = np.zeros((64, 64), np.uint8)
        region = np.zeros_like(raw)
        region[20:30, 20:30] = 255
        result = self.f["simulate_litt_thermal_field"]((25, 25), raw, region, 10.0, 60.0)
        self.assertTrue(result["dose_valid"])
        self.assertFalse(np.any(result["cem43_map"][region == 0]))
        self.assertLessEqual(result["ablation_area_px"], int(np.count_nonzero(region)))

    def test_duration_changes_dose_not_temperature(self):
        raw = np.zeros((64, 64), np.uint8)
        region = np.full_like(raw, 255)
        short = self.f["simulate_litt_thermal_field"]((32, 32), raw, region, 5.0, 30.0)
        long = self.f["simulate_litt_thermal_field"]((32, 32), raw, region, 5.0, 60.0)
        self.assertTrue(np.array_equal(short["temp_map"], long["temp_map"]))
        self.assertGreater(long["max_cem43_min"], short["max_cem43_min"])

    def test_invalid_thermal_guard_suppresses_dose_outputs(self):
        raw = np.zeros((48, 48), np.uint8)
        region = np.full_like(raw, 255)
        result = self.f["simulate_litt_thermal_field"]((24, 24), raw, region, 20.0, 60.0)
        self.assertFalse(result["dose_valid"])
        self.assertFalse(result["numerical_guard_passed"])
        self.assertIsNone(result["max_cem43_min"])
        self.assertIsNone(result["ablation_area_px"])
        self.assertEqual(int(np.count_nonzero(result["cem43_map"])), 0)

    def test_roi_heuristic_export_modality_handles_missing_dicom_metadata(self):
        describe = self.f["_roi_heuristic_modality_description"]
        self.assertIn("raster modality unverified", describe(None))
        self.assertIn("CT use is unvalidated", describe({"modality": "CT"}))
        self.assertEqual(describe({"modality": "MR"}), "MRI-like contrast")

    def test_json_export_payload_round_trip(self):
        sample = {"status": "HEURISTIC_CHECKS_PASSED", "metric_mm": None,
                  "scenarios": [{"threshold": 92.0, "collision": False}]}
        self.assertEqual(json.loads(json.dumps(sample)), sample)

    def test_exported_run_settings_can_be_loaded_and_validated(self):
        from thermoseeg_config import parse_exported_run_config, configuration_widget_values
        exported = {
            "metadata": {"tool_version": "ThermoSEEG_RC_v5.1-research", "image_sha256": "abc",
                         "pixel_spacing_source": "manual isotropic assumption"},
            "reproducibility": {"configuration": {
                "procedure": "LITT Thermal Ablation", "use_otsu": True,
                "manual_threshold": None, "risk_percentile": 97,
                "roi_strip_radius_px": 5, "minimum_risk_area_px2": 20,
                "risk_buffer_px": 3, "target_px": [40, 40], "entry_px": [5, 40],
                "pixel_spacing_mm_xy": [0.5, 0.5], "laser_power_w": 8.0,
                "duration_s": 30.0, "contact_count": None, "contact_spacing_px": None,
            }},
        }
        run_config = parse_exported_run_config(json.dumps(exported).encode())
        values = configuration_widget_values(run_config)
        self.assertEqual(values["procedure_type"], "LITT Thermal Ablation")
        self.assertEqual(values["target_point"], (40, 40))
        self.assertEqual(values["manual_pixel_spacing"], 0.5)

    def test_import_rejects_oversized_and_nonfinite_pixel_spacing_as_config_error(self):
        from thermoseeg_config import RunConfigError, configuration_widget_values
        base = {"configuration": {"procedure": "SEEG Electrode Planning", "use_otsu": True,
                                  "target_px": [4, 5], "entry_px": [1, 2],
                                  "contact_count": 8, "contact_spacing_px": 10,
                                  "pixel_spacing_mm_xy": [1.0, 1.0]},
                "pixel_spacing_source": "manual isotropic assumption"}
        for value in [10**10000, float("inf"), float("nan"), True, "1.0"]:
            run = {**base, "configuration": {**base["configuration"], "pixel_spacing_mm_xy": [value, value]}}
            with self.subTest(value_type=type(value).__name__), self.assertRaises(RunConfigError):
                configuration_widget_values(run)

    def test_pdf_report_is_generated(self):
        pdf = self.f["_build_pdf_report"]({"metadata": {"scope": "2D prototype CEM43≥240 ⚠️"},
                                           "trajectory_analysis": {"length_mm": 4.2}})
        self.assertTrue(pdf.startswith(b"%PDF"))
        large = self.f["_build_pdf_report"]({"metadata": {f"setting_{i}": "value " * 18 for i in range(240)}})
        self.assertTrue(large.endswith(b"%%EOF\n"))
        self.assertGreaterEqual(large.count(b"/Type /Page"), 2)


if __name__ == "__main__":
    unittest.main()