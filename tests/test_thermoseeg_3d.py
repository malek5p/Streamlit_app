"""Tests for the ThermoSEEG 3D engine (numerical sanity only; not a validation study)."""
import unittest

import numpy as np

import thermoseeg_3d as t3


def _uniform_volume(n=24, sp=2.0, label=t3.LABEL_BRAIN):
    labels = np.full((n, n, n), label, dtype=np.uint8)
    anat = np.full((n, n, n), 0.5, dtype=np.float32)
    origin = -(n - 1) * sp / 2 * np.ones(3)
    return t3.Volume3D(anat, labels, np.full(3, sp), origin, source="test")


class PhantomAndGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol = t3.make_phantom(3.0)

    def test_world_index_roundtrip(self):
        p = np.array([-22.0, 4.0, 10.0])
        np.testing.assert_allclose(self.vol.index_to_world(self.vol.world_to_index(p)), p, atol=1e-9)

    def test_labels_present(self):
        for l in (t3.LABEL_BRAIN, t3.LABEL_CSF, t3.LABEL_VESSEL):
            self.assertTrue((self.vol.labels == l).any())

    def test_default_target_is_inside_tissue(self):
        self.assertEqual(self.vol.label_at(self.vol.default_target), t3.LABEL_BRAIN)

    def test_entry_on_surface_and_path_inside(self):
        d = t3.direction_from_angles(180, 10)
        entry = t3.find_entry_on_surface(self.vol, self.vol.default_target, d)
        self.assertIsNotNone(entry)
        res = t3.analyze_trajectory(self.vol, entry, self.vol.default_target, t3.vessel_distance_map_mm(self.vol))
        self.assertEqual(res["outside_mask_mm"], 0.0)
        self.assertAlmostEqual(res["length_mm"], float(np.linalg.norm(entry - self.vol.default_target)), places=1)

    def test_target_outside_returns_none(self):
        self.assertIsNone(t3.find_entry_on_surface(self.vol, [500, 0, 0], [1, 0, 0]))

    def test_vessel_hit_is_danger(self):
        idx = np.argwhere(self.vol.labels == t3.LABEL_VESSEL)[len(np.argwhere(self.vol.labels == t3.LABEL_VESSEL)) // 2]
        tgt = self.vol.index_to_world(idx)
        res = t3.analyze_trajectory(self.vol, tgt + [0, 0, 15], tgt, t3.vessel_distance_map_mm(self.vol))
        self.assertEqual(res["status"], "DANGER")

    def test_angles_roundtrip(self):
        d = t3.direction_from_angles(135, 30)
        az, el = t3.angles_from_direction(d)
        self.assertAlmostEqual(az, 135, places=6)
        self.assertAlmostEqual(el, 30, places=6)

    def test_seeg_contacts_geometry(self):
        r = t3.seeg_contacts([0, 0, 40], [0, 0, 0], 8, 3.5)
        self.assertTrue(r["fits"])
        self.assertEqual(len(r["contacts"]), 8)
        self.assertAlmostEqual(r["required_span_mm"], 24.5)
        r2 = t3.seeg_contacts([0, 0, 10], [0, 0, 0], 8, 3.5)
        self.assertFalse(r2["fits"])
        self.assertLess(len(r2["contacts"]), 8)

    def test_surface_mesh(self):
        m = t3.surface_mesh(self.vol.labels > 0, 0.5, self.vol.spacing, self.vol.origin, smooth_sigma=0.8)
        self.assertIsNotNone(m)
        verts, faces = m
        self.assertGreater(len(faces), 100)
        lo, hi = self.vol.bounds_mm
        self.assertTrue(np.all(verts >= lo - 3) and np.all(verts <= hi + 3))


class ThermalPhysics(unittest.TestCase):
    def test_energy_conservation_without_perfusion(self):
        vol = _uniform_volume()
        p = t3.ThermalParams(power_w=2.0, exposure_s=20.0, cooldown_s=0.0, perfusion_scale=0.0,
                             vessel_sink_scale=0.0, margin_mm=60.0)
        tgt = np.zeros(3)
        d = np.array([0.0, 0.0, 1.0])
        box = t3.crop_box(vol, tgt, d, p)
        Q = t3.source_field(vol, box, tgt, d, p)
        res = t3.run_thermal(vol, box, Q, p, csf_model=False)
        k, C, h = t3._property_fields(vol.labels[t3._box_slices(box)], p, False)
        dV = float(np.prod(vol.spacing)) * 1e-9
        stored = float((C * (res["snapshots"][-1].astype(float) - 37.0)).sum() * dV)
        self.assertAlmostEqual(stored / (p.power_w * p.exposure_s), 1.0, delta=0.01)

    def test_uniform_source_reaches_analytic_steady_state(self):
        vol = _uniform_volume(10)
        p = t3.ThermalParams(exposure_s=4000.0, cooldown_s=0.0, perfusion_scale=1.0, n_snapshots=3)
        box = tuple((0, 10) for _ in range(3))
        Q = np.full((10, 10, 10), 5000.0)
        res = t3.run_thermal(vol, box, Q, p, csf_model=False)
        h = t3.TISSUE[t3.LABEL_BRAIN]["perfusion_1_per_s"] * t3.BLOOD_RHO * t3.BLOOD_C
        expected = 37.0 + 5000.0 / h
        self.assertAlmostEqual(float(res["snapshots"][-1].mean()), expected, delta=0.05 * (expected - 37))

    def test_no_nan_and_monotone_in_power(self):
        vol = t3.make_phantom(3.0)
        tg, d = vol.default_target, t3.direction_from_angles(180, 10)
        peaks = []
        for P in (1.0, 2.0, 3.0):
            r = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(power_w=P))
            self.assertTrue(np.isfinite(r["with_csf"]["snapshots"]).all())
            peaks.append(r["metrics"]["max_temp_with_csf_c"])
        self.assertTrue(peaks[0] < peaks[1] < peaks[2])

    def test_more_perfusion_lowers_peak(self):
        vol = t3.make_phantom(3.0)
        tg, d = vol.default_target, t3.direction_from_angles(180, 10)
        lo = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(perfusion_scale=0.2))["metrics"]
        hi = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(perfusion_scale=3.0))["metrics"]
        self.assertGreater(lo["max_temp_with_csf_c"], hi["max_temp_with_csf_c"])

    def test_csf_never_increases_peak_temperature(self):
        vol = t3.make_phantom(3.0)
        r = t3.run_litt_comparison(vol, vol.default_target, t3.direction_from_angles(180, 10),
                                   t3.ThermalParams(power_w=3.0))
        self.assertGreaterEqual(r["metrics"]["max_csf_dT_reduction_c"], 0.0)
        self.assertLessEqual(r["metrics"]["max_temp_with_csf_c"], r["metrics"]["max_temp_without_csf_c"] + 1e-6)

    def test_overheat_suppresses_dose(self):
        vol = t3.make_phantom(3.0)
        r = t3.run_litt_comparison(vol, vol.default_target, t3.direction_from_angles(180, 10),
                                   t3.ThermalParams(power_w=12.0))
        m = r["metrics"]
        self.assertFalse(m["model_valid"])
        self.assertIsNone(m["with_csf"]["max_cem43_min"])
        self.assertIsNone(m["relative_dose_reduction_pct"])
        self.assertTrue(r["warnings"])
        self.assertIn("n/a", t3.fmt_dose(m["with_csf"]["max_cem43_min"]))

    def test_cem43_reference_values(self):
        self.assertAlmostEqual(float(t3.cem43_rate_per_min(np.array(43.0))), 1.0)
        self.assertAlmostEqual(float(t3.cem43_rate_per_min(np.array(44.0))), 2.0)
        self.assertEqual(float(t3.cem43_rate_per_min(np.array(36.0))), 0.0)
        self.assertTrue(np.isfinite(t3.cem43_rate_per_min(np.array(1e6))))

    def test_target_outside_raises(self):
        vol = t3.make_phantom(3.0)
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3.run_litt_comparison(vol, [0, 0, 300], [0, 0, 1], t3.ThermalParams())


class Segmentation(unittest.TestCase):
    def test_vesselness_finds_bright_tube(self):
        n = 40
        img = np.zeros((n, n, n), dtype=np.float32)
        img[:, 20, 20] = 1.0
        img = __import__("scipy.ndimage", fromlist=["x"]).gaussian_filter(img, 1.0)
        v = t3.vesselness(img, [1.0, 1.0, 1.0], sigmas_mm=(1.0, 1.5))
        self.assertGreater(v[20, 20, 20], 5 * np.median(v[20]))
        self.assertLess(v[20, 5, 5], 0.1 * v[20, 20, 20])

    def test_heuristic_brain_mask_returns_largest_blob(self):
        vol = t3.make_phantom(3.0)
        m = t3.heuristic_brain_mask(vol.anatomy, vol.spacing, strip_mm=3.0)
        parenchyma = vol.labels == t3.LABEL_BRAIN
        self.assertGreater((m & parenchyma).sum() / parenchyma.sum(), 0.9)
        # the heuristic must not spill far outside the intracranial volume
        self.assertLess((m & (vol.labels == 0)).sum() / max(1, m.sum()), 0.05)



class GeometryAndMetrics(unittest.TestCase):
    def test_segment_distance_cases(self):
        self.assertAlmostEqual(t3.segment_segment_distance([0, 0, 0], [10, 0, 0], [0, 3, 0], [10, 3, 0]), 3.0)       # parallel
        self.assertAlmostEqual(t3.segment_segment_distance([0, 0, 0], [10, 0, 0], [5, 4, -5], [5, 4, 5]), 4.0)        # skew
        self.assertAlmostEqual(t3.segment_segment_distance([0, 0, 0], [10, 0, 0], [5, 0, -5], [5, 0, 5]), 0.0)        # crossing
        self.assertAlmostEqual(t3.segment_segment_distance([0, 0, 0], [1, 0, 0], [4, 4, 0], [4, 8, 0]), 5.0)          # endpoints
        self.assertAlmostEqual(t3.point_segment_distance([5, 3, 0], [0, 0, 0], [10, 0, 0]), 3.0)
        self.assertAlmostEqual(t3.point_segment_distance([-3, 4, 0], [0, 0, 0], [10, 0, 0]), 5.0)

    def test_capsule_volume_matches_analytic(self):
        vol = _uniform_volume(60, 1.0)
        m = t3.ablation_capsule_metrics(vol, [0, 0, 0], [0, 0, 1], 12.0, 5.0)
        self.assertAlmostEqual(m["capsule_volume_mm3"] / m["analytic_volume_mm3"], 1.0, delta=0.08)
        self.assertEqual(m["csf_exposure_fraction"], 0.0)

    def test_capsule_csf_fraction(self):
        vol = _uniform_volume(60, 1.0)
        vol.labels[vol.labels.shape[0] // 2:, :, :] = t3.LABEL_CSF       # x >= ~0 is CSF
        m = t3.ablation_capsule_metrics(vol, [0.0, 0.0, 0.0], [0, 0, 1], 10.0, 4.0)
        self.assertAlmostEqual(m["csf_exposure_fraction"], 0.5, delta=0.1)

    def test_capsule_surface_extent(self):
        v, f = t3.capsule_surface([0, 0, 0], [0, 0, 12], 4.0)
        self.assertAlmostEqual(float(v[:, 2].min()), -4.0, delta=1.0)
        self.assertAlmostEqual(float(v[:, 2].max()), 16.0, delta=1.0)
        self.assertLess(float(np.abs(v[:, 0]).max()), 4.8)
        self.assertEqual(f.shape[1], 3)

    def test_sphere_surface_radius(self):
        v, f = t3.sphere_surface([1, 2, 3], 5.0)
        r = np.linalg.norm(v - np.array([1, 2, 3]), axis=1)
        np.testing.assert_allclose(r, 5.0, atol=1e-4)
        self.assertEqual(f.shape[1], 3)
        self.assertLess(f.max(), len(v))

    def test_region_stats_and_history(self):
        vol = t3.make_phantom(3.0)
        tg, d = vol.default_target, t3.direction_from_angles(180, 10)
        r = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(power_w=3.0))
        near = t3.region_thermal_stats(vol, r, tg, 4.0)
        far = t3.region_thermal_stats(vol, r, tg + [0, 0, 22], 3.0)
        self.assertGreater(near["max_temp_c"], far["max_temp_c"])
        self.assertTrue(0.0 <= near["coverage_pct"] <= 100.0)
        h = r["with_csf"]["history"]
        self.assertEqual(len(h["t_s"]), len(h["max_temp_c"]))
        self.assertTrue(np.all(np.diff(h["max_cem43_min"]) >= -1e-9))      # cumulative dose never decreases
        self.assertAlmostEqual(float(h["max_temp_c"].max()), r["metrics"]["max_temp_with_csf_c"], places=1)


def make_nifti1_bytes(data, affine, gz=False):
    """Minimal NIfTI-1 writer (int16, sform) for tests."""
    import gzip
    import struct
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 4)
    struct.pack_into("<h", hdr, 72, 16)
    sp = np.sqrt((affine[:3, :3] ** 2).sum(0))
    struct.pack_into("<8f", hdr, 76, 1.0, *sp, 0, 0, 0, 0)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)
    struct.pack_into("<2h", hdr, 252, 0, 1)
    for row, off in zip(range(3), (280, 296, 312)):
        struct.pack_into("<4f", hdr, off, *affine[row])
    hdr[344:348] = b"n+1\0"
    raw = bytes(hdr) + np.asarray(data, dtype="<i2").tobytes(order="F")
    return gzip.compress(raw) if gz else raw


class NiftiLoading(unittest.TestCase):
    def _blob_volume(self, affine, shape=(60, 70, 50), radius_mm=12.0):
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        world = ijk @ affine[:3, :3].T + affine[:3, 3]
        centre_world = affine[:3, :3] @ (np.array(shape) / 2.0 - 0.5) + affine[:3, 3] + np.array([3.0, -2.0, 1.0])
        d = np.linalg.norm(world - centre_world, axis=-1)
        return (np.where(d < radius_mm, 1000, 0)).astype(np.int16), centre_world

    def _check(self, affine):
        data, centre = self._blob_volume(affine)
        vol = t3.load_nifti_volume({"anatomy": make_nifti1_bytes(data, affine, gz=True)}, 2.0, anatomy_is_brain_only=True)
        idx = np.argwhere(vol.labels > 0).mean(axis=0)
        np.testing.assert_allclose(vol.index_to_world(idx), centre, atol=1.5)

    def test_axis_aligned_ras(self):
        A = np.diag([1.0, 1.0, 1.0, 1.0])
        A[:3, 3] = [-30, -40, -20]
        self._check(A)

    def test_flipped_axes_and_anisotropic(self):
        A = np.array([[-1.2, 0, 0, 30], [0, 0.9, 0, -45], [0, 0, 1.5, -25], [0, 0, 0, 1.0]])
        self._check(A)

    def test_oblique_rotation_keeps_world_coordinates(self):
        th = np.radians(12.0)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        A = np.eye(4)
        A[:3, :3] = R @ np.diag([-1.0, 1.0, 1.0])
        A[:3, 3] = [40, -50, -20]
        self._check(A)

    def test_grid_mismatch_rejected(self):
        A = np.eye(4)
        d = np.ones((10, 10, 10), dtype=np.int16)
        B = A.copy()
        B[0, 3] = 25.0
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3.load_nifti_volume({"anatomy": make_nifti1_bytes(d, A), "brain": make_nifti1_bytes(d, B)})

    def test_bad_files_rejected(self):
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3.load_nifti_volume({"anatomy": b"not a nifti file" * 40})
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3.load_nifti_volume({"anatomy": b"short"})

    def test_real_template_if_available(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "demo", "mni152_brain_T1_2mm.nii.gz")
        if not os.path.exists(path):
            self.skipTest("bundled template not present")
        with open(path, "rb") as fh:
            blob = fh.read()
        try:
            vol = t3.load_nifti_volume({"anatomy": blob}, 2.5, anatomy_is_brain_only=True)
        except t3.ThermoSEEG3DError as exc:   # corrupt download / placeholder file: not evidence about the engine
            self.skipTest(f"template present but not a valid NIfTI ({exc})")
        ml = vol.stats_ml()
        self.assertTrue(1000 < ml["brain tissue"] + ml["CSF"] < 2400)
        self.assertEqual(vol.label_at(vol.default_target), t3.LABEL_BRAIN)
        lo, hi = vol.bounds_mm
        self.assertTrue(lo[0] < -60 and hi[0] > 60)        # spans the head in MNI mm


class RiskRegionsAndSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol = t3.make_phantom(3.0)
        cls.brain = cls.vol.labels > 0

    def test_percentile_threshold_selects_top_fraction(self):
        r = t3.heuristic_risk_mask(self.vol.anatomy, self.brain, self.vol.spacing, True, 97, 140, 0)
        self.assertLess(r["fraction_of_brain"], 0.05)
        self.assertGreater(r["fraction_of_brain"], 0.0)
        self.assertTrue(np.all(self.brain[r["mask"]]))

    def test_manual_threshold_above_max_gives_empty_mask(self):
        r = t3.heuristic_risk_mask(self.vol.anatomy, self.brain, self.vol.spacing, False, 97, 255, 0)
        self.assertFalse(r["mask"].any())

    def test_min_volume_removes_small_regions(self):
        free = t3.heuristic_risk_mask(self.vol.anatomy, self.brain, self.vol.spacing, True, 97, 140, 0)["mask"].sum()
        strict = t3.heuristic_risk_mask(self.vol.anatomy, self.brain, self.vol.spacing, True, 97, 140, 2000)["mask"].sum()
        self.assertLessEqual(strict, free)

    def test_safety_buffer_flags_danger(self):
        tgt = self.vol.default_target
        entry = t3.find_entry_on_surface(self.vol, tgt, t3.direction_from_angles(180, 10))
        dist = t3.vessel_distance_map_mm(self.vol)
        clear = t3.analyze_trajectory(self.vol, entry, tgt, dist)["min_vessel_clearance_mm"]
        res = t3.analyze_trajectory(self.vol, entry, tgt, dist, buffer_mm=clear + 5.0)
        self.assertEqual(res["status"], "DANGER")
        self.assertTrue(any("safety buffer" in r for r in res["reasons"]))

    def test_conductivity_scale_lowers_peak(self):
        tg, d = self.vol.default_target, t3.direction_from_angles(180, 10)
        lo = t3.run_litt_comparison(self.vol, tg, d, t3.ThermalParams(conductivity_scale=0.5))["metrics"]
        hi = t3.run_litt_comparison(self.vol, tg, d, t3.ThermalParams(conductivity_scale=2.0))["metrics"]
        self.assertGreater(lo["max_temp_with_csf_c"], hi["max_temp_with_csf_c"])

    def test_peak_rise_is_linear_in_power(self):
        # the target-temperature mode relies on this (constant properties => linear model)
        tg, d = self.vol.default_target, t3.direction_from_angles(180, 10)
        r1 = t3.run_litt_comparison(self.vol, tg, d, t3.ThermalParams(power_w=1.0))["metrics"]["max_temp_with_csf_c"] - 37
        r3 = t3.run_litt_comparison(self.vol, tg, d, t3.ThermalParams(power_w=3.0))["metrics"]["max_temp_with_csf_c"] - 37
        self.assertAlmostEqual(r3 / r1, 3.0, delta=0.02)


class Presets(unittest.TestCase):
    def test_presets_exist_and_run_valid(self):
        vol = t3.make_phantom(3.0)
        tg, d = vol.default_target, t3.direction_from_angles(180, 10)
        for name in t3.PRESETS:
            p = t3.ThermalParams(preset=name, csf_k_multiplier=t3.PRESETS[name]["csf_k_multiplier"],
                                 csf_sink_w_m3k=t3.PRESETS[name]["csf_sink_w_m3k"], power_w=2.0)
            r = t3.run_litt_comparison(vol, tg, d, p)
            self.assertTrue(r["metrics"]["model_valid"], name)
            self.assertTrue(np.isfinite(r["with_csf"]["snapshots"]).all())

    def test_csf_sink_cools_and_zero_sink_matches_conventional_when_no_other_csf_effect(self):
        vol = t3.make_phantom(3.0)
        tg, d = vol.default_target, t3.direction_from_angles(180, 10)
        base = dict(csf_k_multiplier=1.0, power_w=2.0)
        r0 = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(csf_sink_w_m3k=0.0, **base))["metrics"]
        r1 = t3.run_litt_comparison(vol, tg, d, t3.ThermalParams(csf_sink_w_m3k=15000.0, **base))["metrics"]
        self.assertLess(r1["max_temp_with_csf_c"], r0["max_temp_with_csf_c"] + 1e-9)
        self.assertGreater(r1["max_csf_dT_reduction_c"], r0["max_csf_dT_reduction_c"])
        self.assertAlmostEqual(r0["max_temp_without_csf_c"], r1["max_temp_without_csf_c"], places=6)

    def test_params_key_distinguishes_presets(self):
        a = t3.ThermalParams(preset=t3.DEFAULT_PRESET).key()
        b = t3.ThermalParams(preset=t3.PARITY_PRESET).key()
        self.assertNotEqual(a, b)


class SliceMapping(unittest.TestCase):
    def test_pixel_world_roundtrip_all_planes(self):
        rng = np.random.default_rng(1)
        anat = rng.random((40, 50, 30)).astype(np.float32)
        sp, org = np.array([1.5, 1.5, 1.5]), np.array([-30.0, -40.0, -20.0])
        for plane in t3.PLANES:
            img, m = t3.slice_to_image8(anat, sp, org, plane, 12)
            for col, row in ((0, 0), (img.shape[1] - 1, img.shape[0] - 1), (7, 11)):
                w = m.pixel_to_world(col, row)
                c2, r2 = m.world_to_pixel(w)
                self.assertAlmostEqual(c2, col, places=6)
                self.assertAlmostEqual(r2, row, places=6)
                self.assertEqual(t3.world_to_slice_index(plane, w, sp, org), 12)

    def test_blob_world_position_matches_slice_pixels(self):
        # blob at a known world position must appear at the pixel that maps back to that position
        sp, org = np.array([2.0, 2.0, 2.0]), np.array([-60.0, -80.0, -50.0])
        anat = np.zeros((60, 80, 50), np.float32)
        target = np.array([-10.0, 20.0, 14.0])
        i, j, k = np.rint((target - org) / sp).astype(int)
        anat[i - 2:i + 3, j - 2:j + 3, k - 2:k + 3] = 1.0
        for plane in t3.PLANES:
            idx = t3.world_to_slice_index(plane, target, sp, org)
            img, m = t3.slice_to_image8(anat, sp, org, plane, idx)
            rows, cols = np.nonzero(img > 128)
            w = m.pixel_to_world(cols.mean(), rows.mean())
            ax = {"Axial": (0, 1), "Coronal": (0, 2), "Sagittal": (1, 2)}[plane]
            np.testing.assert_allclose(w[list(ax)], target[list(ax)], atol=1.0, err_msg=plane)

    def test_load_anatomy_only_oblique_keeps_world(self):
        th = np.radians(10.0)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        A = np.eye(4)
        A[:3, :3] = R @ np.diag([-1.0, 1.0, 1.0])
        A[:3, 3] = [40, -50, -20]
        shape = (60, 70, 50)
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        world = ijk @ A[:3, :3].T + A[:3, 3]
        centre = A[:3, :3] @ (np.array(shape) / 2.0 - 0.5) + A[:3, 3]
        data = np.where(np.linalg.norm(world - centre, axis=-1) < 12, 1000, 0).astype(np.int16)
        r = t3.load_anatomy_only(make_nifti1_bytes(data, A, gz=True), 2.0)
        idx = np.argwhere(r["anatomy"] > 0.5).mean(axis=0)
        np.testing.assert_allclose(r["origin"] + idx * r["spacing"], centre, atol=1.5)


if __name__ == "__main__":
    unittest.main()