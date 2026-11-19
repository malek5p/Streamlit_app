"""Tests for the scan-upload converter (ZIP / DICOM series / JPG-PNG stacks -> NIfTI volume)."""
import io
import random
import unittest
import zipfile

import numpy as np

import thermoseeg_3d as t3

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.T[::-1]).save(buf, "PNG")      # (cols, rows) -> image rows x cols
    return buf.getvalue()


def _blob_stack(n_slices=24):
    shape = (64, 64, n_slices)
    g = np.stack(np.meshgrid(*[np.arange(k) for k in shape], indexing="ij"), -1)
    c = np.array([32, 32, n_slices / 2])
    img = np.where((((g - c) / np.array([22, 26, n_slices * 0.4])) ** 2).sum(-1) < 1, 170, 0).astype(np.uint8)
    return [(f"slice_{k:03d}.png", _png(img[:, :, k])) for k in range(n_slices)]


@unittest.skipIf(Image is None, "Pillow not installed")
class ImageStacks(unittest.TestCase):
    def test_zip_of_pngs_becomes_a_loadable_volume_with_requested_spacing(self):
        files = _blob_stack()
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            for n, b in files:
                z.writestr(n, b)
        blob, notes, kind = t3.normalise_scan_upload([("scan.zip", zb.getvalue())], 2.0, 4.0)
        self.assertEqual(kind, "images")
        vol = t3.load_nifti_volume({"anatomy": blob}, 2.0, anatomy_is_brain_only=True)
        self.assertGreater(vol.stats_ml()["brain tissue"], 50.0)
        self.assertTrue(any("ASSUMED" in n for n in notes))

    def test_loose_files_are_sorted_naturally(self):
        files = _blob_stack(12)
        shuffled = files[:]
        random.Random(3).shuffle(shuffled)
        a, _, _ = t3.normalise_scan_upload(files, 2.0, 4.0)
        b, _, _ = t3.normalise_scan_upload(shuffled, 2.0, 4.0)
        self.assertEqual(a, b)

    def test_single_image_is_rejected_with_clear_message(self):
        with self.assertRaisesRegex(t3.ThermoSEEG3DError, "cannot form a 3D volume"):
            t3.normalise_scan_upload([_blob_stack(8)[0]])

    def test_mixed_sizes_rejected(self):
        files = _blob_stack(8)
        files[3] = ("slice_003.png", _png(np.zeros((32, 32), np.uint8)))
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3.normalise_scan_upload(files)


class Containers(unittest.TestCase):
    def test_nifti_inside_zip_is_used(self):
        nii = t3.nifti1_bytes(np.random.default_rng(0).random((12, 12, 12)), np.eye(4))
        zb = io.BytesIO()
        with zipfile.ZipFile(zb, "w") as z:
            z.writestr("study/anat.nii.gz", nii)
        blob, _, kind = t3.normalise_scan_upload([("p.zip", zb.getvalue())])
        self.assertEqual(kind, "nifti")
        self.assertEqual(blob, nii)

    def test_bad_inputs(self):
        for files in ([], [("x.txt", b"hello")], [("bad.zip", b"PK\x03\x04junk")]):
            with self.subTest(files=files), self.assertRaises(t3.ThermoSEEG3DError):
                t3.normalise_scan_upload(files)


class DicomGeometry(unittest.TestCase):
    def test_oblique_series_keeps_world_position(self):
        th = np.radians(12.0)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        row_d, col_d = R @ [1, 0, 0], R @ [0, 1, 0]
        nrm = np.cross(row_d, col_d)
        ps, rows, cols, ns, spc = (0.9, 1.2), 70, 60, 50, 1.5
        ipp0 = np.array([-30.0, -40.0, -20.0])
        cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
        centre = ipp0 + row_d * ps[1] * 30 + col_d * ps[0] * 35 + nrm * spc * 25
        slices = []
        for k in range(ns):
            ipp = ipp0 + nrm * spc * k
            wp = ipp[None, None, :] + cc[..., None] * row_d * ps[1] + rr[..., None] * col_d * ps[0]
            slices.append({"pixels": (np.linalg.norm(wp - centre, axis=-1) < 12) * 1000.0, "ipp": list(ipp),
                           "iop": list(row_d) + list(col_d), "ps": list(ps)})
        random.Random(1).shuffle(slices)                  # order must not matter
        data, aff, notes = t3._volume_from_dicom_slices(slices)
        vol = t3.load_nifti_volume({"anatomy": t3.nifti1_bytes(data, aff)}, 2.0, anatomy_is_brain_only=True)
        got = vol.index_to_world(np.argwhere(vol.labels > 0).mean(axis=0))
        want = np.array([-centre[0], -centre[1], centre[2]])            # LPS -> RAS
        np.testing.assert_allclose(got, want, atol=1.5)

    def test_too_few_or_duplicate_slices_rejected(self):
        one = {"pixels": np.zeros((8, 8)), "ipp": [0, 0, 0], "iop": [1, 0, 0, 0, 1, 0], "ps": [1, 1]}
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3._volume_from_dicom_slices([one])
        with self.assertRaises(t3.ThermoSEEG3DError):
            t3._volume_from_dicom_slices([dict(one) for _ in range(5)])


if __name__ == "__main__":
    unittest.main()