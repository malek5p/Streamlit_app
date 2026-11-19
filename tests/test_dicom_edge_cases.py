import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, JPEG2000, generate_uid
except ImportError:  # pragma: no cover
    pydicom = None

from thermoseeg_dicom import decode_dicom_slice


def make_dicom(modality="MR", rows=32, cols=32, **overrides) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = overrides.pop("_transfer_syntax", ExplicitVRLittleEndian)
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelSpacing = [0.5, 0.5]
    arr = (np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols) * 7) % 4000
    ds.PixelData = arr.tobytes()
    for key, value in overrides.items():
        if value is None:
            if hasattr(ds, key):
                delattr(ds, key)
        else:
            setattr(ds, key, value)
    buf = BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@unittest.skipIf(pydicom is None, "pydicom not installed")
class DicomEdgeCases(unittest.TestCase):
    def test_valid_mr_and_ct_flags(self):
        mr = decode_dicom_slice(make_dicom("MR"))
        ct = decode_dicom_slice(make_dicom("CT"))
        self.assertTrue(mr["metadata"]["roi_heuristic_designed_for_modality"])
        self.assertFalse(ct["metadata"]["roi_heuristic_designed_for_modality"])
        self.assertFalse(ct["metadata"]["quantitative_intensity_preserved"])
        self.assertEqual(mr["gray"].dtype, np.uint8)
        self.assertEqual(mr["pixel_spacing_mm_xy"], (0.5, 0.5))

    def test_pixel_spacing_wrong_length_or_nonnumeric(self):
        for bad in ([0.5], [0.5, 0.5, 0.5], [0.0, 0.5], [-1.0, 0.5]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decode_dicom_slice(make_dicom(PixelSpacing=bad))

    def test_invalid_rescale(self):
        for slope in (0, float("nan"), float("inf")):
            with self.subTest(slope=slope), self.assertRaises(ValueError):
                decode_dicom_slice(make_dicom(RescaleSlope=slope, RescaleIntercept=0))

    def test_orientation_errors(self):
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom(ImageOrientationPatient=[1, 0, 0, 1, 0, 0]))   # not orthogonal
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom(ImageOrientationPatient=[1, 0, 0]))            # wrong length

    def test_wrong_modality_and_color_and_multiframe(self):
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom("US"))
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom(NumberOfFrames=3))

    def test_unsupported_transfer_syntax_gives_clean_error(self):
        # Declares JPEG2000 but stores uncompressed bytes -> decoder must fail with ValueError, not crash.
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom(_transfer_syntax=JPEG2000))

    def test_pixel_limit_enforced_before_decode(self):
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_dicom(rows=64, cols=64), max_pixels=1000)

    def test_constant_image_rejected(self):
        data = make_dicom()
        ds = pydicom.dcmread(BytesIO(data))
        ds.PixelData = np.zeros((32, 32), np.uint16).tobytes()
        buf = BytesIO(); ds.save_as(buf, enforce_file_format=True)
        with self.assertRaises(ValueError):
            decode_dicom_slice(buf.getvalue())

    def test_repeat_decode_is_identical_in_same_environment(self):
        data = make_dicom()
        a, b = decode_dicom_slice(data), decode_dicom_slice(data)
        self.assertTrue(np.array_equal(a["gray"], b["gray"]))


if __name__ == "__main__":
    unittest.main()
