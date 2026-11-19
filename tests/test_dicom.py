from __future__ import annotations

import io
import unittest

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from thermoseeg_dicom import decode_dicom_slice


def make_ct(modality="CT", frames=None, spacing=("0.8", "0.6")):
    pixels = np.arange(64, dtype=np.int16).reshape(8, 8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = list(spacing)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.RescaleSlope, ds.RescaleIntercept = "2", "-1000"
    ds.WindowCenter, ds.WindowWidth = "-940", "80"
    if frames:
        ds.NumberOfFrames = str(frames)
        ds.PixelData = np.stack([pixels] * frames).astype(np.int16).tobytes()
    else:
        ds.PixelData = pixels.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


class DicomTests(unittest.TestCase):
    def test_modality_voi_and_spacing(self):
        result = decode_dicom_slice(make_ct())
        self.assertEqual(result["gray"].shape, (8, 8))
        self.assertEqual(result["pixel_spacing_mm_xy"], (0.6, 0.8))
        self.assertTrue(result["metadata"]["modality_transform_applied"])
        self.assertTrue(result["metadata"]["voi_transform_applied"])

    def test_unsupported_modality_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected MR or CT"):
            decode_dicom_slice(make_ct(modality="OT"))

    def test_multiframe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multi-frame"):
            decode_dicom_slice(make_ct(frames=2))

    def test_invalid_pixel_spacing_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_dicom_slice(make_ct(spacing=("0.8", "0.0")))

    def test_oversized_dimensions_are_rejected_before_pixel_decode(self):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(make_ct()))
        ds.Rows, ds.Columns = 4097, 2
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, "dimensions exceed"):
            decode_dicom_slice(buf.getvalue())

    def test_zero_rescale_slope_is_rejected(self):
        import pydicom
        raw = make_ct()
        ds = pydicom.dcmread(io.BytesIO(raw))
        ds.RescaleSlope = "0"
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, "RescaleSlope"):
            decode_dicom_slice(buf.getvalue())

    def test_non_dicom_raster_falls_through(self):
        self.assertIsNone(decode_dicom_slice(b"not a DICOM file"))


if __name__ == "__main__":
    unittest.main()
