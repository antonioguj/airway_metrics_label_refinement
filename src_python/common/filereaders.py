
from typing import Tuple, Any

import numpy as np
import nibabel as nib


class ImageFileReader(object):

    @classmethod
    def get_image_voxelsize(cls, filename: str) -> Tuple[float, float, float]:
        affine = nib.load(filename).affine
        return tuple(np.abs(np.diag(affine)[:3]))

    @classmethod
    def get_image_metadata_info(cls, filename: str) -> Any:
        return nib.load(filename).affine

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        out_image = nib.load(filename).get_data()
        return np.swapaxes(out_image, 0, 2)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        affine = kwargs['metadata'] if 'metadata' in kwargs.keys() else None
        in_image = np.swapaxes(in_image, 0, 2)
        nib_image = nib.Nifti1Image(in_image, affine)
        nib.save(nib_image, filename)