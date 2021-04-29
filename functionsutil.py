
from typing import List, Tuple, Any

import numpy as np
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import skeletonize_3d
from skimage.measure import label
import nibabel as nib
import glob
import os
import sys


def makedir(dirname: str) -> bool:
    dirname = dirname.strip().rstrip("\\")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return True
    else:
        return False


def join_path_names(pathname_1: str, pathname_2: str) -> str:
    return os.path.join(pathname_1, pathname_2)


def basename(pathname: str) -> str:
    return os.path.basename(pathname)


def list_files_dir(dirname: str, filename_pattern: str = '*') -> List[str]:
    return sorted(glob.glob(join_path_names(dirname, filename_pattern)))


def handle_error_message(message: str) -> None:
    print("ERROR: %s... EXIT" % (message))
    sys.exit(0)


def compute_thresholded_image(in_image: np.ndarray, value_threshold: float) -> np.ndarray:
    return np.where(in_image > value_threshold, 1.0, 0.0).astype(np.uint8)


def compute_eroded_mask(in_image: np.ndarray, num_iters: int) -> np.ndarray:
    return binary_erosion(in_image, iterations=num_iters).astype(in_image.dtype)


def compute_dilated_mask(in_image: np.ndarray, num_iters: int) -> np.ndarray:
    return binary_dilation(in_image, iterations=num_iters).astype(in_image.dtype)


def compute_merged_two_masks(in_image_1: np.ndarray, in_image_2: np.ndarray) -> np.ndarray:
    out_image = in_image_1 + in_image_2
    return np.clip(out_image, 0, 1)


def compute_substracted_two_masks(in_image_1: np.ndarray, in_image_2: np.ndarray) -> np.ndarray:
    out_image = (in_image_1 - in_image_2).astype(np.int8)
    return np.clip(out_image, 0, 1).astype(in_image_1.dtype)


def compute_multiplied_two_masks(in_image_1: np.ndarray, in_image_2: np.ndarray) -> np.ndarray:
    out_image = np.multiply(in_image_1, in_image_2)
    return np.clip(out_image, 0, 1)


def compute_centrelines_mask(in_image: np.ndarray) -> np.ndarray:
    return skeletonize_3d(in_image.astype(np.uint8))


def compute_largest_connected_tree(in_image: np.ndarray, connectivity_dim: int) -> np.ndarray:
    (all_regions, num_regs) = label(in_image, connectivity=connectivity_dim, background=0, return_num=True)

    # retrieve the conn. region with the largest volume
    max_vol_regs = 0.0
    out_image = None
    for ireg in range(num_regs):
        # volume = count voxels for the the conn. region with label "i+1"
        iconreg_vol = np.count_nonzero(all_regions == ireg + 1)
        if iconreg_vol > max_vol_regs:
            # extract the conn. region with label "i+1"
            out_image = np.where(all_regions == ireg + 1, 1, 0).astype(in_image.dtype)
            max_vol_regs = iconreg_vol

    return out_image


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
