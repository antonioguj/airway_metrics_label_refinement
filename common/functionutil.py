
from typing import List, Union
import numpy as np
import glob
import os
import re
import shutil
import sys
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import skeletonize_3d
from skimage.measure import label


def makedir(dirname: str) -> bool:
    dirname = dirname.strip().rstrip("\\")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return True
    else:
        return False


def makelink(src_file: str, dest_link: str) -> None:
    os.symlink(src_file, dest_link)


def copyfile(src_file: str, dest_file: str) -> None:
    shutil.copyfile(src_file, dest_file)


def join_path_names(pathname_1: str, pathname_2: str) -> str:
    return os.path.join(pathname_1, pathname_2)


def basename(pathname: str) -> str:
    return os.path.basename(pathname)


def dirname(pathname: str) -> str:
    return os.path.dirname(pathname)


def basename_filenoext(filename: str) -> str:
    return basename(filename).replace('.nii.gz', '')


def list_files_dir(dirname: str, filename_pattern: str = '*') -> List[str]:
    return sorted(glob.glob(join_path_names(dirname, filename_pattern)))


def list_dirs_dir(dirname: str, dirname_pattern: str = '*') -> List[str]:
    return list_files_dir(dirname, dirname_pattern)


def get_substring_filename(filename: str, pattern_search: str) -> Union[str, None]:
    sre_substring_filename = re.search(pattern_search, filename)
    if sre_substring_filename:
        return sre_substring_filename.group(0)
    else:
        return None


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
