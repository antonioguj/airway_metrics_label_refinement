
from typing import Tuple, Dict, Union
import numpy as np

_EPS = 1.0e-10


def get_vector_two_points(begin_point: Tuple[float, float, float],
                          end_point: Tuple[float, float, float]
                          ) -> Tuple[float, float, float]:
    return (end_point[0] - begin_point[0],
            end_point[1] - begin_point[1],
            end_point[2] - begin_point[2])


def get_norm_vector(in_vector: Tuple[float, float, float]) -> float:
    return np.linalg.norm(in_vector)


def get_point_in_segment(begin_point: Tuple[float, float, float],
                         end_point: Tuple[float, float, float],
                         rel_dist_segm: float
                         ) -> Tuple[float, float, float]:
    vector_segment = get_vector_two_points(begin_point, end_point)
    return (begin_point[0] + rel_dist_segm * vector_segment[0],
            begin_point[1] + rel_dist_segm * vector_segment[1],
            begin_point[2] + rel_dist_segm * vector_segment[2])


def _get_indexes_canditate_inside_blank(point_center: Tuple[float, float, float],
                                        max_dist_2cen: float,
                                        image_size: Tuple[float, float, float],
                                        ) -> np.array:
    min_index_x = int(np.floor(point_center[0] - max_dist_2cen))
    max_index_x = int(np.ceil(point_center[0] + max_dist_2cen))
    min_index_y = int(np.floor(point_center[1] - max_dist_2cen))
    max_index_y = int(np.ceil(point_center[1] + max_dist_2cen))
    min_index_z = int(np.floor(point_center[2] - max_dist_2cen))
    max_index_z = int(np.ceil(point_center[2] + max_dist_2cen))
    min_index_x = max(min_index_x, 0)
    max_index_x = min(max_index_x, image_size[0] - 1)
    min_index_y = max(min_index_y, 0)
    max_index_y = min(max_index_y, image_size[1] - 1)
    min_index_z = max(min_index_z, 0)
    max_index_z = min(max_index_z, image_size[2] - 1)
    indexes_x = np.arange(min_index_x, max_index_x + 1)
    indexes_y = np.arange(min_index_y, max_index_y + 1)
    indexes_z = np.arange(min_index_z, max_index_z + 1)
    return np.stack(np.meshgrid(indexes_x, indexes_y, indexes_z, indexing='ij'), axis=3)


def generate_error_blank_branch_sphere(inout_mask: np.ndarray,
                                       point_center: Tuple[float, float, float],
                                       diam_rad: float
                                       ) -> np.ndarray:
    radius_rad = diam_rad / 2.0
    image_size = inout_mask.shape[::-1]  # get correct format (dim_x, dim_y, dim_z)

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    indexes_candits_inside_blank = _get_indexes_canditate_inside_blank(point_center, radius_rad, image_size)
    # array of indexes, with dims [num_indexes_x, num_indexes_y, num_indexes_z, 3]

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside_blank - point_center

    # distance to center -> compute the norm of relative position vectors
    dist_rel2center_candits_inside = np.linalg.norm(points_rel2center_candits_inside, axis=3)

    # condition for sphere: distance to center is less than radius
    is_indexes_inside_blank = np.abs(dist_rel2center_candits_inside) <= radius_rad
    # array of ['True', 'False'], with 'True' for indexes that are inside the blank

    indexes_inside_blank = indexes_candits_inside_blank[is_indexes_inside_blank]

    # blank error: set '0' to voxels for indexes inside the blank
    (indexes_x_blank, indexes_y_blank, indexes_z_blank) = np.transpose(indexes_inside_blank)
    inout_mask[indexes_z_blank, indexes_y_blank, indexes_x_blank] = 0

    return inout_mask


def generate_error_blank_branch_cylinder(inout_mask: np.ndarray,
                                         point_center: Tuple[float, float, float],
                                         vector_axis: Tuple[float, float, float],
                                         diam_base: float,
                                         length_axis: float
                                         ) -> np.ndarray:
    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = np.array(vector_axis) / norm_vector_axis
    radius_base = diam_base / 2.0
    half_length_axis = length_axis / 2.0
    image_size = inout_mask.shape[::-1] # get correct format (dim_x, dim_y, dim_z)

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    dist_corner_2center = np.sqrt(radius_base ** 2 + half_length_axis ** 2)
    indexes_candits_inside_blank = _get_indexes_canditate_inside_blank(point_center, dist_corner_2center, image_size)
    # array of indexes, with dims [num_indexes_x, num_indexes_y, num_indexes_z, 3]

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside_blank - point_center

    # distance to center, parallel to axis -> dot product of distance vectors with 'vector_axis'
    dist_rel2center_parall_axis_candits = np.dot(points_rel2center_candits_inside, unit_vector_axis)

    # distance to center, perpendicular to axis -> Pythagoras (distance_2center ^2 - distance_2center_parall_axis ^2)
    dist_rel2center_perpen_axis_candits = np.sqrt(np.square(np.linalg.norm(points_rel2center_candits_inside, axis=3))
                                                  - np.square(dist_rel2center_parall_axis_candits) + _EPS)

    # conditions for cylinder: 1) distance to center, parallel to axis, is less than 'half_length_axis'
    #                          2) distance to center, perpendicular to axis, is less than 'radius_base'
    is_indexes_inside_blank_cond1 = np.abs(dist_rel2center_parall_axis_candits) <= half_length_axis
    is_indexes_inside_blank_cond2 = np.abs(dist_rel2center_perpen_axis_candits) <= radius_base

    is_indexes_inside_blank = np.logical_and(is_indexes_inside_blank_cond1, is_indexes_inside_blank_cond2)
    # array of ['True', 'False'], with 'True' for indexes that are inside the blank

    indexes_inside_blank = indexes_candits_inside_blank[is_indexes_inside_blank]

    # blank error: set '0' to voxels for indexes inside the blank
    (indexes_x_blank, indexes_y_blank, indexes_z_blank) = np.transpose(indexes_inside_blank)
    inout_mask[indexes_z_blank, indexes_y_blank, indexes_x_blank] = 0

    return inout_mask
