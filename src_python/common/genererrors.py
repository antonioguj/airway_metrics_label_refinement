
from typing import Tuple, List, Dict
import numpy as np


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
                                        max_dist: float
                                        ) -> np.array:
    min_index_x = int(np.floor(point_center[0] - max_dist))
    max_index_x = int(np.ceil(point_center[0] + max_dist))
    min_index_y = int(np.floor(point_center[1] - max_dist))
    max_index_y = int(np.ceil(point_center[1] + max_dist))
    min_index_z = int(np.floor(point_center[2] - max_dist))
    max_index_z = int(np.ceil(point_center[2] + max_dist))
    indexes_x = np.arange(min_index_x, max_index_x + 1)
    indexes_y = np.arange(min_index_y, max_index_y + 1)
    indexes_z = np.arange(min_index_z, max_index_z + 1)
    return np.stack(np.meshgrid(indexes_x, indexes_y, indexes_z, indexing='ij'), axis=3)


def _get_indexes_inside_blank_sphere(indexes_candits_inside_blank: np.ndarray,
                                     point_center: Tuple[float, float, float],
                                     diam_sphere: float
                                     ) -> np.ndarray:
    # relative distance of candidate indexes to center
    locs_rel2center_candits = indexes_candits_inside_blank - point_center

    radius_sphere = diam_sphere / 2.0

    # condition for sphere: if distance to center is less than radius
    is_indexes_inside_blank = np.linalg.norm(locs_rel2center_candits, axis=3) <= radius_sphere
    # array of ['True', 'False'], with 'True' for indexes that are inside the blank

    return indexes_candits_inside_blank[is_indexes_inside_blank]


def _get_indexes_inside_blank_cylinder(indexes_candits_inside_blank: np.ndarray,
                                       point_center: Tuple[float, float, float],
                                       vector_axis: Tuple[float, float, float],
                                       diam_base: float,
                                       length_axis: float
                                       ) -> np.ndarray:
    # relative distance of candidate indexes to center
    locs_rel2center_candits = indexes_candits_inside_blank - point_center

    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = vector_axis / norm_vector_axis
    radius_base = diam_base / 2.0
    half_length_axis = length_axis / 2.0

    # conditions for cylinder: 1) if distance to center, parallel to axis, is less than length_axis
    #                          2) if distance to center, perpendicular to axis, is less than radius_base
    dist_rel2center_parall_axis_candits = np.dot(locs_rel2center_candits, unit_vector_axis)

    is_indexes_inside_blank_parall = dist_rel2center_parall_axis_candits <= half_length_axis
    is_indexes_inside_blank_perpen = np.sqrt(np.square(np.linalg.norm(locs_rel2center_candits, axis=3))
                                             - np.square(dist_rel2center_parall_axis_candits)) <= radius_base

    is_indexes_inside_blank = np.logical_and(is_indexes_inside_blank_parall, is_indexes_inside_blank_perpen)
    # array of ['True', 'False'], with 'True' for indexes that are inside the blank

    return indexes_candits_inside_blank[is_indexes_inside_blank]


def _get_max_distance_2center_sphere(diam_sphere: float) -> float:
    return diam_sphere / 2.0


def _get_max_distance_2center_cylinder(diam_base: float, length_axis: float) -> float:
    radius_base = diam_base / 2.0
    half_length_axis = length_axis / 2.0
    return np.sqrt(radius_base**2 + half_length_axis**2)


def generate_error_blank_branch(inout_mask: np.ndarray,
                                point_center: Tuple[float, float, float],
                                type_blank_shape: str,
                                params_blank_shape: Dict[str, float]
                                ) -> np.ndarray:
    if type_blank_shape not in ['sphere', 'cylinder']:
        message = 'ERROR: input \'type_blank_shape\' not valid. Options available: [sphere, cylinder]... SKIP'
        print(message)
        return inout_mask

    # candidates: subset of mask indexes where to check condition for error shape -> to save time
    if type_blank_shape == 'sphere':
        diam_blank = params_blank_shape['diam']
        max_dist2cen_blank = _get_max_distance_2center_sphere(diam_blank)
    elif type_blank_shape == 'cylinder':
        diam_base_blank = params_blank_shape['diam_base']
        length_axis_blank = params_blank_shape['length_axis']
        max_dist2cen_blank = _get_max_distance_2center_cylinder(diam_base_blank, length_axis_blank)

    indexes_candits_inside_blank = _get_indexes_canditate_inside_blank(point_center, max_dist2cen_blank)
    # array of indexes, with dims [num_indexes_x, num_indexes_y, num_indexes_z, 3]

    # get points from the candidate indexes that are inside the blank of the given shape
    if type_blank_shape == 'sphere':
        diam_blank = params_blank_shape['diam']
        indexes_inside_blank = _get_indexes_inside_blank_sphere(indexes_candits_inside_blank, point_center,
                                                                diam_blank)
    elif type_blank_shape == 'cylinder':
        vector_axis = params_blank_shape['vector_axis']
        diam_base_blank = params_blank_shape['diam_base']
        length_axis_blank = params_blank_shape['length_axis']
        indexes_inside_blank = _get_indexes_inside_blank_cylinder(indexes_candits_inside_blank, point_center,
                                                                  vector_axis, diam_base_blank, length_axis_blank)

    (indexes_x_error, indexes_y_error, indexes_z_error) = np.transpose(indexes_inside_blank)

    # blank error: set '0' to voxels for indexes inside the blank
    inout_mask[indexes_z_error, indexes_y_error, indexes_x_error] = 0
    return inout_mask
