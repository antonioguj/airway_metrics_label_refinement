
from typing import Tuple, List
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader, CsvFileReader


def get_vector_two_points(point_1: Tuple[float, float, float],
                          point_2: Tuple[float, float, float]
                          ) -> Tuple[float, float, float]:
    return (point_2[0] - point_1[0],
            point_2[1] - point_1[1],
            point_2[2] - point_1[2])


def get_norm_vector(in_vector: Tuple[float, float, float]) -> float:
    return np.sqrt(in_vector[0] * in_vector[0] + in_vector[1] * in_vector[1] + in_vector[2] * in_vector[2])


def get_loc_between_two_points(point_1: Tuple[float, float, float],
                               point_2: Tuple[float, float, float],
                               reldist_p1_p2: float
                               ) -> Tuple[float, float, float]:
    vector_p1_p2 = get_vector_two_points(point_1, point_2)
    return (point_1[0] + reldist_p1_p2 * vector_p1_p2[0],
            point_1[1] + reldist_p1_p2 * vector_p1_p2[1],
            point_1[2] + reldist_p1_p2 * vector_p1_p2[2])


def get_canditate_indexes_gen_error(loc_center_error: Tuple[float, float, float],
                                    max_dist_error: float
                                    ) -> Tuple[np.array]:
    min_index_x = int(np.floor(loc_center_error[0] - max_dist_error))
    max_index_x = int(np.ceil(loc_center_error[0] + max_dist_error))
    min_index_y = int(np.floor(loc_center_error[1] - max_dist_error))
    max_index_y = int(np.ceil(loc_center_error[1] + max_dist_error))
    min_index_z = int(np.floor(loc_center_error[2] - max_dist_error))
    max_index_z = int(np.ceil(loc_center_error[2] + max_dist_error))
    indexes_x = np.arange(min_index_x, max_index_x + 1)
    indexes_y = np.arange(min_index_y, max_index_y + 1)
    indexes_z = np.arange(min_index_z, max_index_z + 1)
    return np.stack(np.meshgrid(indexes_x, indexes_y, indexes_z, indexing='ij'), axis=3)


def generate_error_sphere_shape_inside_branch(inout_mask: np.ndarray,
                                              loc_center_error: Tuple[float, float, float],
                                              diam_spher_error: float
                                              ) -> np.ndarray:
    # candidates: subset of mask indexes where to check condition for error shape -> to save time
    indexes_candits_gen_error = get_canditate_indexes_gen_error(loc_center_error,
                                                                diam_spher_error / 2.0)
    # array of indexes, with dims [num_indexes_x, num_indexes_y, num_indexes_z, 3]

    # condition for sphere shape -> if relative distance of candidate indexes to center is smaller than radius
    relpos_2center_candits_gen_error = indexes_candits_gen_error - loc_center_error
    radius_spher_error = diam_spher_error / 2.0

    is_indexes_inside_error = np.linalg.norm(relpos_2center_candits_gen_error, axis=3) < radius_spher_error
    # array of ['True', 'False'], with 'True' for indexes that meet the error shape condition

    indexes_generate_error = indexes_candits_gen_error[is_indexes_inside_error]
    # array with only indexes that do meet the error shape condition

    (indexes_x_error, indexes_y_error, indexes_z_error) = np.transpose(indexes_generate_error)

    # set to '0' the voxels for indexes that belong to error
    inout_mask[indexes_z_error, indexes_y_error, indexes_x_error] = 0
    return inout_mask


def main(args):

    # SETTINGS
    input_airway_masks_dir = join_path_names(args.inbasedir, './Airways')
    input_airway_measures_dir = join_path_names(args.inbasedir, './AirwayMeasurements')
    input_images_info_file = join_path_names(args.inbasedir, './images_info.csv')

    def get_casename_filename(in_filename: str):
        return basename(in_filename).replace('_manual-airways.nii.gz', '')
    # --------

    makedir(args.output_dir)

    list_input_airway_masks = list_files_dir(input_airway_masks_dir)
    # list_input_airway_measures = list_files_dir(input_airway_measures_dir)

    in_images_info = CsvFileReader.get_data(input_images_info_file)

    in_images_voxelsize_info = OrderedDict()
    for i, i_casename in enumerate(in_images_info['casename']):
        voxel_size_x = in_images_info['voxel_size_x'][i]
        voxel_size_y = in_images_info['voxel_size_y'][i]
        voxel_size_z = in_images_info['voxel_size_z'][i]
        in_images_voxelsize_info[i_casename] = (voxel_size_x, voxel_size_y, voxel_size_z)
    # endfor

    # **********************

    for in_airway_mask_file in list_input_airway_masks:
        print("\nInput: \'%s\'..." % (basename(in_airway_mask_file)))
        in_casename = get_casename_filename(in_airway_mask_file)

        in_airway_measures_file = in_casename + '_ResultsPerBranch.csv'
        in_airway_measures_file = join_path_names(input_airway_measures_dir, in_airway_measures_file)
        print("With measures from: \'%s\'..." % (basename(in_airway_measures_file)))

        in_metadata_file = NiftiFileReader.get_image_metadata_info(in_airway_mask_file)

        inout_airway_mask = NiftiFileReader.get_image(in_airway_mask_file)

        in_airway_measures = CsvFileReader.get_data(in_airway_measures_file)

        # ---------------

        in_air_index_branches  = in_airway_measures['airway_ID']
        in_midpoint_x_branches = in_airway_measures['midPoint_x']
        in_midpoint_y_branches = in_airway_measures['midPoint_y']
        in_midpoint_z_branches = in_airway_measures['midPoint_z']
        in_inner_diam_branches = in_airway_measures['d_inner_global']
        in_air_length_branches = in_airway_measures['airway_length']
        in_generation_branches = in_airway_measures['generation']
        in_begpoint_x_branches = in_airway_measures['begPoint_x']
        in_endpoint_x_branches = in_airway_measures['endPoint_x']
        in_begpoint_y_branches = in_airway_measures['begPoint_y']
        in_endpoint_y_branches = in_airway_measures['endPoint_y']
        in_begpoint_z_branches = in_airway_measures['begPoint_z']
        in_endpoint_z_branches = in_airway_measures['endPoint_z']

        in_voxelsize_image = in_images_voxelsize_info[in_casename]
        in_voxelnorm_image = get_norm_vector(in_voxelsize_image)

        num_branches = len(in_air_index_branches)

        # ---------------

        num_branches_error = int(args.prop_branches_error * num_branches)

        if args.random_seed:
            np.random.seed(args.random_seed)

        random_indexes_branches_error = np.random.choice(range(num_branches), num_branches_error, replace=False)

        for index_brh in random_indexes_branches_error:

            loc_midpoint_branch = (in_midpoint_x_branches[index_brh],
                                   in_midpoint_y_branches[index_brh],
                                   in_midpoint_z_branches[index_brh])
            loc_begpoint_branch = (in_begpoint_x_branches[index_brh],
                                   in_begpoint_y_branches[index_brh],
                                   in_begpoint_z_branches[index_brh])
            loc_endpoint_branch = (in_endpoint_x_branches[index_brh],
                                   in_endpoint_y_branches[index_brh],
                                   in_endpoint_z_branches[index_brh])

            inner_diam_branch = in_inner_diam_branches[index_brh] / in_voxelnorm_image
            length_branch = in_air_length_branches[index_brh] / in_voxelnorm_image

            # random position along the branch and length, for the generated error
            random_relpos_error_branch = np.random.random()
            #random_length_error_branch = np.random.random() * length_branch
            random_length_error_branch = inner_diam_branch * 3

            # vector_direc_branch = get_vector_two_points(loc_begpoint_branch,
            #                                             loc_endpoint_branch)

            loc_point_error_branch = get_loc_between_two_points(loc_begpoint_branch,
                                                                loc_endpoint_branch,
                                                                random_relpos_error_branch)

            inout_airway_mask = generate_error_sphere_shape_inside_branch(inout_airway_mask,
                                                                          loc_point_error_branch,
                                                                          random_length_error_branch)
        # endfor

        out_airway_error_mask_file = in_casename + '_airways-errors.nii.gz'
        out_airway_error_mask_file = join_path_names(args.output_dir, out_airway_error_mask_file)
        print("Output: \'%s\'..." % (basename(out_airway_error_mask_file)))

        NiftiFileReader.write_image(out_airway_error_mask_file, inout_airway_mask, metadata=in_metadata_file)
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbasedir', type=str, default='.')
    parser.add_argument('--prop_branches_error', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=2017)
    parser.add_argument('--output_dir', type=str, default='./ResultsTest/')
    args = parser.parse_args()

    args.output_dir = join_path_names(args.inbasedir, args.output_dir)

    main(args)
