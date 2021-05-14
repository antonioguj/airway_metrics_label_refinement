
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader, CsvFileReader
from common.genererrors import get_vector_two_points, get_point_in_segment, generate_error_blank_branch


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
        in_voxelnorm_image = np.linalg.norm(in_voxelsize_image)

        num_branches = len(in_air_index_branches)

        # ---------------

        num_branches_error = int(args.prop_branches_error * num_branches)

        if args.random_seed:
            np.random.seed(args.random_seed)

        indexes_branches_gener_error = np.random.choice(range(num_branches), num_branches_error, replace=False)

        for index_brh in indexes_branches_gener_error:

            # mid_point_branch = (in_midpoint_x_branches[index_brh],
            #                     in_midpoint_y_branches[index_brh],
            #                     in_midpoint_z_branches[index_brh])
            begin_point_branch = (in_begpoint_x_branches[index_brh],
                                  in_begpoint_y_branches[index_brh],
                                  in_begpoint_z_branches[index_brh])
            end_point_branch = (in_endpoint_x_branches[index_brh],
                                in_endpoint_y_branches[index_brh],
                                in_endpoint_z_branches[index_brh])

            inner_diam_branch = in_inner_diam_branches[index_brh] / in_voxelnorm_image
            length_branch = in_air_length_branches[index_brh] / in_voxelnorm_image

            # random position along the branch and length, for the generated error
            relpos_blank_branch = np.random.random()
            length_blank_branch = np.random.random() * length_branch

            vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)

            point_bank_branch = get_point_in_segment(begin_point_branch, end_point_branch, relpos_blank_branch)
            #params_blank_shape = {'diam': 2 * inner_diam_branch}
            params_blank_shape = {'vector_axis': vector_axis_branch,
                                  'diam_base': inner_diam_branch,
                                  'length_axis': length_blank_branch}

            inout_airway_mask = generate_error_blank_branch(inout_airway_mask,
                                                            point_bank_branch,
                                                            type_blank_shape='cylinder',
                                                            params_blank_shape=params_blank_shape)
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
    parser.add_argument('--output_dir', type=str, default='./ResultsTest_2/')
    args = parser.parse_args()

    args.output_dir = join_path_names(args.inbasedir, args.output_dir)

    main(args)
