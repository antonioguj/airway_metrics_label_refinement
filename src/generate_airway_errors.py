
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader, CsvFileReader
from common.genererrors import get_vector_two_points, get_norm_vector, get_distance_two_points, \
    get_point_inside_segment, generate_error_blank_branch_cylinder

IS_EXCLUDE_SMALL_BRANCHES_ERROR_T1 = True
MIN_LENGTH_BRANCH_CANDITS_ERROR_T1 = 6.0
MIN_GENERATION_ERROR_T1 = 4
INFLATE_DIAM_ERROR_T1 = 4.0
MAX_DIAM_ERROR_T1 = 30.0
MIN_LENGTH_ERROR_T1 = 1.0

INFLATE_DIAM_ERROR_T2 = 6.0
MAX_DIAM_ERROR_T2 = 30.0


def main(args):

    # SETTINGS
    input_airway_masks_dir = join_path_names(args.inbasedir, './Airways')
    input_airway_measures_dir = join_path_names(args.inbasedir, './AirwayMeasurements')
    input_images_info_file = join_path_names(args.inbasedir, './images_info.csv')

    def get_casename_filename(in_filename: str):
        return basename(in_filename).replace('_manual-airways.nii.gz', '')
    # --------

    makedir(args.output_dir)

    list_input_airway_mask_files = list_files_dir(input_airway_masks_dir)
    # list_input_airway_measures_files = list_files_dir(input_airway_measures_dir)

    in_images_info = CsvFileReader.get_data(input_images_info_file)

    in_images_voxelsize_info = OrderedDict()
    for i, i_casename in enumerate(in_images_info['casename']):
        voxel_size_x = in_images_info['voxel_size_x'][i]
        voxel_size_y = in_images_info['voxel_size_y'][i]
        voxel_size_z = in_images_info['voxel_size_z'][i]
        in_images_voxelsize_info[i_casename] = (voxel_size_x, voxel_size_y, voxel_size_z)
    # endfor

    if args.is_gener_error_type1:
        print("Generate errors of Type 1: blanking small regions in random branches...")
        print("Cylindres: 1) center: random position along the branch...")
        print("           2) diameter: the branch diameter (inflated %sx times)..." % (INFLATE_DIAM_ERROR_T1))
        print("           3) length: random between min. %smm and the branch length..." % (MIN_LENGTH_ERROR_T1))

    if args.is_gener_error_type2:
        print("Generate errors of Type 2: blanking partially random (most of) terminal branches...")
        print("Cylindres: 1) center: random position in the first half of the branch...")
        print("           2) diameter: the branch diameter (inflated %sx times)..." % (INFLATE_DIAM_ERROR_T2))
        print("           3) length: distance between start blank and end of the branch...")

    if args.is_output_error_measures:
        print("Output measures of Generated errors (info: i) branch index, ii) center location and iii) length...")

    # **********************

    # **********************

    for in_airway_mask_file in list_input_airway_mask_files:
        print("\n\nInput: \'%s\'..." % (basename(in_airway_mask_file)))
        in_casename = get_casename_filename(in_airway_mask_file)

        in_airway_measures_file = in_casename + '_ResultsPerBranch.csv'
        in_airway_measures_file = join_path_names(input_airway_measures_dir, in_airway_measures_file)
        print("With measures from: \'%s\'..." % (basename(in_airway_measures_file)))

        in_metadata_file = NiftiFileReader.get_image_metadata_info(in_airway_mask_file)

        inout_airway_mask = NiftiFileReader.get_image(in_airway_mask_file)

        if args.is_test_error_shapes:
            inout_airway_mask = np.ones_like(inout_airway_mask)

        in_airway_measures = CsvFileReader.get_data(in_airway_measures_file)

        # in_airway_ids_branches = np.array(in_airway_measures['airway_ID'])
        # in_midpoint_x_branches = np.array(in_airway_measures['midPoint_x'])
        # in_midpoint_y_branches = np.array(in_airway_measures['midPoint_y'])
        # in_midpoint_z_branches = np.array(in_airway_measures['midPoint_z'])
        in_inner_diam_branches = np.array(in_airway_measures['d_inner_global'])
        in_airlength_branches = np.array(in_airway_measures['airway_length'])
        in_generation_branches = np.array(in_airway_measures['generation'])
        # in_parent_id_branches = np.array(in_airway_measures['parent_ID'])
        in_children_id_branches = np.array(in_airway_measures['childrenID'])
        in_begpoint_x_branches = np.array(in_airway_measures['begPoint_x'])
        in_endpoint_x_branches = np.array(in_airway_measures['endPoint_x'])
        in_begpoint_y_branches = np.array(in_airway_measures['begPoint_y'])
        in_endpoint_y_branches = np.array(in_airway_measures['endPoint_y'])
        in_begpoint_z_branches = np.array(in_airway_measures['begPoint_z'])
        in_endpoint_z_branches = np.array(in_airway_measures['endPoint_z'])

        in_voxelsize_image = in_images_voxelsize_info[in_casename]
        in_voxelnorm_image = get_norm_vector(in_voxelsize_image)
        # # normalize the airway inner diameter and length measures
        # in_inner_diam_branches /= in_voxelnorm_image
        # in_airlength_branches /= in_voxelnorm_image

        num_branches = len(in_inner_diam_branches)
        print("Num total branches: %s..." % (num_branches))

        # --------------------
        if args.is_output_error_measures:
            out_dict_air_error_measures = OrderedDict()
            out_dict_air_error_measures['airway_id'] = []
            out_dict_air_error_measures['type_error'] = []
            out_dict_air_error_measures['loc_center_x'] = []
            out_dict_air_error_measures['loc_center_y'] = []
            out_dict_air_error_measures['loc_center_z'] = []
            out_dict_air_error_measures['diam_blank'] = []
            out_dict_air_error_measures['length_blank'] = []

        # --------------------

        if args.random_seed:
            np.random.seed(args.random_seed)

        # *********************************************************
        # Type 1 Errors : Blanking small regions in random branches
        # *********************************************************

        if args.is_gener_error_type1:
            print('\nGenerate errors of type1: blanking small regions in random branches...')

            if IS_EXCLUDE_SMALL_BRANCHES_ERROR_T1:
                print("Consider only branches of length larger than \'%s\' voxels to generate errors..."
                      % (MIN_LENGTH_BRANCH_CANDITS_ERROR_T1))

                index_excluded_branches = []

                for ibrh in range(num_branches):
                    begin_point_branch = (in_begpoint_x_branches[ibrh],
                                          in_begpoint_y_branches[ibrh],
                                          in_begpoint_z_branches[ibrh])
                    end_point_branch = (in_endpoint_x_branches[ibrh],
                                        in_endpoint_y_branches[ibrh],
                                        in_endpoint_z_branches[ibrh])
                    length_branch = get_distance_two_points(begin_point_branch, end_point_branch)

                    if length_branch < MIN_LENGTH_BRANCH_CANDITS_ERROR_T1:
                        index_excluded_branches.append(ibrh)
                # endfor

                num_excluded_branches = len(index_excluded_branches)
                print("Num branches \'%s\' (out of total \'%s\') excluded because they are too short... "
                      % (num_excluded_branches, num_branches))
            else:
                index_excluded_branches = []
                num_excluded_branches = 0

            # --------------------

            # exclude the larger main branches (with lowest generation number)
            index_excluded_branches_more = [ibrh for ibrh, igen in enumerate(in_generation_branches)
                                            if igen < MIN_GENERATION_ERROR_T1]
            index_excluded_branches += index_excluded_branches_more

            # get candidate branches, without the excluded ones
            index_candits_branches = list(range(num_branches))
            index_candits_branches = [ibrh for ibrh in index_candits_branches if ibrh not in index_excluded_branches]
            num_candits_branches = len(index_candits_branches)

            # As sample probability, use AIRWAY GENERATION NUMBER, so that terminal branches have more likely errors
            sample_probs_generation = [in_generation_branches[ibrh] - MIN_GENERATION_ERROR_T1 + 1
                                       for ibrh in index_candits_branches]
            sample_probs_generation = np.array(sample_probs_generation) / np.sum(sample_probs_generation)

            num_branches_error = int(args.prop_branches_error_type1 * num_candits_branches)
            print("Num branches with errors type1: %s..." % (num_branches_error))

            index_branches_generate_error = np.random.choice(index_candits_branches, num_branches_error,
                                                             replace=False, p=sample_probs_generation)
            index_branches_generate_error = np.sort(index_branches_generate_error)

            # --------------------

            for ibrh in index_branches_generate_error:
                begin_point_branch = (in_begpoint_x_branches[ibrh],
                                      in_begpoint_y_branches[ibrh],
                                      in_begpoint_z_branches[ibrh])
                end_point_branch = (in_endpoint_x_branches[ibrh],
                                    in_endpoint_y_branches[ibrh],
                                    in_endpoint_z_branches[ibrh])
                inner_diam_branch = in_inner_diam_branches[ibrh]

                vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)
                length_branch = get_norm_vector(vector_axis_branch)

                # position center blank: random along the branch
                rel_pos_center_blank = np.random.random()
                loc_center_blank = get_point_inside_segment(begin_point_branch, end_point_branch, rel_pos_center_blank)

                # diameter base blank: the branch diameter (inflated several times)
                diam_base_blank = INFLATE_DIAM_ERROR_T1 * inner_diam_branch
                if diam_base_blank > MAX_DIAM_ERROR_T1:
                    print("Warning: branch \'%s\' with too large diam: \'%s\'... Clipping it to: \'%s\'..."
                          % (ibrh, diam_base_blank, MAX_DIAM_ERROR_T1))
                diam_base_blank = min(diam_base_blank, MAX_DIAM_ERROR_T1)

                # length blank: random between min. (1 voxel) and the branch length
                length_axis_blank = np.random.random() * length_branch
                length_axis_blank = max(length_axis_blank, MIN_LENGTH_ERROR_T1)

                inout_airway_mask = generate_error_blank_branch_cylinder(inout_airway_mask,
                                                                         loc_center_blank,
                                                                         vector_axis_branch,
                                                                         diam_base_blank,
                                                                         length_axis_blank)

                # ----------
                if args.is_output_error_measures:
                    # info for error type 1 generated in this branch
                    out_dict_air_error_measures['airway_id'].append(ibrh + 1)
                    out_dict_air_error_measures['type_error'].append(1)
                    out_dict_air_error_measures['loc_center_x'].append(loc_center_blank[0])
                    out_dict_air_error_measures['loc_center_y'].append(loc_center_blank[1])
                    out_dict_air_error_measures['loc_center_z'].append(loc_center_blank[2])
                    out_dict_air_error_measures['diam_blank'].append(diam_base_blank)
                    out_dict_air_error_measures['length_blank'].append(length_axis_blank)
            # endfor

        # --------------------

        # *********************************************************************
        # Type 2 Errors : Blanking partially random (most of) terminal branches
        # *********************************************************************

        if args.is_gener_error_type2:
            print('\nGenerate errors of type2: blanking partially random (most of) terminal branches...')

            # get terminal branches, as those that have no children branches
            index_terminal_branches = [ibrh for ibrh, child_ids_brhs in enumerate(in_children_id_branches)
                                       if child_ids_brhs == '']
            num_terminal_branches = len(index_terminal_branches)

            num_terminal_branches_error = int(args.prop_branches_error_type2 * num_terminal_branches)
            print("Num branches with errors type2: %s..." % (num_terminal_branches_error))

            index_branches_generate_error = np.random.choice(index_terminal_branches, num_terminal_branches_error,
                                                             replace=False)
            index_branches_generate_error = np.sort(index_branches_generate_error)

            # --------------------

            for ibrh in index_branches_generate_error:
                begin_point_branch = (in_begpoint_x_branches[ibrh],
                                      in_begpoint_y_branches[ibrh],
                                      in_begpoint_z_branches[ibrh])
                end_point_branch = (in_endpoint_x_branches[ibrh],
                                    in_endpoint_y_branches[ibrh],
                                    in_endpoint_z_branches[ibrh])
                inner_diam_branch = in_inner_diam_branches[ibrh]

                vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)
                length_branch = get_norm_vector(vector_axis_branch)

                # position center blank: random in the first half of the branch
                rel_pos_begin_blank = np.random.random() * 0.5
                rel_pos_center_blank = (rel_pos_begin_blank + 1.0) / 2.0
                loc_center_blank = get_point_inside_segment(begin_point_branch, end_point_branch, rel_pos_center_blank)

                # diameter base blank: the branch diameter (inflated several times)
                diam_base_blank = INFLATE_DIAM_ERROR_T2 * inner_diam_branch
                if diam_base_blank > MAX_DIAM_ERROR_T2:
                    print("Warning: terminal branch \'%s\' with too large diam: \'%s\'... Clipping it to: \'%s\'..."
                          % (ibrh, diam_base_blank, MAX_DIAM_ERROR_T2))
                diam_base_blank = min(diam_base_blank, MAX_DIAM_ERROR_T2)

                # length blank: distance between start blank and end of the branch
                length_axis_blank = (1.0 - rel_pos_begin_blank) * length_branch

                inout_airway_mask = generate_error_blank_branch_cylinder(inout_airway_mask,
                                                                         loc_center_blank,
                                                                         vector_axis_branch,
                                                                         diam_base_blank,
                                                                         length_axis_blank)

                # ----------
                if args.is_output_error_measures:
                    # info for error type 2 generated in this branch
                    out_dict_air_error_measures['airway_id'].append(ibrh + 1)
                    out_dict_air_error_measures['type_error'].append(2)
                    out_dict_air_error_measures['loc_center_x'].append(loc_center_blank[0])
                    out_dict_air_error_measures['loc_center_y'].append(loc_center_blank[1])
                    out_dict_air_error_measures['loc_center_z'].append(loc_center_blank[2])
                    out_dict_air_error_measures['diam_blank'].append(diam_base_blank)
                    out_dict_air_error_measures['length_blank'].append(length_axis_blank)
            # endfor

        # --------------------

        if args.is_test_error_shapes:
            inout_airway_mask = np.ones_like(inout_airway_mask) - inout_airway_mask

        # --------------------

        out_airway_error_mask_file = in_casename + '_airways-error.nii.gz'
        if args.is_test_error_shapes:
            out_airway_error_mask_file = in_casename + '_test-error-shapes.nii.gz'

        out_airway_error_mask_file = join_path_names(args.output_dir, out_airway_error_mask_file)
        print("Output: \'%s\'..." % (basename(out_airway_error_mask_file)))

        NiftiFileReader.write_image(out_airway_error_mask_file, inout_airway_mask, metadata=in_metadata_file)

        if args.is_output_error_measures:
            out_air_error_measures_file = in_casename + '_air-error-measures.csv'
            out_air_error_measures_file = join_path_names(args.output_dir, out_air_error_measures_file)
            print("And: \'%s\'..." % (basename(out_air_error_measures_file)))

            CsvFileReader.write_data(out_air_error_measures_file, out_dict_air_error_measures,
                                     format_out_data=['%0.3d', '%0.1d', '%0.1f', '%0.1f', '%0.1f', '%0.3f', '%0.3f'])
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbasedir', type=str, default='.')
    parser.add_argument('--is_gener_error_type1', type=bool, default=True)
    parser.add_argument('--prop_branches_error_type1', type=float, default=0.4)
    parser.add_argument('--is_gener_error_type2', type=bool, default=True)
    parser.add_argument('--prop_branches_error_type2', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=2017)
    parser.add_argument('--output_dir', type=str, default='./AirwaysErrors/')
    parser.add_argument('--is_test_error_shapes', type=bool, default=False)
    parser.add_argument('--is_output_error_measures', type=bool, default=True)
    args = parser.parse_args()

    args.output_dir = join_path_names(args.inbasedir, args.output_dir)

    main(args)
