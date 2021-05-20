
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader, CsvFileReader
from common.genererrors import get_vector_two_points, get_norm_vector, get_point_in_segment, \
    generate_error_blank_branch_cylinder

MIN_BRANCH_GENER_ERROR_T1 = 4
INFLATE_DIAM_ERROR_T1 = 4.0
MAX_DIAM_ERROR_T1 = 30.0
MIN_LENGTH_ERROR_T1 = 1.0
INFLATE_LENGTH_ERROR_T1 = 1.0

INFLATE_DIAM_ERROR_T2 = 6.0
MAX_DIAM_ERROR_T2 = 30.0
INFLATE_LENGTH_ERROR_T2 = 2.0


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

    if args.is_gener_error_type1:
        print("Generate errors of Type 1: blanking small regions in random branches...")
        print("Cylindres: 1) center: random position along the branch...")
        print("           2) diameter: the branch diameter (inflated %sx times)..." % (INFLATE_DIAM_ERROR_T1))
        print("           3) length: random between min. %smm and the branch length (inflated %sx times)..."
              % (MIN_LENGTH_ERROR_T1, INFLATE_LENGTH_ERROR_T1))

    if args.is_gener_error_type2:
        print("Generate errors of Type 2: blanking partially random (most of) terminal branches...")
        print("Cylindres: 1) center: random position in the first half of the branch...")
        print("           2) diameter: the branch diameter (inflated %sx times)..." % (INFLATE_DIAM_ERROR_T2))
        print("           3) length: distance between start blank and end of the branch (inflated %sx times)..."
              % (INFLATE_LENGTH_ERROR_T2))

    # **********************

    for in_airway_mask_file in list_input_airway_masks:
        print("\nInput: \'%s\'..." % (basename(in_airway_mask_file)))
        in_casename = get_casename_filename(in_airway_mask_file)

        in_airway_measures_file = in_casename + '_ResultsPerBranch.csv'
        in_airway_measures_file = join_path_names(input_airway_measures_dir, in_airway_measures_file)
        print("With measures from: \'%s\'..." % (basename(in_airway_measures_file)))

        in_metadata_file = NiftiFileReader.get_image_metadata_info(in_airway_mask_file)

        inout_airway_mask = NiftiFileReader.get_image(in_airway_mask_file)

        if args.is_test_blank_shapes:
            inout_airway_mask = np.ones_like(inout_airway_mask)

        in_airway_measures = CsvFileReader.get_data(in_airway_measures_file)

        # ---------------

        in_airway_id_branches = in_airway_measures['airway_ID']
        in_midpoint_x_branches = in_airway_measures['midPoint_x']
        in_midpoint_y_branches = in_airway_measures['midPoint_y']
        in_midpoint_z_branches = in_airway_measures['midPoint_z']
        in_inner_diam_branches = in_airway_measures['d_inner_global']
        in_air_length_branches = in_airway_measures['airway_length']
        in_generation_branches = in_airway_measures['generation']
        in_parent_id_branches = in_airway_measures['parent_ID']
        in_children_id_branches = in_airway_measures['childrenID']
        in_begpoint_x_branches = in_airway_measures['begPoint_x']
        in_endpoint_x_branches = in_airway_measures['endPoint_x']
        in_begpoint_y_branches = in_airway_measures['begPoint_y']
        in_endpoint_y_branches = in_airway_measures['endPoint_y']
        in_begpoint_z_branches = in_airway_measures['begPoint_z']
        in_endpoint_z_branches = in_airway_measures['endPoint_z']

        in_voxelsize_image = in_images_voxelsize_info[in_casename]
        in_voxelnorm_image = get_norm_vector(in_voxelsize_image)

        num_branches = len(in_airway_id_branches)
        print("Num total branches: %s..." % (num_branches))

        # ---------------

        if args.random_seed:
            np.random.seed(args.random_seed)

        # ********************
        # Type 1 Errors : Blanking small regions in random branches
        # ********************

        if args.is_gener_error_type1:
            num_branches_error = int(args.prop_branches_error_type1 * num_branches)
            print("Num branches with errors type1: %s..." % (num_branches_error))

            # use sample probability as twice the generation number, so that terminal branches have more likely errors
            # exclude larger main branches
            geners_sample_probs = [max(elem, MIN_BRANCH_GENER_ERROR_T1-1) - (MIN_BRANCH_GENER_ERROR_T1-1)
                                   for elem in in_generation_branches]
            geners_sample_probs = np.array(geners_sample_probs) / np.sum(geners_sample_probs)

            indexes_branches_gener_error = np.random.choice(range(num_branches), num_branches_error, replace=False,
                                                            p=geners_sample_probs)
            indexes_branches_gener_error = np.sort(indexes_branches_gener_error)
        else:
            indexes_branches_gener_error = []

        for index_brh in indexes_branches_gener_error:

            begin_point_branch = (in_begpoint_x_branches[index_brh],
                                  in_begpoint_y_branches[index_brh],
                                  in_begpoint_z_branches[index_brh])
            end_point_branch = (in_endpoint_x_branches[index_brh],
                                in_endpoint_y_branches[index_brh],
                                in_endpoint_z_branches[index_brh])

            inner_diam_branch = in_inner_diam_branches[index_brh] / in_voxelnorm_image
            length_branch = in_air_length_branches[index_brh] / in_voxelnorm_image

            vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)

            # center: random position along the branch
            reldist_center_blank = np.random.random()
            loc_center_blank_branch = get_point_in_segment(begin_point_branch, end_point_branch, reldist_center_blank)

            # diameter: the branch diameter (inflated several times)
            diam_base_blank = INFLATE_DIAM_ERROR_T1 * inner_diam_branch
            if diam_base_blank > MAX_DIAM_ERROR_T1:
                print("Warning: branch \'%s\' with too large diam: \'%s\'... Clipping it to: \'%s\'..."
                      % (index_brh, diam_base_blank, MAX_DIAM_ERROR_T1))
            diam_base_blank = min(diam_base_blank, MAX_DIAM_ERROR_T1)

            # length: random between min. 1mm and the branch length (inflated several times)
            length_axis_blank = np.random.random() * (INFLATE_LENGTH_ERROR_T1 * length_branch)
            min_length_blank = MIN_LENGTH_ERROR_T1 / in_voxelnorm_image
            length_axis_blank = max(length_axis_blank, min_length_blank)

            inout_airway_mask = generate_error_blank_branch_cylinder(inout_airway_mask,
                                                                     loc_center_blank_branch,
                                                                     vector_axis_branch,
                                                                     diam_base_blank,
                                                                     length_axis_blank)
        # endfor

        # ********************
        # Type 2 Errors : Blanking partially random (most of) terminal branches
        # ********************

        if args.is_gener_error_type2:
            # get terminal branches, as those that have no children branches
            in_airway_id_termin_branches = [ind for ind, child_id in enumerate(in_children_id_branches)
                                            if child_id == '']
            num_termin_branches = len(in_airway_id_termin_branches)

            num_termin_branches_error = int(args.prop_branches_error_type2 * num_termin_branches)
            print("Num branches with errors type2: %s..." % (num_termin_branches_error))

            indexes_branches_gener_error = \
                np.random.choice(in_airway_id_termin_branches, num_termin_branches_error, replace=False)
            indexes_branches_gener_error = np.sort(indexes_branches_gener_error)
        else:
            indexes_branches_gener_error = []

        for index_brh in indexes_branches_gener_error:

            begin_point_branch = (in_begpoint_x_branches[index_brh],
                                  in_begpoint_y_branches[index_brh],
                                  in_begpoint_z_branches[index_brh])
            end_point_branch = (in_endpoint_x_branches[index_brh],
                                in_endpoint_y_branches[index_brh],
                                in_endpoint_z_branches[index_brh])

            inner_diam_branch = in_inner_diam_branches[index_brh] / in_voxelnorm_image
            # length_branch = in_air_length_branches[index_brh] / in_voxelnorm_image

            vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)

            # center: random position in the first half of the branch
            reldist_begin_blank = np.random.random() * 0.5
            reldist_center_blank = (reldist_begin_blank + 1.0) / 2.0
            loc_center_blank_branch = get_point_in_segment(begin_point_branch, end_point_branch, reldist_center_blank)

            # diameter: the branch diameter (inflated several times) (clip for terminal branches with wrong large diam)
            diam_base_blank = INFLATE_DIAM_ERROR_T2 * inner_diam_branch
            if diam_base_blank > MAX_DIAM_ERROR_T2:
                print("Warning: terminal branch \'%s\' with too large diam: \'%s\'... Clipping it to: \'%s\'..."
                      % (index_brh, diam_base_blank, MAX_DIAM_ERROR_T2))
            diam_base_blank = min(diam_base_blank, MAX_DIAM_ERROR_T2)

            # length: distance between start blank and end of the branch (inflated several times)
            loc_begin_blank_branch = get_point_in_segment(begin_point_branch, end_point_branch, reldist_begin_blank)
            vector_begin_end_blank_branch = get_vector_two_points(loc_begin_blank_branch, end_point_branch)
            length_axis_blank = get_norm_vector(vector_begin_end_blank_branch)
            length_axis_blank = INFLATE_LENGTH_ERROR_T2 * length_axis_blank

            inout_airway_mask = generate_error_blank_branch_cylinder(inout_airway_mask,
                                                                     loc_center_blank_branch,
                                                                     vector_axis_branch,
                                                                     diam_base_blank,
                                                                     length_axis_blank)
        # endfor

        # ---------------

        if args.is_test_blank_shapes:
            inout_airway_mask = np.ones_like(inout_airway_mask) - inout_airway_mask

        out_airway_error_mask_file = in_casename + '_airways-errors.nii.gz'
        if args.is_test_blank_shapes:
            out_airway_error_mask_file = in_casename + '_blank-shapes.nii.gz'

        out_airway_error_mask_file = join_path_names(args.output_dir, out_airway_error_mask_file)
        print("Output: \'%s\'..." % (basename(out_airway_error_mask_file)))

        NiftiFileReader.write_image(out_airway_error_mask_file, inout_airway_mask, metadata=in_metadata_file)
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
    parser.add_argument('--is_test_blank_shapes', type=bool, default=False)
    args = parser.parse_args()

    args.output_dir = join_path_names(args.inbasedir, args.output_dir)

    main(args)
