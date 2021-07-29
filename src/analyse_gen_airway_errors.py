
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import CsvFileReader
from common.errorgenerator import get_norm_vector


def main(args):

    # SETTINGS
    input_airway_measures_dir = join_path_names(args.inbasedir, './AirwayMeasurements')
    input_images_info_file = join_path_names(args.inbasedir, './images_info.csv')

    def get_casename_filename(in_filename: str):
        return basename(in_filename).replace('_air-error-measures.csv', '')
    # --------

    list_input_airway_error_files = list_files_dir(args.input_dir, '*.csv')

    in_images_info = CsvFileReader.get_data(input_images_info_file)

    in_images_voxelsize_info = OrderedDict()
    for i, i_casename in enumerate(in_images_info['casename']):
        voxel_size_x = in_images_info['voxel_size_x'][i]
        voxel_size_y = in_images_info['voxel_size_y'][i]
        voxel_size_z = in_images_info['voxel_size_z'][i]
        in_images_voxelsize_info[i_casename] = (voxel_size_x, voxel_size_y, voxel_size_z)
    # endfor

    out_dict_ratio_extent_air_errors = OrderedDict()
    out_dict_ratio_extent_air_errors['case'] = []
    out_dict_ratio_extent_air_errors['ratio_num_branch_error_type1'] = []
    out_dict_ratio_extent_air_errors['ratio_tree_length_error_type1'] = []
    out_dict_ratio_extent_air_errors['ratio_num_branch_error_type2'] = []
    out_dict_ratio_extent_air_errors['ratio_tree_length_error_type2'] = []

    # **********************

    for in_airway_error_file in list_input_airway_error_files:
        print("\nInput: \'%s\'..." % (basename(in_airway_error_file)))
        in_casename = get_casename_filename(in_airway_error_file)

        in_airway_measures_file = in_casename + '_ResultsPerBranch.csv'
        in_airway_measures_file = join_path_names(input_airway_measures_dir, in_airway_measures_file)
        print("With measures from: \'%s\'..." % (basename(in_airway_measures_file)))

        in_airway_error_measures = CsvFileReader.get_data(in_airway_error_file)

        in_airway_measures_data = CsvFileReader.get_data(in_airway_measures_file)

        in_airway_ids_air_errors = in_airway_error_measures['airway_id']
        in_type_error_air_errors = in_airway_error_measures['type_error']
        in_length_blank_air_errors = in_airway_error_measures['length_blank']

        in_airlength_branches = in_airway_measures_data['airway_length']
        in_generation_branches = in_airway_measures_data['generation']

        in_voxelsize_image = in_images_voxelsize_info[in_casename]
        in_voxelnorm_image = get_norm_vector(in_voxelsize_image)

        # normalize the airway length measures
        in_airlength_branches = [elem / in_voxelnorm_image for elem in in_airlength_branches]

        # ---------------

        indexes_branches_type1_air_errors = [ind for ind, type in enumerate(in_type_error_air_errors) if type == 1]
        indexes_branches_type2_air_errors = [ind for ind, type in enumerate(in_type_error_air_errors) if type == 2]

        num_branches_total = len(in_airlength_branches)
        sum_length_branches_total = np.sum(in_airlength_branches)

        num_branches_error_type1 = len(indexes_branches_type1_air_errors)

        sum_length_blanks_error_type1 = 0.0
        for index_brh in indexes_branches_type1_air_errors:
            length_blank_error = in_length_blank_air_errors[index_brh]
            sum_length_blanks_error_type1 += length_blank_error
        # endfor

        num_branches_error_type2 = len(indexes_branches_type2_air_errors)

        sum_length_blanks_error_type2 = 0.0
        for index_brh in indexes_branches_type2_air_errors:
            length_blank_error = in_length_blank_air_errors[index_brh]
            sum_length_blanks_error_type2 += length_blank_error
        # endfor

        # check that all measures of blank lengths of errors are lower than branch lengths
        in_airlength_where_air_errors = [in_airlength_branches[ind - 1] for ind in in_airway_ids_air_errors]
        check_diff_lengths_air_errors = np.array(in_airlength_where_air_errors) - np.array(in_length_blank_air_errors)

        if np.any(check_diff_lengths_air_errors < -1.0e-03):
            message = 'found branches where the generated error with blank length that is larger than branch length'
            handle_error_message(message)

        # ---------------

        # save info about generated errors
        ratio_num_branch_error_type1 = num_branches_error_type1 / float(num_branches_total)
        ratio_tree_length_error_type1 = sum_length_blanks_error_type1 / float(sum_length_branches_total)
        ratio_num_branch_error_type2 = num_branches_error_type2 / float(num_branches_total)
        ratio_tree_length_error_type2 = sum_length_blanks_error_type2 / float(sum_length_branches_total)

        out_dict_ratio_extent_air_errors['case'].append(in_casename)
        out_dict_ratio_extent_air_errors['ratio_num_branch_error_type1'].append(ratio_num_branch_error_type1)
        out_dict_ratio_extent_air_errors['ratio_tree_length_error_type1'].append(ratio_tree_length_error_type1)
        out_dict_ratio_extent_air_errors['ratio_num_branch_error_type2'].append(ratio_num_branch_error_type2)
        out_dict_ratio_extent_air_errors['ratio_tree_length_error_type2'].append(ratio_tree_length_error_type2)
    # endfor

    print("\nOutput: \'%s\'..." % (args.output_file))

    CsvFileReader.write_data(args.output_file, out_dict_ratio_extent_air_errors,
                             format_out_data=['%s', '%0.3f', '%0.3f', '%0.3f', '%0.3f'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default='./AirwaysErrors/')
    parser.add_argument('output_file', type=str, default='./extent_airway_error.csv/')
    parser.add_argument('--inbasedir', type=str, default='.')
    args = parser.parse_args()

    main(args)
