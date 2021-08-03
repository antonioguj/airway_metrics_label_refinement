
from collections import OrderedDict
import argparse
import sys

from common.functionutil import *
from common.filereader import CsvFileReader


def main(args):

    # SETTINGS
    input_airway_measures_dir = join_path_names(args.inbasedir, './AirwayMeasurements')
    input_crop_boundboxes_file = join_path_names(args.inbasedir, 'cropBoundingBoxes_images.npy')
    input_images_info_file = join_path_names(args.inbasedir, './images_info.csv')

    output_dir = join_path_names(args.inbasedir, './AirwayMeasurementsInsideLungs')

    def get_casename_filename(in_filename: str):
        return basename(in_filename).replace('_ResultsPerBranch.csv', '')
    # --------

    makedir(output_dir)

    list_input_airway_measures_files = list_files_dir(input_airway_measures_dir, '*.csv')

    input_crop_boundboxes = dict(np.load(input_crop_boundboxes_file, allow_pickle=True).item())

    in_images_info = CsvFileReader.get_data(input_images_info_file)

    in_images_voxelsize_info = OrderedDict()
    for i, i_casename in enumerate(in_images_info['casename']):
        voxel_size_x = in_images_info['voxel_size_x'][i]
        voxel_size_y = in_images_info['voxel_size_y'][i]
        voxel_size_z = in_images_info['voxel_size_z'][i]
        in_images_voxelsize_info[i_casename] = (voxel_size_x, voxel_size_y, voxel_size_z)
    # endfor

    # **********************

    for in_airway_measures_file in list_input_airway_measures_files:
        print("\nInput: \'%s\'..." % (basename(in_airway_measures_file)))
        in_casename = get_casename_filename(in_airway_measures_file)

        in_airway_measures_data = CsvFileReader.get_data(in_airway_measures_file)

        in_boundbox_left_lung = input_crop_boundboxes[in_casename][0]
        in_boundbox_right_lung = input_crop_boundboxes[in_casename][1]

        # in_midpoint_x_branches = np.array(in_airway_measures_data['midPoint_x'])
        # in_midpoint_y_branches = np.array(in_airway_measures_data['midPoint_y'])
        # in_midpoint_z_branches = np.array(in_airway_measures_data['midPoint_z'])
        in_begpoint_x_branches = np.array(in_airway_measures_data['begPoint_x'])
        in_endpoint_x_branches = np.array(in_airway_measures_data['endPoint_x'])
        in_begpoint_y_branches = np.array(in_airway_measures_data['begPoint_y'])
        in_endpoint_y_branches = np.array(in_airway_measures_data['endPoint_y'])
        in_begpoint_z_branches = np.array(in_airway_measures_data['begPoint_z'])
        in_endpoint_z_branches = np.array(in_airway_measures_data['endPoint_z'])

        # --------------------

        # get branches that are inside left lung
        z_beg_left_lung, z_end_left_lung = in_boundbox_left_lung[0]
        y_beg_left_lung, y_end_left_lung = in_boundbox_left_lung[1]
        x_beg_left_lung, x_end_left_lung = in_boundbox_left_lung[2]

        is_begpoint_inside_left_lung = (in_begpoint_x_branches > x_beg_left_lung) \
                                       & (in_begpoint_x_branches < x_end_left_lung) \
                                       & (in_begpoint_y_branches > y_beg_left_lung) \
                                       & (in_begpoint_y_branches < y_end_left_lung) \
                                       & (in_begpoint_z_branches > z_beg_left_lung) \
                                       & (in_begpoint_z_branches < z_end_left_lung)
        is_endpoint_inside_left_lung = (in_endpoint_x_branches > x_beg_left_lung) \
                                       & (in_endpoint_x_branches < x_end_left_lung) \
                                       & (in_endpoint_y_branches > y_beg_left_lung) \
                                       & (in_endpoint_y_branches < y_end_left_lung) \
                                       & (in_endpoint_z_branches > z_beg_left_lung) \
                                       & (in_endpoint_z_branches < z_end_left_lung)
        is_branches_inside_left_lung = is_begpoint_inside_left_lung & is_endpoint_inside_left_lung

        # get branches that are inside right lung
        z_beg_right_lung, z_end_right_lung = in_boundbox_right_lung[0]
        y_beg_right_lung, y_end_right_lung = in_boundbox_right_lung[1]
        x_beg_right_lung, x_end_right_lung = in_boundbox_right_lung[2]

        is_begpoint_inside_right_lung = (in_begpoint_x_branches > x_beg_right_lung) \
                                        & (in_begpoint_x_branches < x_end_right_lung) \
                                        & (in_begpoint_y_branches > y_beg_right_lung) \
                                        & (in_begpoint_y_branches < y_end_right_lung) \
                                        & (in_begpoint_z_branches > z_beg_right_lung) \
                                        & (in_begpoint_z_branches < z_end_right_lung)
        is_endpoint_inside_right_lung = (in_endpoint_x_branches > x_beg_right_lung) \
                                        & (in_endpoint_x_branches < x_end_right_lung) \
                                        & (in_endpoint_y_branches > y_beg_right_lung) \
                                        & (in_endpoint_y_branches < y_end_right_lung) \
                                        & (in_endpoint_z_branches > z_beg_right_lung) \
                                        & (in_endpoint_z_branches < z_end_right_lung)
        is_branches_inside_right_lung = is_begpoint_inside_right_lung & is_endpoint_inside_right_lung

        # --------------------

        # check that all branch measures in one of the bound-boxes
        is_branch_not_inside_either_boundbox = np.invert(is_branches_inside_left_lung) \
                                               & np.invert(is_branches_inside_right_lung)
        num_branches_excluded = np.count_nonzero(is_branch_not_inside_either_boundbox)

        if num_branches_excluded > 0:
            indexes_branches_excluded = np.argwhere(is_branch_not_inside_either_boundbox)[0]

            generation_branches_excluded = [in_airway_measures_data['generation'][i] for i in indexes_branches_excluded]

            # branches of max generation 1 (trachea and main bronchi) can be excluded,
            # as these are not used when generating errors
            if np.max(generation_branches_excluded) > 1:
                print('ERROR: found branch that is not included in either bound-box for left / right lung, '
                      'and is different than trachea or main bronchi... EXIT')
                sys.exit(0)

        # --------------------

        # split the measures in two groups, for the branches included in either left / right lung
        # IMPORTANT: correct coordinates so that measures can be used directly in cropped images

        indexes_branches_inside_left_lung = np.where(is_branches_inside_left_lung)[0]
        indexes_branches_inside_right_lung = np.where(is_branches_inside_right_lung)[0]

        in_airway_measures_data_left_lung = OrderedDict()
        in_airway_measures_data_right_lung = OrderedDict()

        for ifield, idata in in_airway_measures_data.items():

            if (ifield == 'midPoint_x') or (ifield == 'begPoint_x') or (ifield == 'endPoint_x'):
                idata_left_lung = [idata[ind] - x_beg_left_lung for ind in indexes_branches_inside_left_lung]
                idata_right_lung = [idata[ind] - x_beg_right_lung for ind in indexes_branches_inside_right_lung]

            elif (ifield == 'midPoint_y') or (ifield == 'begPoint_y') or (ifield == 'endPoint_y'):
                idata_left_lung = [idata[ind] - y_beg_left_lung for ind in indexes_branches_inside_left_lung]
                idata_right_lung = [idata[ind] - y_beg_right_lung for ind in indexes_branches_inside_right_lung]

            elif (ifield == 'midPoint_z') or (ifield == 'begPoint_z') or (ifield == 'endPoint_z'):
                idata_left_lung = [idata[ind] - z_beg_left_lung for ind in indexes_branches_inside_left_lung]
                idata_right_lung = [idata[ind] - z_beg_right_lung for ind in indexes_branches_inside_right_lung]

            else:
                idata_left_lung = [idata[ind] for ind in indexes_branches_inside_left_lung]
                idata_right_lung = [idata[ind] for ind in indexes_branches_inside_right_lung]

            in_airway_measures_data_left_lung[ifield] = idata_left_lung
            in_airway_measures_data_right_lung[ifield] = idata_right_lung

        # --------------------

        # output measures for left / right lungs in separate .csv files
        in_airway_measures_file_left = in_casename + '_LeftLung_ResultsPerBranch.csv'
        in_airway_measures_file_left = join_path_names(output_dir, in_airway_measures_file_left)

        in_airway_measures_file_right = in_casename + '_RightLung_ResultsPerBranch.csv'
        in_airway_measures_file_right = join_path_names(output_dir, in_airway_measures_file_right)

        print("Output: \'%s\'..." % (basename(in_airway_measures_file_left)))
        print("And: \'%s\'..." % (basename(in_airway_measures_file_right)))

        format_out_data = ['%s', '%0.3d', '%0.1f', '%0.1f', '%0.1f', '%0.3f', '%0.3f', '%0.3f', '%d', '%s', '%s',
                           '%0.1f', '%0.1f', '%0.1f', '%0.1f', '%0.1f', '%0.1f']

        CsvFileReader.write_data(in_airway_measures_file_left, in_airway_measures_data_left_lung, format_out_data)
        CsvFileReader.write_data(in_airway_measures_file_right, in_airway_measures_data_right_lung, format_out_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbasedir', type=str, default='.')
    args = parser.parse_args()

    main(args)