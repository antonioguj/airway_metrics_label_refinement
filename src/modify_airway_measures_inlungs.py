
from collections import OrderedDict
import argparse

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

        print(is_branches_inside_left_lung & is_branches_inside_right_lung)
        print(np.invert(is_branches_inside_left_lung) & np.invert(is_branches_inside_right_lung))

        check = np.invert(is_branches_inside_left_lung) & np.invert(is_branches_inside_right_lung)

        print(np.count_nonzero(check[0]))
        if np.count_nonzero(check[1:]) != 0:
            print('HELLO, SHIT')
            exit(0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbasedir', type=str, default='.')
    args = parser.parse_args()

    main(args)