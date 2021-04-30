
from collections import OrderedDict
import argparse

from functionsutil import *
from metrics import get_metric

LIST_CALC_METRICS_DEFAULT = ['DiceCoefficient',
                             'AirwayCompleteness',
                             'AirwayVolumeLeakage',
                             'AirwayCentrelineLeakage',
                             'AirwayTreeLength',
                             'AirwayCentrelineDistanceFalsePositiveError',
                             'AirwayCentrelineDistanceFalseNegativeError',
                             ]


def main(args):

    # SETTINGS
    # input_reference_images_dir = join_path_names(args.refer_datadir, './Images')
    input_reference_masks_dir = join_path_names(args.refer_datadir, './Airways')
    input_reference_cenlines_dir = join_path_names(args.refer_datadir, './Centrelines')
    input_coarse_airways_dir = join_path_names(args.refer_datadir, './CoarseAirways')

    def get_casename_filename(in_predicted_mask_file: str):
        return basename(in_predicted_mask_file).replace('_binmask.nii.gz', '')
    # --------

    list_input_predicted_masks_files = list_files_dir(args.input_masks_dir)
    list_input_predicted_cenlines_files = list_files_dir(args.input_cenlines_dir)
    # list_input_reference_images_files = list_files_dir(input_reference_images_dir)
    # list_input_reference_masks_files = list_files_dir(input_reference_masks_dir)
    # list_input_reference_cenlines_files = list_files_dir(input_reference_cenlines_dir)
    # list_input_coarse_airways_files = list_files_dir(input_coarse_airways_dir)

    if len(list_input_predicted_masks_files) != len(list_input_predicted_cenlines_files):
        message = 'Input dirs for predicted masks and centrelines have different number of files...'
        handle_error_message(message)

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        list_metrics[new_metric._name_fun_out] = new_metric
    # endfor

    # **********************

    outdict_calc_metrics = OrderedDict()

    for i, (in_predicted_mask_file, in_predicted_cenline_file) in \
            enumerate(zip(list_input_predicted_masks_files, list_input_predicted_cenlines_files)):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))
        print("And: \'%s\'..." % (basename(in_predicted_cenline_file)))
        in_casename = get_casename_filename(in_predicted_mask_file)

        in_reference_mask_file = in_casename + '_manual-airways.nii.gz'
        in_reference_mask_file = join_path_names(input_reference_masks_dir, in_reference_mask_file)
        print("Reference mask file: \'%s\'..." % (basename(in_reference_mask_file)))

        in_reference_cenline_file = in_casename + '_manual-airways_cenlines.nii.gz'
        in_reference_cenline_file = join_path_names(input_reference_cenlines_dir, in_reference_cenline_file)
        print("Reference centrelines file: \'%s\'..." % (basename(in_reference_cenline_file)))

        in_predicted_mask = ImageFileReader.get_image(in_predicted_mask_file)
        in_predicted_cenline = ImageFileReader.get_image(in_predicted_cenline_file)
        in_reference_mask = ImageFileReader.get_image(in_reference_mask_file)
        in_reference_cenline = ImageFileReader.get_image(in_reference_cenline_file)

        # ---------------

        if args.is_remove_trachea_calc_metrics:
            print("Remove trachea and main bronchi masks in computed metrics...")

            in_coarse_airways_file = in_casename + '-airways.nii.gz'
            in_coarse_airways_file = join_path_names(input_coarse_airways_dir, in_coarse_airways_file)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            print("Dilate coarse airways masks 4 levels to remove completely the trachea and main bronchi from "
                  "the predictions and the ground-truth...")
            in_coarse_airways = compute_dilated_mask(in_coarse_airways, num_iters=4)

            in_predicted_mask = compute_substracted_two_masks(in_predicted_mask, in_coarse_airways)
            in_predicted_cenline = compute_substracted_two_masks(in_predicted_cenline, in_coarse_airways)
            in_reference_mask = compute_substracted_two_masks(in_reference_mask, in_coarse_airways)
            in_reference_cenline = compute_substracted_two_masks(in_reference_cenline, in_coarse_airways)

        # ---------------

        print("\nCompute the Metrics:")
        outdict_calc_metrics[in_casename] = []

        for (imetric_name, imetric) in list_metrics.items():
            if imetric._is_use_voxelsize:
                in_mask_voxel_size = ImageFileReader.get_image_voxelsize(in_predicted_mask_file)
                imetric.set_voxel_size(in_mask_voxel_size)

            outval_metric = imetric.compute(in_reference_mask, in_predicted_mask,
                                            in_reference_cenline, in_predicted_cenline)

            print("\'%s\': %s..." % (imetric_name, outval_metric))
            outdict_calc_metrics[in_casename].append(outval_metric)
        # endfor
    # endfor

    # write out computed metrics in file
    fout = open(args.output_result_file, 'w')
    strheader = ', '.join(['/case/'] + ['/%s/' % (key) for key in list_metrics.keys()]) + '\n'
    fout.write(strheader)

    for (in_casename, outlist_calc_metrics) in outdict_calc_metrics.items():
        list_write_data = [in_casename] + ['%0.6f' % (elem) for elem in outlist_calc_metrics]
        strdata = ', '.join(list_write_data) + '\n'
        fout.write(strdata)
    # endfor
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_basedir', type=str, default='.')
    # parser.add_argument('--refer_datadir', type=str, default='./ReferenceData/')
    parser.add_argument('--input_masks_dir', type=str, default='./BinaryMasks/')
    parser.add_argument('--input_cenlines_dir', type=str, default='./Centrelines/')
    parser.add_argument('--list_type_metrics', type=str, nargs='*', default=LIST_CALC_METRICS_DEFAULT)
    parser.add_argument('--output_result_file', type=str, default='./result_metrics.csv')
    parser.add_argument('--is_remove_trachea_calc_metrics', type=bool, default=True)
    args = parser.parse_args()

    args.refer_datadir = '/mnt/mydrive/PythonCodes/Airway_segmentation/resources/THIRONA_Fullsize/'

    args.input_masks_dir = join_path_names(args.input_basedir, args.input_masks_dir)
    args.input_cenlines_dir = join_path_names(args.input_basedir, args.input_cenlines_dir)
    args.output_result_file = join_path_names(args.input_basedir, args.output_result_file)

    main(args)
