
from collections import OrderedDict
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader
from common.metrics import get_metric

LIST_CALC_METRICS_DEFAULT = ['DiceCoefficient',
                             'AirwayCompleteness',
                             'AirwayVolumeLeakage',
                             'AirwayCentrelineLeakage',
                             'AirwayTreeLength',
                             'AirwayCentrelineDistanceFalsePositiveError',
                             'AirwayCentrelineDistanceFalseNegativeError',
                             ]


def main(args):

    list_input_predicted_masks_files = list_files_dir(args.input_predicted_dir)
    list_input_reference_masks_files = list_files_dir(args.input_reference_dir)

    if len(list_input_predicted_masks_files) != len(list_input_reference_masks_files):
        message = 'Input dirs for predicted and reference masks have different number of files...'
        handle_error_message(message)

    list_metrics = OrderedDict()
    for itype_metric in args.list_type_metrics:
        new_metric = get_metric(itype_metric)
        list_metrics[new_metric._name_fun_out] = new_metric
    # endfor

    is_calc_cenlines_predicted_masks = ('AirwayCentrelineLeakage' in args.list_type_metrics) \
                                       or ('AirwayCentrelineDistanceFalsePositiveError' in args.list_type_metrics) \
                                       or ('AirwayCentrelineDistanceFalseNegativeError' in args.list_type_metrics)
    is_calc_cenlines_reference_masks = ('AirwayCompleteness' in args.list_type_metrics) \
                                       or ('AirwayTreeLength' in args.list_type_metrics) \
                                       or ('AirwayCentrelineDistanceFalsePositiveError' in args.list_type_metrics) \
                                       or ('AirwayCentrelineDistanceFalseNegativeError' in args.list_type_metrics)

    # **********************

    outdict_calc_metrics = OrderedDict()

    for i, (in_predicted_mask_file, in_reference_mask_file) in enumerate(zip(list_input_predicted_masks_files,
                                                                             list_input_reference_masks_files)):
        print("\nInput: \'%s\'..." % (basename(in_predicted_mask_file)))
        print("And: \'%s\'..." % (basename(in_reference_mask_file)))

        # in_casename = basename_filenoext(in_predicted_mask_file)
        in_casename = 'case-%0.2d' % (i + 1)

        in_predicted_mask = NiftiFileReader.get_image(in_predicted_mask_file)
        in_reference_mask = NiftiFileReader.get_image(in_reference_mask_file)

        # Compute Centerlines from the masks
        if is_calc_cenlines_predicted_masks:
            in_predicted_cenline = compute_centrelines_mask(in_predicted_mask)
        else:
            in_predicted_cenline = None

        if is_calc_cenlines_reference_masks:
            in_reference_cenline = compute_centrelines_mask(in_reference_mask)
        else:
            in_reference_cenline = None

        # Compute the dilation of the reference masks
        if args.is_dilate_refer_masks:
            in_reference_mask = compute_dilated_mask(in_reference_mask, args.times_dilate_refer_masks)

        # ---------------

        print("\nCompute the Metrics:")
        outdict_calc_metrics[in_casename] = []

        for (imetric_name, imetric) in list_metrics.items():
            if imetric._is_use_voxelsize:
                in_mask_voxel_size = NiftiFileReader.get_image_voxelsize(in_predicted_mask_file)
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
    parser.add_argument('input_predicted_dir', type=str, default='./Predicted/')
    parser.add_argument('input_reference_dir', type=str, default='./Reference/')
    parser.add_argument('--list_type_metrics', type=str, nargs='*', default=LIST_CALC_METRICS_DEFAULT)
    parser.add_argument('--output_result_file', type=str, default='./result_metrics.csv')
    parser.add_argument('--is_dilate_refer_masks', type=bool, default=False)
    parser.add_argument('--times_dilate_refer_masks', type=int, default=1)
    args = parser.parse_args()

    #args.input_predicted_dir = '/OPTIONAL__PUT_THE_FULL_PATH/'
    #args.input_reference_dir = '/OPTIONAL__PUT_THE_FULL_PATH/'

    main(args)
