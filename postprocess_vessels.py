
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader


def main(args):

    # SETTINGS
    list_input_masks_files = list_files_dir(args.input_masks_dir)

    def get_casename_filename(in_filename: str):
        suffix_name = ''    # IF INPUT VESSEL MASK FILES HAVE A SUFFIX, PUT HERE
        return basename(in_filename).replace(suffix_name + '.nii.gz', '')
    # --------

    if args.is_calc_connected_mask:
        makedir(args.output_connected_masks_dir)

    if args.is_calc_cenlines:
        makedir(args.output_cenlines_dir)

    # **********************

    for i, in_mask_file in enumerate(list_input_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))
        in_casename = get_casename_filename(in_mask_file)

        in_metadata_file = NiftiFileReader.get_image_metadata_info(in_mask_file)

        in_binmask = NiftiFileReader.get_image(in_mask_file)

        # ---------------

        if args.is_calc_connected_mask:
            print("Compute the largest Connected Component from the Binary Masks, with connectivity \'%s\'..." %
                  (args.in_connectivity_dim))
            out_binmask = compute_largest_connected_tree(in_binmask, args.in_connectivity_dim)
        else:
            out_binmask = in_binmask

        # ---------------

        if args.is_calc_cenlines:
            print("Compute the Centrelines from the Binary Masks by thinning operation...")
            out_cenlines_mask = compute_centrelines_mask(out_binmask)
        else:
            out_cenlines_mask = None

        # ---------------

        if args.is_calc_connected_mask:
            out_con_binmask_file = in_casename + '_connected.nii.gz'
            out_con_binmask_file = join_path_names(args.output_connected_masks_dir, out_con_binmask_file)
            print("Output: \'%s\'..." % (basename(out_con_binmask_file)))

            NiftiFileReader.write_image(out_con_binmask_file, out_binmask, metadata=in_metadata_file)

        if args.is_calc_cenlines:
            out_cenlines_file = in_casename + '_cenlines.nii.gz'
            out_cenlines_file = join_path_names(args.output_cenlines_dir, out_cenlines_file)
            print("Output: \'%s\'..." % (basename(out_cenlines_file)))

            NiftiFileReader.write_image(out_cenlines_file, out_cenlines_mask, metadata=in_metadata_file)
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_basedir', type=str, default='.')
    parser.add_argument('--input_masks_dir', type=str, default='./BinaryMasks/')
    parser.add_argument('--is_calc_connected_mask', type=bool, default=False)
    parser.add_argument('--in_connectivity_dim', type=int, default=3)
    parser.add_argument('--output_connected_masks_dir', type=str, default='./BinMasks_Connected/')
    parser.add_argument('--is_calc_cenlines', type=bool, default=True)
    parser.add_argument('--output_cenlines_dir', type=str, default='./Centrelines/')
    args = parser.parse_args()

    # ONLY NEED TO INDICATE BASE PATHS TO PREDICTED RESULTS
    args.input_basedir = '/home/antonio/Results/LabelRefinement_THIRONA/Predictions_VESSELS/'

    main(args)