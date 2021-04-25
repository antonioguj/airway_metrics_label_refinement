
import argparse

from functionsutil import makedir, basename, join_path_names, list_files_dir, compute_thresholded_image, \
    compute_merged_two_masks, compute_largest_connected_tree, compute_centrelines_mask, ImageFileReader


def main():

    # SETTINGS
    input_refer_images_dir = join_path_names(args.refer_datadir, './Images')
    input_coarse_airways_dir = join_path_names(args.refer_datadir, './CoarseAirways')

    def get_casename_filename(in_filename: str):
        suffix_name = '_probmap'    # IF POSTERIOR FILES HAVE A SUFFIX, PUT HERE
        return basename(in_filename).replace(suffix_name + '.nii.gz', '')
    # --------

    makedir(args.output_dir)

    if args.is_calc_cenlines:
        makedir(args.output_cenlines_dir)

    list_input_posteriors_files = list_files_dir(args.input_dir)
    # list_input_refer_images_files = list_files_dir(input_refer_images_dir)
    # list_input_coarse_airways_files = list_files_dir(input_coarse_airways_dir)

    # **********************

    for i, in_posterior_file in enumerate(list_input_posteriors_files):
        print("\nInput: \'%s\'..." % (basename(in_posterior_file)))
        in_casename = get_casename_filename(in_posterior_file)

        in_refer_image_file = in_casename + '.nii.gz'
        in_refer_image_file = join_path_names(input_refer_images_dir, in_refer_image_file)

        in_metadata_file = ImageFileReader.get_image_metadata_info(in_refer_image_file)

        in_posterior = ImageFileReader.get_image(in_posterior_file)

        print("Compute Binary Masks thresholded to \'%s\'..." % (args.value_threshold))

        out_binary_mask = compute_thresholded_image(in_posterior, args.value_threshold)

        # ---------------

        if args.is_attach_coarse_airways:
            print("Attach Trachea and Main Bronchi mask to complete the computed Binary Masks...")

            in_coarse_airways_file = in_casename + '-airways.nii.gz'
            in_coarse_airways_file = join_path_names(input_coarse_airways_dir, in_coarse_airways_file)
            print("Coarse Airways mask file: \'%s\'..." % (basename(in_coarse_airways_file)))

            in_coarse_airways = ImageFileReader.get_image(in_coarse_airways_file)

            out_binary_mask = compute_merged_two_masks(out_binary_mask, in_coarse_airways)

        # ---------------

        if args.is_calc_connected_tree:
            print("Compute the largest Connected Component from the Binary Masks, with connectivity \'%s\'..." %
                  (args.in_connectivity_dim))
            out_binary_mask = compute_largest_connected_tree(out_binary_mask, args.in_connectivity_dim)

        # ---------------

        if args.is_calc_cenlines:
            print("Compute the Centrelines from the Binary Masks by thinning operation...")
            out_cenlines_mask = compute_centrelines_mask(out_binary_mask)
        else:
            out_cenlines_mask = None

        # ---------------

        out_binmask_file = in_casename + '_binmask.nii.gz'
        out_binmask_file = join_path_names(args.output_dir, out_binmask_file)
        print("Output: \'%s\', of dims \'%s\'..." % (basename(out_binmask_file), str(out_binary_mask.shape)))

        ImageFileReader.write_image(out_binmask_file, out_binary_mask, metadata=in_metadata_file)

        if args.is_calc_cenlines:
            out_cenlines_file = in_casename + '_binmask_cenlines.nii.gz'
            out_cenlines_file = join_path_names(args.output_cenlines_dir, out_cenlines_file)
            print("Output: \'%s\', of dims \'%s\'..." % (basename(out_cenlines_file), str(out_cenlines_mask.shape)))

            ImageFileReader.write_image(out_cenlines_file, out_cenlines_mask, metadata=in_metadata_file)
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refer_datadir', type=str, default='./ReferenceData/')
    parser.add_argument('--input_dir', type=str, default='./Posteriors/')
    parser.add_argument('--output_dir', type=str, default='./BinaryMasks/')
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--is_attach_coarse_airways', type=bool, default=True)
    parser.add_argument('--is_calc_connected_tree', type=bool, default=False)
    parser.add_argument('--in_connectivity_dim', type=int, default=3)
    parser.add_argument('--is_calc_cenlines', type=bool, default=True)
    parser.add_argument('--output_cenlines_dir', type=str, default='./Centrelines/')
    args = parser.parse_args()

    main()
