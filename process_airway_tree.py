
import argparse

from functionsutil import makedir, basename, join_path_names, list_files_dir, compute_thresholded_image, \
    compute_merged_two_masks, compute_largest_connected_tree, compute_centrelines_mask, ImageFileReader


def main():

    # SETTINGS
    input_refer_images_dir = join_path_names(args.datadir, './Images')
    input_coarse_airways_dir = join_path_names(args.datadir, './CoarseAirways')

    def template_output_filenames(in_posterior_file: str, in_threshold: float):
        return basename(in_posterior_file) + '_binmask_thres%s.nii.gz' % (in_threshold)

    def template_output_centrelines_filenames(in_binary_mask_file: str):
        return basename(in_binary_mask_file) + '_cenlines.nii.gz'

    def get_casename_filename(in_posterior_file: str):
        suffix_name = '#####'   # IF POSTERIOR FILES HAVE A SUFFIX, PUT HERE
        return basename(in_posterior_file).replace('.nii.gz', '').replace(suffix_name, '')
    # --------

    makedir(args.output_dir)

    if args.is_calc_centrelines:
        makedir(args.output_centrelines_dir)

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

        print("Compute Binary Masks thresholded to \'%s\'..." % (args.post_threshold_value))

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

        if args.is_calc_centrelines:
            print("Compute the Centrelines from the Binary Masks by thinning operation...")
            out_centrelines = compute_centrelines_mask(out_binary_mask)
        else:
            out_centrelines = None

        # ---------------

        output_binary_mask_file = join_path_names(args.output_dir,
                                                  template_output_filenames(in_posterior_file, args.value_threshold))
        print("Output: \'%s\', of dims \'%s\'..." % (basename(output_binary_mask_file), str(out_binary_mask.shape)))

        ImageFileReader.write_image(output_binary_mask_file, out_binary_mask, metadata=in_metadata_file)

        if args.is_calc_centrelines:
            output_centrelines_file = join_path_names(args.output_centrelines_dir,
                                                      template_output_centrelines_filenames(output_binary_mask_file))
            print("Output: \'%s\', of dims \'%s\'..." % (basename(output_centrelines_file), str(out_centrelines.shape)))

            ImageFileReader.write_image(output_centrelines_file, out_centrelines, metadata=in_metadata_file)
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--datadir', type=str, default='./BaseData/')
    parser.add_argument('--output_dir', type=str, default='./BinaryMasks/')
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--is_attach_coarse_airways', type=bool, default=True)
    parser.add_argument('--is_calc_connected_tree', type=bool, default=True)
    parser.add_argument('--in_connectivity_dim', type=int, default=3)
    parser.add_argument('--is_calc_centrelines', type=bool, default=True)
    parser.add_argument('--output_centrelines_dir', type=str, default='./Centrelines/')
    args = parser.parse_args()

    main()
