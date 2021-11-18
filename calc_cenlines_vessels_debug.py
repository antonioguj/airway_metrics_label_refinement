
import argparse

from common.functionutil import *
from common.filereader import NiftiFileReader


def main(args):

    list_input_masks_files = list_files_dir(args.input_dir)

    makedir(args.output_dir)

    for i, in_mask_file in enumerate(list_input_masks_files):
        print("\nInput: \'%s\'..." % (basename(in_mask_file)))

        in_mask = NiftiFileReader.get_image(in_mask_file)

        in_metadata_file = NiftiFileReader.get_image_metadata_info(in_mask_file)

        out_cenlines = compute_centrelines_mask(in_mask)

        out_cenlines_file = join_path_names(args.output_dir, basename_filenoext(in_mask_file) + '_cenlines.nii.gz')
        print("Output: \'%s\'..." % (basename(out_cenlines_file)))

        NiftiFileReader.write_image(out_cenlines_file, out_cenlines, metadata=in_metadata_file)
    # endfor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    if args.input_dir.endswith('/'):
        args.output_dir = args.input_dir[:-1] + '_Cenlines/'
    else:
        args.output_dir = args.input_dir + '_Cenlines/'

    main(args)