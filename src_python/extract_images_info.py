
from common.functionutil import *
from common.filereader import NiftiFileReader
import argparse


def main(args):

    def get_casename_filename(in_filename: str):
        return basename(in_filename).replace('.nii.gz', '')

    input_dir = join_path_names(args.inbasedir, './Images')
    output_file = join_path_names(args.inbasedir, './images_info.csv')

    list_input_files = list_files_dir(input_dir)

    with open(output_file, 'w') as fout:

        list_fields = ['casename', 'image_size_z', 'image_size_x', 'image_size_y',
                       'voxel_size_z', 'voxel_size_x', 'voxel_size_y']
        header = ', '.join(list_fields) + '\n'
        fout.write(header)

        for in_file in list_input_files:
            print("\nInput: \'%s\'..." % (basename(in_file)))
            in_casename = get_casename_filename(in_file)

            in_image_size = NiftiFileReader.get_image_size(in_file)
            in_voxel_size = NiftiFileReader.get_image_voxelsize(in_file)
            in_voxel_size = tuple(np.roll(in_voxel_size, 1))    # place 'voxel_size_z' in first place

            print("Image size: %s..." % (str(in_image_size)))
            print("Voxel size: %s..." % (str(in_voxel_size)))

            data_write = [in_casename] + ['%d' % (elem) for elem in in_image_size] + \
                         ['%.3f' % (elem) for elem in in_voxel_size]
            row_write = ', '.join(data_write) + '\n'
            fout.write(row_write)
        # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbasedir', type=str, default='.')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
