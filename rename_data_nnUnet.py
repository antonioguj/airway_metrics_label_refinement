
from collections import OrderedDict
from common.functionutil import *
import argparse
import csv


def main(args):

    # SETTINGS
    input_images_dir = join_path_names(args.input_datadir, './ImagesWorkData/')
    input_labels_dir = join_path_names(args.input_datadir, './LabelsWorkData/')
    output_images_dir = join_path_names(args.input_datadir, './ImagesWorkData_Renamed-nnUnet/')
    output_labels_dir = join_path_names(args.input_datadir, './LabelsWorkData_Renamed-nnUnet/')
    out_reference_keys_file = join_path_names(args.input_datadir, 'referenceKeys_nnUnetimages.npy')

    output_templ_image_filenames = 'Air_1%0.3i_0000.nii.gz'
    output_templ_label_filenames = 'Air_1%0.3i.nii.gz'
    # --------

    makedir(output_images_dir)
    makedir(output_labels_dir)

    list_input_images_files = sorted(list_files_dir(input_images_dir))
    list_input_labels_files = sorted(list_files_dir(input_labels_dir))

    if len(list_input_images_files) != len(list_input_labels_files):
        message = 'Not equal num files found in input dirs for \'ImagesWorkData\' and \'LabelsWorkData\''
        handle_error_message(message)

    # ------------------

    outdict_reference_keys = OrderedDict()
    count_image_this = 0

    for in_image_file, in_label_file in zip(list_input_images_files, list_input_labels_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        out_image_file = output_templ_image_filenames % (count_image_this)
        out_image_file = join_path_names(output_images_dir, out_image_file)
        out_label_file = output_templ_label_filenames % (count_image_this)
        out_label_file = join_path_names(output_labels_dir, out_label_file)

        print("%s -> %s..." % (basename(in_image_file), basename(out_image_file)))
        print("%s -> %s..." % (basename(in_label_file), basename(out_label_file)))
        #copyfile(in_image_file, out_image_file)
        #copyfile(in_label_file, out_label_file)
        makelink(join_path_names('../', in_image_file), out_image_file)
        makelink(join_path_names('../', in_label_file), out_label_file)

        outdict_reference_keys[basename_filenoext(out_image_file)] = basename(in_image_file)
        count_image_this += 1
    # endfor

    # Save reference keys
    np.save(out_reference_keys_file, outdict_reference_keys)

    out_reference_keys_file = out_reference_keys_file.replace('.npy', '.csv')
    with open(out_reference_keys_file, 'w') as fout:
        writer = csv.writer(fout)
        for key, value in outdict_reference_keys.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_datadir', type=str, default='.')
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
