
from common.functionutil import *
import argparse


def main(args):

    # SETTINGS
    input_images_dir = join_path_names(args.input_datadir, './ImagesWorkData/')
    input_labels_dir = join_path_names(args.input_datadir, './LabelsWorkData/')
    # input_labels_errors_dirs = join_path_names(args.input_datadir, './LabelsWorkData_WithErrors_v*/')
    input_names_train_images_file = join_path_names(args.input_datadir, 'traindatafiles_20imgs.txt')
    input_names_valid_images_file = join_path_names(args.input_datadir, 'validdatafiles_4imgs.txt')
    # pattern_rootname_infiles = 'labels_proc-[0-9]+_crop-[0-9]+'

    output_train_dir = join_path_names(args.input_datadir, './TrainData_20imgs/')
    output_valid_dir = join_path_names(args.input_datadir, './ValidData_4imgs/')

    output_templ_train_image_filenames = 'Air_1%0.3i_0000.nii.gz'
    output_templ_train_label_filenames = 'Air_1%0.3i.nii.gz'
    # output_templ_train_label_error_filenames = 'Air_1%0.3i_0001_%0.2i.nii.gz'
    output_templ_valid_image_filenames = 'Air_2%0.3i_0000.nii.gz'
    output_templ_valid_label_filenames = 'Air_2%0.3i.nii.gz'
    # output_templ_valid_label_error_filenames = 'Air_2%0.3i_0001_%0.2i.nii.gz'

    # num_labels_witherrrors_percase = 10
    # --------

    makedir(output_train_dir)
    makedir(output_valid_dir)

    list_input_images_files = sorted(list_files_dir(input_images_dir))
    list_input_labels_files = sorted(list_files_dir(input_labels_dir))
    # list_input_labels_errors_files = sorted(list_files_dir(input_labels_errors_dirs))

    if len(list_input_images_files) != len(list_input_labels_files):
        message = 'Not equal num files found in input dirs for \'ImagesWorkData\' and \'LabelsWorkData\''
        handle_error_message(message)

    # if len(list_input_labels_errors_files) != num_labels_witherrrors_percase * len(list_input_labels_files):
    #     message = 'Not equal num files found in input dirs for \'LabelsWorkData\' and \'LabelsWorkData_WithErrors_v*\''
    #     handle_error_message(message)

    list_names_train_images = []
    with open(input_names_train_images_file, 'r') as fin:
        for it_fin_line in fin.readlines():
            it_name_train_image = it_fin_line.replace('\n', '')
            list_names_train_images.append(it_name_train_image)

    list_names_valid_images = []
    with open(input_names_valid_images_file, 'r') as fin:
        for it_fin_line in fin.readlines():
            it_name_valid_image = it_fin_line.replace('\n', '')
            list_names_valid_images.append(it_name_valid_image)

    # ------------------

    count_train_images = 0
    count_valid_images = 0

    for in_image_file, in_label_file in zip(list_input_images_files, list_input_labels_files):
        print("\nInput: \'%s\'..." % (basename(in_image_file)))

        if basename(in_image_file) in list_names_train_images:
            print("File assigned to Training Data...")
            output_dir_this = output_train_dir
            output_templ_image_filenames_this = output_templ_train_image_filenames
            output_templ_label_filenames_this = output_templ_train_label_filenames
            # output_templ_label_error_filenames_this = output_templ_train_label_error_filenames
            count_image_this = count_train_images
            count_train_images += 1

        elif basename(in_image_file) in list_names_valid_images:
            print("File assigned to Validation Data...")
            output_dir_this = output_valid_dir
            output_templ_image_filenames_this = output_templ_valid_image_filenames
            output_templ_label_filenames_this = output_templ_valid_label_filenames
            # output_templ_label_error_filenames_this = output_templ_valid_label_error_filenames
            count_image_this = count_valid_images
            count_valid_images += 1

        else:
            print("File not in Training or Validation Data... Skip")
            continue

        # rootname_label_file = get_substring_filename(in_label_file, pattern_rootname_infiles)
        # list_in_label_error_files = []
        # for iter_file in list_input_labels_errors_files:
        #     iter_rootname_file = get_substring_filename(iter_file, pattern_rootname_infiles)
        #     if iter_rootname_file == rootname_label_file:
        #         list_in_label_error_files.append(iter_file)
        # endfor

        # --------------------

        out_image_file = output_templ_image_filenames_this % (count_image_this)
        out_image_file = join_path_names(output_dir_this, out_image_file)
        out_label_file = output_templ_label_filenames_this % (count_image_this)
        out_label_file = join_path_names(output_dir_this, out_label_file)

        print("%s -> %s..." % (basename(in_image_file), basename(out_image_file)))
        print("%s -> %s..." % (basename(in_label_file), basename(out_label_file)))
        # copyfile(in_image_file, out_image_file)
        # copyfile(in_label_file, out_label_file)
        makelink(in_image_file, out_image_file)
        makelink(in_label_file, out_label_file)

        # for ilab_error, in_label_error_file in enumerate(list_in_label_error_files):
        #     out_label_error_file = output_templ_label_error_filenames_this % (count_image_this, ilab_error + 1)
        #     out_label_error_file = join_path_names(output_dir_this, out_label_error_file)
        #
        #     print("%s -> %s..." % (basename(in_label_error_file), basename(out_label_error_file)))
        #     copyfile(in_label_error_file, out_label_error_file)
        # endfor
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_datadir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
