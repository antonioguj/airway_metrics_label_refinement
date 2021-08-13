
from common.functionutil import *
import argparse

def main(args):

    # SETTINGS
    input_images_dir = join_path_names(args.input_datadir, './ImagesWorkData/')
    input_labels_dir = join_path_names(args.input_datadir, './LabelsWorkData/')
    input_names_train_images_file = join_path_names(args.input_datadir, 'traindatafiles_20imgs.txt')
    input_names_valid_images_file = join_path_names(args.input_datadir, 'validdatafiles_4imgs.txt')
    in_reference_keys_file = join_path_names(args.input_datadir, 'referenceKeys_nnUnetimages.npy')

    output_train_dir = join_path_names(args.input_datadir, './TrainData_20imgs/')
    output_valid_dir = join_path_names(args.input_datadir, './ValidData_4imgs/')
    output_test_dir = join_path_names(args.input_datadir, './TestData/')
    # --------

    makedir(output_train_dir)
    makedir(output_valid_dir)
    makedir(output_test_dir)

    list_input_images_files = sorted(list_files_dir(input_images_dir))
    list_input_labels_files = sorted(list_files_dir(input_labels_dir))
    indict_reference_keys = dict(np.load(in_reference_keys_file, allow_pickle=True).item())

    if len(list_input_images_files) != len(list_input_labels_files):
        message = 'Not equal num files found in input dirs for \'ImagesWorkData\' and \'LabelsWorkData\''
        handle_error_message(message)

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

    for in_image_file, in_label_file in zip(list_input_images_files, list_input_labels_files):
        print("\nInput: \'%s\'..." % (basename(in_label_file)))
        in_referencekey_image = indict_reference_keys[basename_filenoext(in_label_file)]
        print("Or Image: \'%s\'..." % (in_referencekey_image))

        if in_referencekey_image in list_names_train_images:
            print("File assigned to Training Data...")
            output_dir_this = output_train_dir
        elif in_referencekey_image in list_names_valid_images:
            print("File assigned to Validation Data...")
            output_dir_this = output_valid_dir
        else:
            print("File assigned to Testing Data...")
            output_dir_this = output_test_dir

        out_image_file = join_path_names(output_dir_this, basename(in_image_file))
        out_label_file = join_path_names(output_dir_this, basename(in_label_file))

        print("%s -> %s..." % (basename(in_image_file), basename(out_image_file)))
        print("%s -> %s..." % (basename(in_label_file), basename(out_label_file)))
        #copyfile(in_image_file, out_image_file)
        #copyfile(in_label_file, out_label_file)
        makelink(join_path_names('../', in_image_file), out_image_file)
        makelink(join_path_names('../', in_label_file), out_label_file)
    # endfor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_datadir', type=str)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).items():
        print("\'%s\' = %s" %(key, value))

    main(args)
