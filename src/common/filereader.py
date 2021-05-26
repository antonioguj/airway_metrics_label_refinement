
from typing import Tuple, Any
from collections import OrderedDict
import numpy as np
import nibabel as nib
import csv


class NiftiFileReader(object):

    @staticmethod
    def get_image_voxelsize(filename: str) -> Tuple[float, float, float]:
        affine = nib.load(filename).affine
        return tuple(np.abs(np.diag(affine)[:3]))

    @classmethod
    def get_image_size(cls, filename: str) -> Tuple[float, float, float]:
        return cls.get_image(filename).shape

    @staticmethod
    def get_image_metadata_info(filename: str) -> Any:
        return nib.load(filename).affine

    @staticmethod
    def get_image(filename: str) -> np.ndarray:
        out_image = nib.load(filename).get_data()
        return np.swapaxes(out_image, 0, 2)

    @staticmethod
    def write_image(filename: str, in_image: np.ndarray, **kwargs) -> None:
        affine = kwargs['metadata'] if 'metadata' in kwargs.keys() else None
        in_image = np.swapaxes(in_image, 0, 2)
        nib_image = nib.Nifti1Image(in_image, affine)
        nib.save(nib_image, filename)


class CsvFileReader(object):

    @staticmethod
    def get_data_type(in_value_str: str) -> str:
        if in_value_str.isdigit():
            if in_value_str.count(' ') > 1:
                return 'group_integer'
            else:
                return 'integer'
        elif in_value_str.replace('.', '', 1).isdigit() and in_value_str.count('.') < 2:
            return 'float'
        else:
            return 'string'

    @classmethod
    def get_data(cls, input_file: str):
        with open(input_file, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',')

            # read header and get field labels
            list_fields = next(csv_reader)
            list_fields = [elem.lstrip() for elem in list_fields]  # remove empty leading spaces ' '

            # output data as dictionary with (key: field_name, value: field data, same column)
            out_dict_data = OrderedDict([(ifield, []) for ifield in list_fields])

            num_fields = len(list_fields)
            for irow, row_data in enumerate(csv_reader):
                row_data = [elem.lstrip() for elem in row_data]  # remove empty leading spaces ' '

                if irow == 0:
                    # get the data type for each field
                    list_datatype_fields = []
                    for ifie in range(num_fields):
                        in_value_str = row_data[ifie]
                        in_data_type = cls.get_data_type(in_value_str)
                        list_datatype_fields.append(in_data_type)

                for ifie in range(num_fields):
                    field_name = list_fields[ifie]
                    in_value_str = row_data[ifie]
                    in_data_type = list_datatype_fields[ifie]

                    if in_value_str == 'NaN':
                        out_value = np.NaN
                    elif in_data_type == 'integer':
                        out_value = int(in_value_str)
                    elif in_data_type == 'group_integer':
                        out_value = tuple([int(elem) for elem in in_value_str.split(' ')])
                    elif in_data_type == 'float':
                        out_value = float(in_value_str)
                    else:
                        out_value = in_value_str

                    out_dict_data[field_name].append(out_value)

        return out_dict_data
