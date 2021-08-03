
from typing import List, Tuple, Dict, Any
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
    def get_data(cls, input_file: str) -> Dict[str, Any]:
        with open(input_file, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',')

            list_fields = next(csv_reader)  # read header
            list_fields = [elem.lstrip() for elem in list_fields]  # remove empty leading spaces ' '

            # output data as dictionary (key: field name, value: field data column)
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

                    if in_value_str == 'NaN' or in_value_str == 'nan':
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

    @classmethod
    def write_data(cls, output_file: str, out_dict_data: Dict[str, Any], format_out_data: List[str] = None) -> None:
        with open(output_file, 'w') as fout:

            list_fields = list(out_dict_data.keys())
            str_header = ', '.join(list_fields) + '\n'
            fout.write(str_header)

            list_out_list_data = list(out_dict_data.values())
            num_cols = len(list_fields)
            num_rows = len(list_out_list_data[0])

            if format_out_data is None:
                format_out_data = ['%0.3f'] * num_cols

            for irow in range(num_rows):
                list_data_row = [list_out_list_data[icol][irow] for icol in range(num_cols)]
                str_data_row = ', '.join([format_out_data[i] % (elem) for i, elem in enumerate(list_data_row)]) + '\n'
                fout.write(str_data_row)

    @classmethod
    def write_data_other(cls, output_file: str, out_dict_data: Dict[str, Any]) -> None:
        with open(output_file, 'w') as fout:
            csv_writer = csv.writer(fout, delimiter=',')

            list_fields = list(out_dict_data.keys())
            csv_writer.writerow(list_fields)

            list_out_list_data = list(out_dict_data.values())
            num_cols = len(list_fields)
            num_rows = len(list_out_list_data[0])

            for irow in range(num_rows):
                list_data_row = [list_out_list_data[icol][irow] for icol in range(num_cols)]
                csv_writer.writerow(list_data_row)
