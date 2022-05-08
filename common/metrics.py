
import numpy as np
from scipy.spatial import distance

from common.functionutil import handle_error_message, compute_dilated_mask, compute_connected_components

_EPS = 1.0e-7
_SMOOTH = 1.0

LIST_AVAIL_METRICS = ['DiceCoefficient',
                      'AirwayCompleteness',
                      'AirwayVolumeLeakage',
                      'AirwayCentrelineLeakage',
                      'AirwayTreeLength',
                      'AirwayCentrelineDistanceFalsePositiveError',
                      'AirwayCentrelineDistanceFalseNegativeError',
                      'AirwayNumberFNErrors',
                      'AirwayNumberFNGAPErrors',
                      ]


class MetricBase(object):
    _is_airway_metric = False
    _is_use_voxelsize = False

    def __init__(self) -> None:
        self._name_fun_out = None

    def compute(self, target: np.ndarray, input: np.ndarray, *args) -> np.ndarray:
        if self._is_airway_metric:
            target_cenline = args[0]
            input_cenline = args[1]
            return self._compute_airs(target, target_cenline, input, input_cenline)
        else:
            return self._compute(target, input)

    def _compute(self, target: np.ndarray, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def set_voxel_size(self, voxel_size: np.ndarray) -> None:
        self._voxel_size = np.array(voxel_size)


class DiceCoefficient(MetricBase):

    def __init__(self) -> None:
        super(DiceCoefficient, self).__init__()
        self._name_fun_out = 'dice'

    def _compute(self, target: np.ndarray, input: np.ndarray) -> np.ndarray:
        return (2.0 * np.sum(target * input)) / (np.sum(target) + np.sum(input) + _SMOOTH)


class DiceCoefficientMaskedTraining(MetricBase):

    def __init__(self) -> None:
        super(DiceCoefficientMaskedTraining, self).__init__()
        self._name_fun_out = 'dice_masked'

    def _get_masked_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.where(target == -1, 0, input)

    def _compute(self, target: np.ndarray, input: np.ndarray) -> np.ndarray:
        input = self._get_masked_input(input, target)
        target = self._get_masked_input(target, target)
        return (2.0 * np.sum(target * input)) / (np.sum(target) + np.sum(input) + _SMOOTH)


class AirwayCompleteness(MetricBase):
    _is_airway_metric = True

    def __init__(self) -> None:
        super(AirwayCompleteness, self).__init__()
        self._name_fun_out = 'completeness'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        return np.sum(target_cenline * input) / (np.sum(target_cenline) + _SMOOTH)


class AirwayVolumeLeakage(MetricBase):
    _is_airway_metric = True

    def __init__(self) -> None:
        super(AirwayVolumeLeakage, self).__init__()
        self._name_fun_out = 'volume_leakage'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - target) * input) / (np.sum(target) + _SMOOTH)


class AirwayVolumeLeakageDilatedGT(MetricBase):
    _is_airway_metric = True

    def __init__(self) -> None:
        super(AirwayVolumeLeakageDilatedGT, self).__init__()
        self._name_fun_out = 'volume_leakage_dil3'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        target_eval = compute_dilated_mask(target, in_struct='cube')
        return np.sum((1.0 - target_eval) * input) / (np.sum(target) + _SMOOTH)


class AirwayCentrelineLeakage(MetricBase):
    _is_airway_metric = True

    def __init__(self) -> None:
        super(AirwayCentrelineLeakage, self).__init__()
        self._name_fun_out = 'cenline_leakage'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        return np.sum((1.0 - target) * input_cenline) / (np.sum(target_cenline) + _SMOOTH)


class AirwayTreeLength(MetricBase):
    _is_airway_metric = True
    _is_use_voxelsize = True

    def __init__(self) -> None:
        super(AirwayTreeLength, self).__init__()
        self._name_fun_out = 'tree_length'

    def _get_voxel_length_unit(self) -> np.ndarray:
        return np.prod(self._voxel_size) ** (1.0 / len(self._voxel_size))

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        return np.sum(target_cenline * input) * self._get_voxel_length_unit()


class AirwayCentrelineDistanceFalsePositiveError(MetricBase):
    _is_airway_metric = True
    _is_use_voxelsize = True

    def __init__(self) -> None:
        super(AirwayCentrelineDistanceFalsePositiveError, self).__init__()
        self._name_fun_out = 'cenline_dist_fp_err'

    def _get_cenline_coords(self, input_cenline: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(input_cenline > 0)) * self._voxel_size

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        target_coords = self._get_cenline_coords(target_cenline)
        input_coords = self._get_cenline_coords(input_cenline)
        dists = distance.cdist(input_coords, target_coords)
        return np.mean(np.min(dists, axis=1))


class AirwayCentrelineDistanceFalseNegativeError(MetricBase):
    _is_airway_metric = True
    _is_use_voxelsize = True

    def __init__(self) -> None:
        super(AirwayCentrelineDistanceFalseNegativeError, self).__init__()
        self._name_fun_out = 'cenline_dist_fn_err'

    def _get_cenline_coords(self, input_cenline: np.ndarray) -> np.ndarray:
        return np.asarray(np.argwhere(input_cenline > 0)) * self._voxel_size

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        target_coords = self._get_cenline_coords(target_cenline)
        input_coords = self._get_cenline_coords(input_cenline)
        dists = distance.cdist(input_coords, target_coords)
        return np.mean(np.min(dists, axis=0))


class AirwayNumberFNErrors(MetricBase):
    _is_airway_metric = True
    _is_dilate_rm_noise = False

    def __init__(self) -> None:
        super(AirwayNumberFNErrors, self).__init__()
        self._name_fun_out = 'number_fn_err'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        if self._is_dilate_rm_noise:
            input = compute_dilated_mask(input)
        falseneg_target_cenline = target_cenline * (1.0 - input)
        (_, num_errors) = compute_connected_components(falseneg_target_cenline, connectivity_dim=3)
        return np.array(num_errors)


class AirwayNumberFNGAPErrors(MetricBase):
    _is_airway_metric = True
    _is_dilate_rm_noise = False

    def __init__(self) -> None:
        super(AirwayNumberFNGAPErrors, self).__init__()
        self._name_fun_out = 'number_fn_gap_err'

    def _compute_airs(self, target: np.ndarray, target_cenline: np.ndarray,
                      input: np.ndarray, input_cenline: np.ndarray) -> np.ndarray:
        truepos_target_cenline = target_cenline * input
        if self._is_dilate_rm_noise:
            target_cenline = compute_dilated_mask(target_cenline)
            truepos_target_cenline = compute_dilated_mask(truepos_target_cenline)
        (_, num_regions_init) = compute_connected_components(target_cenline, connectivity_dim=3)
        (_, num_regions_truepos) = compute_connected_components(truepos_target_cenline, connectivity_dim=3)
        num_gaps = num_regions_truepos - num_regions_init
        return np.array(num_gaps)


def get_metric(type_metric: str, **kwargs) -> MetricBase:
    if type_metric == 'DiceCoefficient':
        return DiceCoefficient()
    elif type_metric == 'AirwayCompleteness':
        return AirwayCompleteness()
    elif type_metric == 'AirwayVolumeLeakage':
        return AirwayVolumeLeakage()
    elif type_metric == 'AirwayVolumeLeakageDilatedGT':
        return AirwayVolumeLeakageDilatedGT()
    elif type_metric == 'AirwayCentrelineLeakage':
        return AirwayCentrelineLeakage()
    elif type_metric == 'AirwayTreeLength':
        return AirwayTreeLength()
    elif type_metric == 'AirwayCentrelineDistanceFalsePositiveError':
        return AirwayCentrelineDistanceFalsePositiveError()
    elif type_metric == 'AirwayCentrelineDistanceFalseNegativeError':
        return AirwayCentrelineDistanceFalseNegativeError()
    elif type_metric == 'AirwayNumberFNErrors':
        return AirwayNumberFNErrors()
    elif type_metric == 'AirwayNumberFNGAPErrors':
        return AirwayNumberFNGAPErrors()
    else:
        message = 'Choice Metric not found: \'%s\'. Metrics available: \'%s\'' \
                  % (type_metric, ', '.join(LIST_AVAIL_METRICS))
        handle_error_message(message)
