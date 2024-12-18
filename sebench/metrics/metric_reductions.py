# local imports for:
from .utils import reduce_bin_errors
# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Union, Literal, Optional


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_reduction(
    cal_info: dict,
    metric_type: str,
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    """
    Calculates the reduction for Expected Calibration Error (ECE) metrics.
    """
    # Finally, get the calibration score.
    cal_info['cal_error'] = reduce_bin_errors(
        error_per_bin=cal_info["bin_cal_errors"], 
        amounts_per_bin=cal_info["bin_amounts"]
        )
    # Return the calibration information.
    if cal_info['bin_amounts'].sum() > 0:
        assert 0.0 <= cal_info['cal_error'] <= 1.0,\
            f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def class_ece_reduction(
    cal_info: dict,
    metric_type: str,
    class_weighting: Literal['uniform', 'proportional'],
    ignore_index: Optional[int] = None,
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    """
    Calculates the reduction for class-based Expected Calibration Error (C ECE) metrics.
    """
    # If there are no samples, then the ECE is 0.
    if cal_info['bin_amounts'].sum() == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
        if return_dict:
            cal_info['metric_type'] = metric_type
            return cal_info
        else:
            return cal_info['cal_error']
    # Go through each label and calculate the ECE.
    L, _ = cal_info["bin_cal_errors"].shape
    score_per_lab = torch.zeros(L)
    weights_per_lab = torch.zeros(L)
    amounts_per_lab = torch.zeros(L)
    # Iterate through each label and calculate the weighted ece.
    for lab_idx in range(L):
        # If we are ignoring an index, skip it in calculations.
        if (ignore_index is None) or (lab_idx != ignore_index):
            lab_ece = reduce_bin_errors(
                error_per_bin=cal_info['bin_cal_errors'][lab_idx], 
                amounts_per_bin=cal_info['bin_amounts'][lab_idx], 
                )
            lab_amount = cal_info['bin_amounts'][lab_idx].sum()
            amounts_per_lab[lab_idx] = lab_amount
            # If uniform then apply no weighting.
            if class_weighting == 'uniform':
                lab_prob = 1.0 if lab_amount > 0 else 0.0
            else:
                lab_prob = lab_amount 
            # Weight the ECE by the prob of the label.
            score_per_lab[lab_idx] = lab_ece
            weights_per_lab[lab_idx] = lab_prob
    # Calculate the wECE per bin by probs.
    total_weight = weights_per_lab.sum()
    if total_weight > 0:
        prob_per_lab = weights_per_lab / total_weight
        ece_per_lab = score_per_lab * prob_per_lab
        # Finally, get the calibration score.
        cal_info['cal_error'] = ece_per_lab.sum()
    else:
        cal_info['cal_error'] = torch.tensor(0.0)
    # If cal_error is not nan, then it should be in [0, 1].
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def elm_reduction(
    cal_info: dict,
    metric_type: str,
    class_weighting: Literal['uniform', 'proportional'],
    return_dict: bool = False,
) -> Union[dict, Tensor]:
    """
    Calculates the reduction for Expected Local Miscalibration (ELM) metrics.
    """
    # If there are no samples, then the ECE is 0.
    if cal_info['bin_amounts'].sum() == 0:
        cal_info['cal_error'] = torch.tensor(0.0)
        if return_dict:
            cal_info['metric_type'] = metric_type
            return cal_info
        else:
            return cal_info['cal_error']
    # Go through each neighborhood class and calculate the ECE.
    NN, _ = cal_info["bin_cal_errors"].shape
    score_per_nn = torch.zeros(NN)
    weights_per_nn = torch.zeros(NN)
    amounts_per_nn = torch.zeros(NN)
    # Iterate through each neighborhood class and calculate the weighted ece.
    for nn_idx in range(NN):
        # If we are ignoring an index, skip it in calculations.
        nn_ece = reduce_bin_errors(
            error_per_bin=cal_info['bin_cal_errors'][nn_idx], 
            amounts_per_bin=cal_info['bin_amounts'][nn_idx], 
        )
        nn_amount = cal_info['bin_amounts'][nn_idx].sum()
        amounts_per_nn[nn_idx] = nn_amount
        # If uniform then apply no weighting.
        if class_weighting == 'uniform':
            nn_prob = 1.0 if nn_amount > 0 else 0.0
        else:
            nn_prob = nn_amount
        # Weight the ECE by the prob of the neighborhood class.
        score_per_nn[nn_idx] = nn_ece
        weights_per_nn[nn_idx] = nn_prob
    # Calculate the wECE per bin by probs.
    total_weight = weights_per_nn.sum()
    if total_weight > 0:
        prob_per_nn = weights_per_nn / total_weight
        ece_per_nn = score_per_nn * prob_per_nn
        # Finally, get the calibration score.
        cal_info['cal_error'] = ece_per_nn.sum()
    else:
        cal_info['cal_error'] = torch.tensor(0.0)
    # Return the calibration information.
    assert 0.0 <= cal_info['cal_error'] <= 1.0,\
        f"Expected calibration error to be in [0, 1]. Got {cal_info['cal_error']}."
    # Return the calibration information.
    if return_dict:
        cal_info['metric_type'] = metric_type
        return cal_info
    else:
        return cal_info['cal_error']