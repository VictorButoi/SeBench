# torch imports
from torch import Tensor
# misc imports
import matplotlib.pyplot as plt
from pydantic import validate_arguments
from typing import Dict, Optional, Union, List, Literal
# ionpy imports
from ionpy.util.meter import Meter
from ionpy.loss.util import _loss_module_from_func
# local imports for:
# - pixel statistics
from .metric_reductions import (
    ece_reduction,
    class_ece_reduction
)
from .local_ps import (
    bin_stats, 
    top_label_bin_stats,
    joint_label_bin_stats
)
# - global statistics
from .global_ps import (
    tl_prob_bin_stats,
    classwise_prob_bin_stats
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
    **kwargs
) -> Union[dict, Tensor]:
    
    # Calculate the mean statistics per bin.
    cal_info = bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_prob_bins=num_prob_bins,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        from_logits=from_logits,
        preloaded_obj_dict=preloaded_obj_dict
    )
    metric_dict = {
        "metric_type": "local",
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Local Bin counts: ", cal_info["bin_amounts"])
    # print("Local Bin cal errors: ", cal_info["bin_cal_errors"])
    # Return the calibration information
    return ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_tl_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
    **kwargs
) -> Union[dict, Tensor]:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    cal_info = top_label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_prob_bins=num_prob_bins,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        from_logits=from_logits,
        preloaded_obj_dict=preloaded_obj_dict
    )
    metric_dict = {
        "metric_type": "local",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "ignore_index": ignore_index,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # Return the calibration information
    return class_ece_reduction(**metric_dict)



@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_cw_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    num_prob_bins: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    from_logits: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    preloaded_obj_dict: Optional[dict] = None,
    **kwargs
) -> Union[dict, Tensor]:
    # Quick check to see if we are dealing with binary segmentation
    if y_pred.shape[1] == 1:
        assert ignore_index is None, "ignore_index is not supported for binary segmentation."

    cal_info = joint_label_bin_stats(
        y_pred=y_pred,
        y_true=y_true,
        num_prob_bins=num_prob_bins,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only,
        from_logits=from_logits,
        preloaded_obj_dict=preloaded_obj_dict
    )
    metric_dict = {
        "metric_type": "local",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "ignore_index": ignore_index,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Local Bin counts: ", cal_info["bin_amounts"])
    # print("Local Bin cal errors: ", cal_info["bin_cal_errors"])
    # Return the calibration information
    return class_ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    num_prob_bins: int,
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    **kwargs
) -> Union[dict, Tensor]:
    cal_info = tl_prob_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        num_prob_bins=num_prob_bins,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only
    )
    metric_dict = {
        "metric_type": "global",
        "cal_info": cal_info,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Global Bin counts: ", cal_info["bin_amounts"])
    # print("Global Bin cal errors: ", cal_info["bin_cal_errors"])
    # Return the calibration information
    return ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def tl_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    num_prob_bins: int,
    num_classes: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
) -> Union[dict, Tensor]:
    cal_info = classwise_prob_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        num_prob_bins=num_prob_bins,
        num_classes=num_classes,
        class_wise=True,
        local=False,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only
    )
    metric_dict = {
        "metric_type": "global",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "ignore_index": ignore_index,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Global Bin counts: ", cal_info["bin_amounts"])
    # print("Global Bin cal errors: ", cal_info["bin_cal_errors"])
    # Return the calibration information
    return class_ece_reduction(**metric_dict)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    num_prob_bins: int,
    num_classes: int,
    class_weighting: Literal["uniform", "proportional"],
    edge_only: bool = False,
    square_diff: bool = False,
    neighborhood_width: Optional[int] = None,
    ignore_index: Optional[int] = None,
    **kwargs
) -> Union[dict, Tensor]:
    # Get the statistics either from images or pixel meter dict.
    cal_info = classwise_prob_bin_stats(
        pixel_meters_dict=pixel_meters_dict,
        num_prob_bins=num_prob_bins,
        num_classes=num_classes,
        class_wise=True,
        local=False,
        square_diff=square_diff,
        neighborhood_width=neighborhood_width,
        edge_only=edge_only
    )
    metric_dict = {
        "metric_type": "global",
        "cal_info": cal_info,
        "class_weighting": class_weighting,
        "ignore_index": ignore_index,
        "return_dict": kwargs.get("return_dict", False) 
    }
    # print("Global Bin counts: ", cal_info["bin_amounts"])
    # print("Global Bin cal errors: ", cal_info["bin_cal_errors"])
    # Return the calibration information
    return class_ece_reduction(**metric_dict)

# Edge only versions of the above functions.
##################################################################################################

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_edge_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_etl_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_tl_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def image_ecw_ece_loss(
    y_pred: Tensor,
    y_true: Tensor,
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["y_pred"] = y_pred
    kwargs["y_true"] = y_true
    kwargs["edge_only"] = True
    # Return the calibration information
    return image_cw_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def edge_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def etl_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return tl_ece_loss(**kwargs)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def ecw_ece_loss(
    pixel_meters_dict: Dict[tuple, Meter],
    **kwargs
) -> Union[dict, Tensor]:
    assert "neighborhood_width" in kwargs, "Must provide neighborhood width if doing an edge metric."
    kwargs["pixel_meters_dict"] = pixel_meters_dict 
    kwargs["edge_only"] = True
    # Return the calibration information
    return cw_ece_loss(**kwargs)


#############################################################################
# Global metrics
#############################################################################

# Loss modules
ECE = _loss_module_from_func("ECE", ece_loss)
TL_ECE = _loss_module_from_func("TL_ECE", tl_ece_loss)
CW_ECE = _loss_module_from_func("CW_ECE", cw_ece_loss)

# Edge loss modules
Edge_ECE = _loss_module_from_func("Edge_ECE", edge_ece_loss)
ETL_ECE = _loss_module_from_func("ETL_ECE", etl_ece_loss)
ECW_ECE = _loss_module_from_func("ECW_ECE", ecw_ece_loss)

#############################################################################
# Image-based metrics
#############################################################################

# Loss modules
Image_ECE = _loss_module_from_func("Image_ECE", image_ece_loss)
Image_TL_ECE = _loss_module_from_func("Image_TL_ECE", image_tl_ece_loss)
Image_CW_ECE = _loss_module_from_func("Image_CW_ECE", image_cw_ece_loss)

# Edge loss modules
Image_Edge_ECE = _loss_module_from_func("Image_Edge_ECE", image_edge_ece_loss)
Image_ETL_ECE = _loss_module_from_func("Image_ETL_ECE", image_etl_ece_loss)
Image_ECW_ECE = _loss_module_from_func("Image_ECW_ECE", image_ecw_ece_loss)