import numpy as np
import torch
from tqdm import tqdm

from skimage import metrics as skimage_metrics
from skimage.util import img_as_ubyte

# # LPIPS metrics with AlexNet and VGG
# import lpips
# lpips_alex = lpips.LPIPS(net="alex", version="0.1")
# lpips_vgg = lpips.LPIPS(net="vgg", version="0.1")

# # Nanopyx metrics: Error map (RSE and RSP) and decorrelation analysis 
# from nanopyx.core.transform.new_error_map import ErrorMap
# from nanopyx.core.analysis.decorr import DecorrAnalysis

# ILNIQE (in a local file)
from .ILNIQE import calculate_ilniqe

metric_list = ["ssim", "psnr", "mse", "alex", "vgg", "ilniqe", "fsim", "gmsd", "vsi", "haarpsi", "mdsi", "pieapp", "dists", "brisqe", "fid", "gt_rse", "gt_rsp", "pred_rse", "pred_rsp", "decor"]

def calculate_metrics(gt_image, predicted_image, wf_image):
    
    dict_metrics = {metric: None for metric in metric_list}

    # Load the widefield image, ground truth image, and predicted image
    gt_image = gt_image[:, :, 0]
    predicted_image = predicted_image[:, :, 0]
    wf_image = wf_image[:, :, 0]

    # Print info about the images
    print(
        f"gt_image: {gt_image.shape} - {gt_image.min()} {gt_image.max()} - {gt_image.dtype}"
    )
    print(
        f"predicted_image: {predicted_image.shape} - {predicted_image.min()} {predicted_image.max()} - {predicted_image.dtype}"
    )
    print(
        f"wf_image: {wf_image.shape} - {wf_image.min()} {wf_image.max()} - {wf_image.dtype}"
    )

    # Convert the Numpy images into Pytorch tensors
    # Pass the images into Pytorch format (1, 1, X, X)
    gt_image_piq = np.expand_dims(gt_image, axis=0)
    gt_image_piq = np.expand_dims(gt_image_piq, axis=0)
    
    predicted_image_piq = np.expand_dims(predicted_image, axis=0)
    predicted_image_piq = np.expand_dims(predicted_image_piq, axis=0)

    # Pytorch does not support uint16
    if gt_image_piq.dtype == np.uint16:
        gt_image_piq = gt_image_piq.astype(np.uint8)
    if predicted_image_piq.dtype == np.uint16:
        predicted_image_piq = predicted_image_piq.astype(np.uint8) 
        
    # Convert the images into Pytorch tensors
    gt_image_piq = torch.from_numpy(gt_image_piq)
    predicted_image_piq = torch.from_numpy(predicted_image_piq)

    # Assert that there are no negative values
    assert wf_image.min() <= 0. and wf_image.max() >= 0.

    # In case all the predicted values are equal (all zeros for example)
    all_equals = np.all(predicted_image==np.ravel(predicted_image)[0])

    #####################################
    #
    # Calculate the skimage metrics

    dict_metrics["mse"] = skimage_metrics.mean_squared_error(gt_image, predicted_image)

    dict_metrics["ssim"] = skimage_metrics.structural_similarity(predicted_image, gt_image, data_range=1.0)
    dict_metrics["psnr"] = skimage_metrics.peak_signal_noise_ratio(gt_image, predicted_image)

    #
    #####################################

    #####################################
    #
    # Calculate the LPIPS metrics

    # dict_metrics["alex"] = np.squeeze(
    #         lpips_alex(gt_image_piq.float(), predicted_image_piq.float())
    #         .detach()
    #         .numpy()
    #     )

    # dict_metrics["vgg"] = np.squeeze(
    #     lpips_vgg(gt_image_piq.float(), predicted_image_piq.float())
    #     .detach()
    #     .numpy()
    # )

    #
    #####################################

    #####################################
    #
    # Calculate the Nanopyx metrics

    # error_map = ErrorMap()
    # error_map.optimise(wf_image, gt_image)
    # dict_metrics["gt_rse"] = error_map.getRSE()
    # dict_metrics["gt_rsp"] = error_map.getRSP()

    # if not all_equals:
    #     error_map = ErrorMap()
    #     error_map.optimise(wf_image, predicted_image)
    #     dict_metrics["pred_rse"] = error_map.getRSE()
    #     dict_metrics["pred_rsp"] = error_map.getRSP()
    # else: 
    #     dict_metrics["pred_rse"] = np.nan
    #     dict_metrics["pred_rsp"] = np.nan

    # if not all_equals:
    #     decorr_calculator_raw = DecorrAnalysis()
    #     decorr_calculator_raw.run_analysis(predicted_image)
    #     dict_metrics["decor"] = decorr_calculator_raw.resolution
    # else: 
    #     dict_metrics["decor"] = np.nan

    #
    #####################################

    # #####################################
    # #
    # # Calculate the ILNIQE
    
    # # Temporally commented to avoid long evaluation times (83 seconds for each image)
    # if not all_equals:
    #     dict_metrics['ilniqe'] = calculate_ilniqe(img_as_ubyte(predicted_image), 0,
    #                                     input_order='HW', resize=True, version='python')
    # else: 
    #     dict_metrics['ilniqe'] = np.nan
    # #####################################

    return dict_metrics
        
def get_metrics_dict():
    return {metric: [] for metric in metric_list}

def obtain_metrics(gt_image_list, predicted_image_list, wf_image_list, test_metric_indexes):
    """
    Calculate various metrics for evaluating the performance of an image prediction model.

    Args:
        gt_image_list (List[np.ndarray]): A list of ground truth images.
        predicted_image_list (List[np.ndarray]): A list of predicted images.
        wf_image_list (List[np.ndarray]): A list of wavefront images.
        test_metric_indexes (List[int]): A list of indexes to calculate additional metrics.

    Returns:
        dict: A dictionary containing different metrics as keys and their corresponding values as lists.

    Raises:
        AssertionError: If the minimum value of the wavefront image is greater than 0 or the maximum value is less than 0.

    Note:
        This function uses various image metrics including MSE, SSIM, PSNR, GT RSE, GT RSP, Pred RSE, Pred RSP, and Decorrelation.
        It also calculates metrics using the LPIPS (Learned Perceptual Image Patch Similarity) model, ILNIQE (Image Lab Non-Reference Image Quality Evaluation), and other metrics.
        The calculated metrics are stored in a dictionary with the metric names as keys and lists of values as their corresponding values.
    """

    disct_metrics_lists = get_metrics_dict()

    for i in tqdm(range(len(gt_image_list))):
        disct_metrics_items = calculate_metrics(
            gt_image_list[i],
            predicted_image_list[i],
            wf_image_list[i],
        )

        for key, value in disct_metrics_items.items():
            if key in test_metric_indexes:
                disct_metrics_lists[key].append(value)

    return disct_metrics_lists
