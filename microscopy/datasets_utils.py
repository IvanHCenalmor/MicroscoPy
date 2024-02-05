import numpy as np
import os
from skimage import io

from . import crappifiers
from .utils import min_max_normalization as normalization

#####################################
#
# Functions to sample an image using a probability density function.
# Code from: https://github.com/esgomezm/microscopy-dl-suite-tf/blob/fcb8870624208bfb72dc7aea18a90738a081217f/dl-suite/utils

def index_from_pdf(pdf_im):
    """
    Generate the index coordinates from a probability density function (pdf) image.

    Parameters:
    - pdf_im: numpy.ndarray
        The input pdf image.

    Returns:
    - tuple
        A tuple containing the index coordinates (indexh, indexw) of the randomly chosen element from the pdf image.

    Example:
    ```
    pdf_image = np.array([[0.1, 0.2, 0.3],
                          [0.2, 0.3, 0.4],
                          [0.3, 0.4, 0.5]])
    index = index_from_pdf(pdf_image)
    print(index)  # (1, 2)
    ```
    """
    prob = np.copy(pdf_im)

    # Normalize values to create a pdf with sum = 1
    prob = prob.ravel() / np.sum(prob)
    
    # Convert into a 1D pdf
    choices = np.prod(pdf_im.shape)
    index = np.random.choice(choices, size=1, p=prob)
    
    # Recover 2D shape
    coordinates = np.unravel_index(index, shape=pdf_im.shape)
    
    # Extract index
    indexh = coordinates[0][0]
    indexw = coordinates[1][0]
    
    return indexh, indexw

def sampling_pdf(y, pdf_flag, height, width):
    """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Parameters:
        - y: the input array
        - pdf_flag: a flag indicating whether to select indexes randomly (0) or based on a PDF (1)
        - height: the height of the crop
        - width: the width of the crop

        Returns:
        - indexh: the index for the center of the crop along the height dimension
        - indexw: the index for the center of the crop along the width dimension
    """

    # Obtain the height and width of the input array
    h, w = y.shape[0], y.shape[1]

    if pdf_flag == 0:
         # If pdf_flag is 0 then select the indexes for the center of the crop randomly
        indexw = np.random.randint(
            np.floor(width // 2),
            max(w - np.floor(width // 2), np.floor(width // 2) + 1),
        )
        indexh = np.random.randint(
            np.floor(height // 2),
            max(h - np.floor(height // 2), np.floor(height // 2) + 1),
        )
    else:
        # If pdf_flag is 1 then select the indexes for the center of the crop based on a PDF

        # crop to fix patch size
        # croped_y = y[int(np.floor(height // 2)):-int(np.floor(height // 2)),
        #              int(np.floor(width // 2)) :-int(np.floor(width // 2))]
        # indexh, indexw = index_from_pdf(croped_y)

        # In order to speed the process, this is done on the Fourier domain
        kernel = np.ones((height, width))

        pdf = np.fft.irfft2(np.fft.rfft2(y) * np.fft.rfft2(kernel, y.shape))
        pdf = normalization(pdf)
        pdf_cropped = pdf[
            min(kernel.shape[0], pdf.shape[0] - 1) :,
            min(kernel.shape[1], pdf.shape[1] - 1) :,
        ]

        indexh, indexw = index_from_pdf(pdf_cropped)
        indexw = indexw + int(np.floor(width // 2))
        indexh = indexh + int(np.floor(height // 2))

    return indexh, indexw

#
#####################################

#####################################
#
# Functions to read image pairs from a given path.

def read_image(file_path, desired_accuracy=np.float32):
    """
    Reads an image from a given file path.

    Args:
        file_path (str): The path to the image file.
        desired_accuracy (type, optional): The desired accuracy of the image. Defaults to np.float32.

    Returns:
        The normalized image.
    """
    return normalization(io.imread(file_path), desired_accuracy=desired_accuracy)


def obtain_scale_factor(hr_filename, lr_filename, scale_factor, crappifier_name):
    """
    Calculates the scale factor between a low-resolution image and a high-resolution image.

    Args:
        hr_filename (str): The path to the high-resolution image file.
        lr_filename (str): The path to the low-resolution image file.
        scale_factor (int): The scale factor to be applied to the low-resolution image.
        crappifier_name (str): The name of the crappifier to use for generating the low-resolution image.

    Raises:
        ValueError: If no scale factor is given and no low-resolution image file is provided.

    Returns:
        int: The scale factor of the images.
    """
    
    if scale_factor is None and lr_filename is None:
        # In case that there is no LR image and no scale factor is given, raise an error
        raise ValueError("A scale factor has to be given.")

    # HR image should always be given, herefore read it
    hr_img = read_image(hr_filename)
    
    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated with a crappifier
        lr_img = normalization(
            crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name)
        )
    else:
        # Otherwise, read the LR image
        lr_img = read_image(lr_filename)

    # Obtain the real scale factor of the image
    images_scale_factor = hr_img.shape[0] // lr_img.shape[0]

    return images_scale_factor


def read_image_pairs(hr_filename, lr_filename, scale_factor, crappifier_name):
    """
    Reads a pair of high-resolution (HR) and low-resolution (LR) images and returns them.

    Parameters:
        hr_filename (str): The path to the HR image file.
        lr_filename (str): The path to the LR image file. If None, the LR image will be artificially generated.
        scale_factor (int): The scale factor for downsampling the LR image.
        crappifier_name (str): The name of the crappifier to be used for generating the LR image.

    Returns:
        tuple: A tuple containing the HR and LR images.
            - hr_img (ndarray): The high-resolution image.
            - lr_img (ndarray): The low-resolution image.
    """
    hr_img = read_image(hr_filename)

    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated with a crappifier
        lr_img = normalization(
            crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name)
        )
    else:
        # Otherwise, read the LR image
        lr_img = read_image(lr_filename)

        # Then calculate the scale factor
        images_scale_factor = hr_img.shape[0] // lr_img.shape[0]

        if scale_factor > images_scale_factor:
            # And in case that the given scale factor is larger than the real scale factor of the images,
            # downsample the low-resolution image to match the given scale factor 
            lr_img = normalization(
                crappifiers.apply_crappifier(
                    lr_img, scale_factor // images_scale_factor, "downsampleonly"
                )
            )

    return hr_img, lr_img

#
#####################################

#####################################
#
# Functions to read images and extract patches from them.

def extract_random_patches_from_image(
    hr_filename,
    lr_filename,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    verbose = 0
):
    """
    Extracts random patches from an image.

    :param hr_filename: The path to the high-resolution image file.
    :param lr_filename: The path to the low-resolution image file.
    :param scale_factor: The scale factor used for downsampling the image.
    :param crappifier_name: The name of the crappifier used for generating the low-resolution image.
    :param lr_patch_shape: The shape of the patches in the low-resolution image. If None, the complete image will be used.
    :param datagen_sampling_pdf: A flag indicating whether a probability density function (PDF) is used for sampling the patch coordinates.
    :return: A tuple containing the low-resolution and high-resolution patches.
    :raises ValueError: If the patch size is bigger than the given images.
    """

    # First lets read the images from given paths
    hr_img, lr_img = read_image_pairs(
        hr_filename, lr_filename, scale_factor, crappifier_name
    )

    if lr_patch_shape is None:
        # In case that the patch shape (on the low-resolution image) is not given, 
        # the complete image will be used
        lr_patch_size_width = lr_img.shape[0]
        lr_patch_size_height = lr_img.shape[1]
    else:
        # Otherwise, use the given patch shape
        lr_patch_size_width = lr_patch_shape[0]
        lr_patch_size_height = lr_patch_shape[1]

    if (
        lr_img.shape[0] < lr_patch_size_width
        or hr_img.shape[0] < lr_patch_size_width * scale_factor
    ):
        # In case that the patch size is bigger than the given images, raise an error
        raise ValueError("Patch size is bigger than the given images.")

    if (
        lr_patch_size_width == lr_img.shape[0]
        and lr_patch_size_height == lr_img.shape[1]
    ):
        # In case that the patch size is the same as the given images, return the images
        lr_patch = lr_img
        hr_patch = hr_img
    else:
        # Otherwise, extract the patch 

        # For that the indexes for the center of the patch are calculated (using a PDF or ranfomly)
        lr_idx_width, lr_idx_height = sampling_pdf(
            y=lr_img,
            pdf_flag=datagen_sampling_pdf,
            height=lr_patch_size_height,
            width=lr_patch_size_width,
        )

        # Calculate the lower-row (lr) and upper-row (ur) coordinates
        lr = int(lr_idx_height - np.floor(lr_patch_size_height // 2))
        ur = int(lr_idx_height + np.round(lr_patch_size_height // 2))

        # Calculate the lower-column (lc) and upper-column (uc) coordinates
        lc = int(lr_idx_width - np.floor(lr_patch_size_width // 2))
        uc = int(lr_idx_width + np.round(lr_patch_size_width // 2))

        # Extract the patches
        lr_patch = lr_img[lc:uc, lr:ur]
        hr_patch = hr_img[
            lc * scale_factor : uc * scale_factor, lr * scale_factor : ur * scale_factor
        ]

    if verbose > 3:
        print('\nExtracting patches:')
        print("lr_patch[{}:{}, {}:{}] - {} - min: {} max: {}".format(lc, uc, lr, ur, lr_patch.shape,
                                                                lr_patch.min(), lr_patch.max()))
        print(lr_filename)
        print(f'\tLR_patch: {lr_patch[0,:5]}')
        print(f'\t{lr_img[0,:5]}')
        print("hr_patch[{}:{}, {}:{}] - {} - min: {} max: {}".format(lc * scale_factor, uc * scale_factor, 
                                              lr * scale_factor, ur * scale_factor, hr_patch.shape,
                                              hr_patch.min(), hr_patch.max()))
        print(f'\t{hr_patch[0,:5]}')
        print(f'\t{hr_img[0,:5]}')
        print(hr_filename)

    return lr_patch, hr_patch


def extract_random_patches_from_folder(
    hr_data_path,
    lr_data_path,
    filenames,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    verbose = 0
):
    """
    Extracts random patches from a folder of high-resolution and low-resolution images.
    
    Args:
        hr_data_path (str): The path to the folder containing the high-resolution images.
        lr_data_path (str): The path to the folder containing the low-resolution images.
        filenames (list): A list of filenames of the images to extract patches from.
        scale_factor (float): The scale factor for downsampling the images.
        crappifier_name (str): The name of the crappifier to use for downsampling.
        lr_patch_shape (tuple): The shape of the low-resolution patches to extract.
        datagen_sampling_pdf (str): The probability density function for sampling the patches.
    
    Returns:
        final_lr_patches (numpy.ndarray): An array of extracted low-resolution patches.
        final_hr_patches (numpy.ndarray): An array of extracted high-resolution patches.
        actual_scale_factor (float): The actual scale factor used for downsampling.
    """
    
    # First lets check what is the scale factor, in case None is given
    actual_scale_factor = obtain_scale_factor(
        hr_filename=os.path.join(hr_data_path, filenames[0]),
        lr_filename=None
        if lr_data_path is None
        else os.path.join(lr_data_path, filenames[0]),
        scale_factor=scale_factor,
        crappifier_name=crappifier_name,
    )

    final_lr_patches = []
    final_hr_patches = []

    # Then for a fiven list of filenames, extract a single patch for each image
    for f in filenames:
        hr_image_path = os.path.join(hr_data_path, f)
        if lr_data_path is not None:
            lr_image_path = os.path.join(lr_data_path, f)
        else:
            lr_image_path = None
        lr_patches, hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            actual_scale_factor,
            crappifier_name,
            lr_patch_shape,
            datagen_sampling_pdf,
            verbose=verbose
        )
        final_lr_patches.append(lr_patches)
        final_hr_patches.append(hr_patches)

    final_lr_patches = np.array(final_lr_patches)
    final_hr_patches = np.array(final_hr_patches)

    return final_lr_patches, final_hr_patches, actual_scale_factor

#
#####################################