import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

#####
#
# Function that define different losses

def ssim_loss(y_true, y_pred):
    """
    Calculates the Structural Similarity Index (SSIM) loss between two images.

    Parameters:
        y_true (tensor): The true image.
        y_pred (tensor): The predicted image.

    Returns:
        tensor: The SSIM loss.

    Example:
        >>> y_true = tf.constant([0.5, 0.8, 0.2, 0.3])
        >>> y_pred = tf.constant([0.6, 0.7, 0.3, 0.4])
        >>> ssim_loss(y_true, y_pred)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.75>
    """

    # Printing the loss
    # tf.print('\nSSIM:')
    # tf.print(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def vgg_loss(image_shape):
    """
    Generates the VGG loss function for image style transfer.

    Args:
        image_shape (tuple): The shape of the input image. Should be a tuple of (height, width).

    Returns:
        function: The VGG loss function that takes in the ground truth image and the predicted image 
        as inputs, and returns the mean squared difference between the VGG feature representations of the two images.

    Note:
        The VGG loss function is based on the VGG19 model pretrained on the ImageNet dataset. 
        It computes the mean squared difference between the VGG feature maps of the ground truth image and the predicted image. 
        The VGG19 model is frozen and not trainable during the execution of this loss function.

    Example:
        vgg_loss = vgg_loss(image_shape=(256, 256))
        loss = vgg_loss(ground_truth_image, predicted_image)
    """
    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(image_shape[0], image_shape[1], 3),
    )
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output)
    model.trainable = False

    def vgg_loss_fixed(y_true, y_pred):
        y_true_3chan = K.concatenate([y_true, y_true, y_true], axis=-1)
        y_pred_3chan = K.concatenate([y_pred, y_pred, y_pred], axis=-1)
        return K.mean(K.square(model(y_true_3chan) - model(y_pred_3chan)))

    return vgg_loss_fixed

def perceptual_loss(image_shape, percp_coef=0.1):
    """
    Returns a loss function that combines the mean absolute error loss and the VGG loss.
    
    Parameters:
        image_shape (tuple): The shape of the input images.
        percp_coef (float, optional): The coefficient for the perceptual loss. Defaults to 0.1.
        
    Returns:
        mixed_loss (function): A loss function that combines the mean absolute error loss and the perceptual loss.
            The function takes in two tensors, y_true and y_pred, and returns the sum of the mean absolute error loss
            and the product of the perceptual loss and the percp_coef.
    """

    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
    percp_loss = vgg_loss(image_shape)

    def mixed_loss(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred) + percp_coef * percp_loss(
            y_true, y_pred
        )

    return mixed_loss

#
#####################################

#####################################
#
# Function for adding embeddings to the images

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = np.stack((np.sin(sin_inp), np.cos(sin_inp)), -1)
    emb = np.reshape(emb, (*emb.shape[:-2], -1))
    return emb


def concatenate_encoding(images, channels):
    self_channels = int(2 * np.ceil(channels / 4))
    inv_freq = np.float32(
        1 / np.power(10000, np.arange(0, self_channels, 2) / np.float32(self_channels))
    )

    _, x, y, org_channels = images.shape

    pos_x = np.arange(x)
    pos_y = np.arange(y)

    sin_inp_x = np.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = np.einsum("i,j->ij", pos_y, inv_freq)

    emb_x = np.expand_dims(get_emb(sin_inp_x), 1)
    emb_y = np.expand_dims(get_emb(sin_inp_y), 0)

    emb_x = np.tile(emb_x, (1, y, 1))
    emb_y = np.tile(emb_y, (x, 1, 1))
    emb = np.concatenate((emb_x, emb_y), -1)
    cached_penc = np.repeat(emb[None, :, :, :org_channels], np.shape(images)[0], axis=0)
    return np.concatenate((images, cached_penc), -1)

#
#####################################
