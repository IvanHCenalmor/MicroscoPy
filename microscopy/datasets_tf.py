import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .datasets_utils import extract_random_patches_from_image, extract_random_patches_from_folder

#####################################
#
# Functions to define a TensorFlow datasets with its generator.

class TFDataGenerator:
    def __init__(
        self,
        filenames,
        hr_data_path,
        lr_data_path,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        datagen_sampling_pdf,
        validation_split,
        verbose
    ):
        self.filenames = np.array(filenames)
        self.indexes = np.arange(len(self.filenames))

        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.validation_split = validation_split      
    
        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_factor = actual_scale_factor
        self.verbose = verbose

    def __len__(self):
        """
        Returns the length of the object.
        Which will be used for the number of images on each epoch (not the batches).

        :return: int
            The length of the object.
        """
        return int(len(self.filenames))

    def __getitem__(self, idx):
        """
        Retrieves a pair of low-resolution and high-resolution image patches from the dataset.

        Parameters:
            idx (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the low-resolution and high-resolution image patches.
                - lr_patches (ndarray): A 4D numpy array of low-resolution image patches.
                - hr_patches (ndarray): A 4D numpy array of high-resolution image patches.
        """
        hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
        if self.lr_data_path is not None:
            lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
        else:
            lr_image_path = None

        if self.verbose > 3:
            print('Extracting patches for image {}'.format(os.path.join(self.hr_data_path, self.filenames[idx])))

        aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            self.actual_scale_factor,
            self.crappifier_name,
            self.lr_patch_shape,
            self.datagen_sampling_pdf,
            verbose=self.verbose
        )

        lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
        hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

        return lr_patches, hr_patches

    def __call__(self):
        """
        Calls the object as a function.

        Yields each item in the object by iterating over it.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def prerpoc_func(x, y, rotation, horizontal_flip, vertical_flip):
    """
    Applies random preprocessing transformations to the input images.

    Args:
        x (Tensor): The input image tensor.
        y (Tensor): The target image tensor.
        rotation (bool): Whether to apply rotation.
        horizontal_flip (bool): Whether to apply horizontal flip.
        vertical_flip (bool): Whether to apply vertical flip.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed input and target image tensors.
    """
    apply_rotation = (tf.random.uniform(shape=[]) < 0.5) and rotation
    apply_horizontal_flip = (tf.random.uniform(shape=[]) < 0.5) and horizontal_flip
    apply_vertical_flip = (tf.random.uniform(shape=[]) < 0.5) and vertical_flip

    if apply_rotation:
        rotation_times = np.random.randint(0, 5)
        x = tf.image.rot90(x, rotation_times)
        y = tf.image.rot90(y, rotation_times)
    if apply_horizontal_flip:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if apply_vertical_flip:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    return x, y


def TFDataset(
    filenames,
    hr_data_path,
    lr_data_path,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    validation_split,
    batch_size,
    rotation,
    horizontal_flip,
    vertical_flip,
    verbose
):
    """
    Generate a TensorFlow Dataset for training and validation.

    Args:
        filenames (list): List of filenames to be used for generating the dataset.
        hr_data_path (str): Path to the high-resolution data directory.
        lr_data_path (str): Path to the low-resolution data directory.
        scale_factor (int): Scale factor for upsampling the low-resolution data.
        crappifier_name (str): Name of the crappifier to be used for generating low-resolution data.
        lr_patch_shape (tuple): Shape of the low-resolution patches.
        datagen_sampling_pdf (str): Path to the sampling PDF file for data generation.
        validation_split (float): Proportion of data to be used for validation.
        batch_size (int): Number of samples per batch.
        rotation (bool): Whether to apply random rotations to the data.
        horizontal_flip (bool): Whether to apply random horizontal flips to the data.
        vertical_flip (bool): Whether to apply random vertical flips to the data.

    Returns:
        tuple: A tuple containing the following elements:
            - dataset (tf.data.Dataset): The generated TensorFlow Dataset.
            - lr_shape (tuple): Shape of the low-resolution data.
            - hr_shape (tuple): Shape of the high-resolution data.
            - actual_scale_factor (float): The actual scale factor used for upsampling.
    """
    data_generator = TFDataGenerator(
        filenames=filenames,
        hr_data_path=hr_data_path,
        lr_data_path=lr_data_path,
        scale_factor=scale_factor,
        crappifier_name=crappifier_name,
        lr_patch_shape=lr_patch_shape,
        datagen_sampling_pdf=datagen_sampling_pdf,
        validation_split=validation_split,
        verbose=verbose
    )

    # Get the first item to extract information from it
    lr, hr = data_generator.__getitem__(0)
    actual_scale_factor = data_generator.actual_scale_factor

    # Create the dataset generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(lr.dtype, hr.dtype),
        output_shapes=(tf.TensorShape(lr.shape), tf.TensorShape(hr.shape)),
    )

    # Map the preprocessing function
    dataset = dataset.map(
        lambda x, y: prerpoc_func(x, y, rotation, horizontal_flip, vertical_flip),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Batch the data
    dataset = dataset.batch(batch_size)

    return (
        dataset,
        (data_generator.__len__(),) + (lr.shape),
        (data_generator.__len__(),) + (hr.shape),
        actual_scale_factor,
    )

#
#####################################

#####################################
#
# Functions to define a different Tensorflow Data generator which is based on Sequence.

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        filenames,
        hr_data_path,
        lr_data_path,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        datagen_sampling_pdf,
        validation_split,
        batch_size,
        rotation,
        horizontal_flip,
        vertical_flip,
        shuffle=True,
    ):
        """
        Suffle is used to take everytime a different
        sample from the list in a random way so the
        training order differs. We create two instances
        with the same arguments.
        """
        self.filenames = np.array(filenames)
        self.indexes = np.arange(len(self.filenames))

        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.validation_split = validation_split
        self.batch_size = batch_size

        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        # Make an initial shuffle
        self.shuffle = shuffle
        self.on_epoch_end()

        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_facto

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int: The number of batches per epoch.
        """
        return int(np.floor(len(self.filenames) / self.batch_size))

    def get_sample(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the x and y values of the sample, as well as the actual scale factor.
        """
        x, y = self.__getitem__(idx)

        return x, y, self.actual_scale_factor

    def on_epoch_end(self):
        """
        Perform actions at the end of each epoch.

        This method is called at the end of each epoch in the training process.
        It updates the `indexes` attribute by creating a numpy array of indices
        corresponding to the length of the `filenames` attribute. If `shuffle`
        is set to `True`, it shuffles the indices using the `np.random.shuffle`
        function.
        """
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Returns a tuple of low-resolution patches and high-resolution patches corresponding to the given index.

        Parameters:
            index (int): The index of the batch.
        
        Returns:
            tuple: A tuple containing the low-resolution patches and high-resolution patches.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in indexes]
        # Generate data
        lr_patches, hr_patches = self.__data_generation(list_IDs_temp)
        return lr_patches, hr_patches

    def __call__(self):
        """
        Calls the object as a function.

        Yields each item in the object by iterating over it.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __preprocess(self, x, y):
        """
        Preprocesses the input data by applying random rotations, horizontal flips, and vertical flips.

        Parameters:
            x (ndarray): The input data to be preprocessed.
            y (ndarray): The target data to be preprocessed.

        Returns:
            processed_x (ndarray): The preprocessed input data.
            processed_y (ndarray): The preprocessed target data.
        """
        apply_rotation = (np.random.random() < 0.5) * self.rotation
        apply_horizontal_flip = (np.random.random() < 0.5) * self.horizontal_flip
        apply_vertical_flip = (np.random.random() < 0.5) * self.vertical_flip

        processed_x = np.copy(x)
        processed_y = np.copy(y)

        if apply_rotation:
            rotation_times = np.random.randint(0, 5)
            processed_x = np.rot90(processed_x, rotation_times, axes=(1, 2))
            processed_y = np.rot90(processed_y, rotation_times, axes=(1, 2))
        if apply_horizontal_flip:
            processed_x = np.flip(processed_x, axis=2)
            processed_y = np.flip(processed_y, axis=2)
        if apply_vertical_flip:
            processed_x = np.flip(processed_x, axis=1)
            processed_y = np.flip(processed_y, axis=1)

        return processed_x, processed_y

    def __data_generation(self, list_IDs_temp):
        """
        Generate data batches for training or validation.

        Parameters:
            list_IDs_temp (list): The list of data IDs to generate the data for.

        Returns:
            lr_patches (ndarray): The low-resolution image patches generated from the data.
            hr_patches (ndarray): The high-resolution image patches generated from the data.
        """

        final_lr_patches = []
        final_hr_patches = []

        for idx in list_IDs_temp:
            hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
            if self.lr_data_path is not None:
                lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
            else:
                lr_image_path = None

            aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
                hr_image_path,
                lr_image_path,
                self.actual_scale_factor,
                self.crappifier_name,
                self.lr_patch_shape,
                self.datagen_sampling_pdf,
            )

            lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
            hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

            lr_patches, hr_patches = self.__preprocess(lr_patches, hr_patches)

            final_lr_patches.append(lr_patches)
            final_hr_patches.append(hr_patches)


        return np.concatenate(final_lr_patches, axis=0), np.concatenate(final_hr_patches, axis=0) # lr_patches, hr_patches

#
#####################################

#####################################
#
# Functions that will be used to define and create an old version of the TensorFlow dataset.
# This version would load the complete dataset on memory and only once, therefore
# it would have the same images for each epoch. Even if is may be faster, 

from skimage import transform

def random_90rotation( img ):
    """
    Rotate an image randomly by 90 degrees.

    Parameters:
        img (array-like): The image to be rotated.

    Returns:
        array-like: The rotated image.
    """
    return transform.rotate(img, 90*np.random.randint( 0, 5 ), preserve_range=True)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators(X_data, Y_data, batch_size=32, seed=42, show_examples=False):
    """
    Generate train and validation data generators using image data augmentation.

    :param X_data: The input data for training.
    :param Y_data: The target data for training.
    :param batch_size: The batch size used for training. Default is 32.
    :param seed: The seed used for random number generation. Default is 42.
    :param show_examples: Whether to show examples of augmented images. Default is False.
    :return: The train generator that yields augmented image and target data.
    """

    # Image data generator distortion options
    data_gen_args = dict( #rotation_range = 45,
                          #width_shift_range=0.2,
                          #height_shift_range=0.2,
                          #shear_range=0.2,
                          #brightness_range=[1., 1.],
                          #rescale=1./255,
                          preprocessing_function=random_90rotation,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect')


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_data, augment=True, seed=seed)
    Y_datagen.fit(Y_data, augment=True, seed=seed)
    X_data_augmented = X_datagen.flow(X_data, batch_size=batch_size, shuffle=True, seed=seed)
    Y_data_augmented = Y_datagen.flow(Y_data, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_data_augmented, Y_data_augmented)

    return train_generator

#
#####################################