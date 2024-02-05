import numpy as np
import time
import os

from skimage import io

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from . import datasets_tf
from . import datasets_utils
from . import utils
from . import model_utils
from . import optimizer_scheduler_utils
from . import custom_callbacks_tf

from .tf_losses import ssim_loss
from .trainers_utils import ModelsTrainer

class TensorflowTrainer(ModelsTrainer):
    def __init__(
        self,
        config,
        train_lr_path,
        train_hr_path,
        val_lr_path,
        val_hr_path,
        test_lr_path,
        test_hr_path,
        saving_path,
        verbose=0,
        data_on_memory=0
    ):
        super().__init__(
            config,
            train_lr_path,
            train_hr_path,
            val_lr_path,
            val_hr_path,
            test_lr_path,
            test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory
        )

        tf.config.run_functions_eagerly(False)

        self.library_name = "tensorflow"

    def prepare_data(self):
        utils.set_seed(self.seed)
        if self.data_on_memory:
            if self.verbose > 0:
                print('Data will be loaded on memory for all the epochs the same.')
            X_train, Y_train, actual_scale_factor = datasets_utils.extract_random_patches_from_folder( 
                                                        lr_data_path = self.train_lr_path,
                                                        hr_data_path = self.train_hr_path,
                                                        filenames = self.train_filenames,
                                                        scale_factor = self.scale_factor,
                                                        crappifier_name = self.crappifier_method,
                                                        lr_patch_shape = (self.lr_patch_size_x, self.lr_patch_size_y),
                                                        datagen_sampling_pdf = self.datagen_sampling_pdf)
            X_train = np.expand_dims(X_train, axis=-1)
            Y_train = np.expand_dims(Y_train, axis=-1)

            self.input_data_shape = X_train.shape
            self.output_data_shape = Y_train.shape

            train_generator = datasets_utils.get_train_val_generators(X_data=X_train,
                                                                Y_data=Y_train,
                                                                batch_size=self.batch_size)


            X_val, Y_val, _ = datasets_utils.extract_random_patches_from_folder( 
                                                        lr_data_path = self.val_lr_path,
                                                        hr_data_path = self.val_hr_path,
                                                        filenames = self.val_filenames,
                                                        scale_factor = self.scale_factor,
                                                        crappifier_name = self.crappifier_method,
                                                        lr_patch_shape = (self.lr_patch_size_x, self.lr_patch_size_y),
                                                        datagen_sampling_pdf = self.datagen_sampling_pdf)
            X_val = np.expand_dims(X_val, axis=-1)
            Y_val = np.expand_dims(Y_val, axis=-1)

            self.val_input_data_shape = X_val.shape
            self.val_output_data_shape = Y_val.shape

            val_generator = datasets_utils.get_train_val_generators(X_data=X_val,
                                                             Y_data=Y_val,
                                                             batch_size=self.batch_size)
        else:
            print('Data will be loaded on the fly, each batch new data will be loaded..')

            utils.set_seed(self.seed)

            train_generator, train_input_shape,train_output_shape, actual_scale_factor = datasets_tf.TFDataset(
                filenames=self.train_filenames,
                hr_data_path=self.train_hr_path,
                lr_data_path=self.train_lr_path,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                validation_split=0.1,
                batch_size=self.batch_size,
                rotation=self.rotation,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                verbose=self.verbose
            )
            
            self.input_data_shape = train_input_shape
            self.output_data_shape = train_output_shape

            # training_images_path = os.path.join(self.saving_path, f'special_folder_{actual_scale_factor}')
            # os.makedirs(training_images_path, exist_ok=True)
            # cont = 0
            # for lr_img, hr_img in train_generator:
            #     for i in range(hr_img.shape[0]):
            #         io.imsave(os.path.join(training_images_path, "hr" + str(cont) + ".tif"), np.array(hr_img[i,...]))
            #         io.imsave(os.path.join(training_images_path, "lr" + str(cont) + ".tif"), np.array(lr_img[i,...]))
            #         if cont > 100:
            #             break
            #         cont += 1
            #     if cont > 100:
            #         break

            val_generator, _, _, _ = datasets_tf.TFDataset(
                filenames=self.val_filenames,
                hr_data_path=self.val_hr_path,
                lr_data_path=self.val_lr_path,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                validation_split=0.1,
                batch_size=self.batch_size,
                rotation=self.rotation,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                verbose=self.verbose
            )

        if self.verbose > 0:
            print("input_data_shape: {}".format(self.input_data_shape))
            print("output_data_shape: {}".format(self.output_data_shape))

        if self.scale_factor is None or self.scale_factor != actual_scale_factor:
            self.scale_factor = actual_scale_factor
            utils.update_yaml(
                os.path.join(self.saving_path, "train_configuration.yaml"),
                "actual_scale_factor",
                actual_scale_factor,
            )
            if self.verbose > 0:
                print(
                    "Actual scale factor that will be used is: {}".format(
                        self.scale_factor
                    )
                )

        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "input_data_shape",
            self.input_data_shape,
        )
        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "output_data_shape",
            self.output_data_shape,
        )

        self.train_generator = train_generator
        self.val_generator = val_generator

    def train_model(self):

        utils.set_seed(self.seed)

        callbacks = []
        lr_schedule = optimizer_scheduler_utils.select_lr_schedule(
                    library_name=self.library_name,
                    lr_scheduler_name=self.lr_scheduler_name,
                    data_len=self.input_data_shape[0] // self.batch_size,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    monitor_loss='val_ssim_loss',
                    name=None,
                    optimizer=None,
                    frequency=None,
                    additional_configuration=self.config,
                    verbose=self.verbose
        )

        if self.lr_scheduler_name in ["CosineDecay", "MultiStepScheduler"]:
            self.optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=lr_schedule,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
                verbose=self.verbose
            )
        else:
            self.optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
                verbose=self.verbose
            )
            if not lr_schedule is None:
                callbacks.append(lr_schedule)

        model = model_utils.select_model(
            model_name=self.model_name,
            input_shape=self.input_data_shape,
            output_channels=self.output_data_shape[-1],
            scale_factor=self.scale_factor,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            model_configuration=self.config.used_model,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        loss_funct = tf.keras.losses.mean_absolute_error
        eval_metric = tf.keras.losses.mean_squared_error

        model.compile(
            optimizer=self.optim,
            loss=loss_funct,
            metrics=[eval_metric, ssim_loss],
        )

        trainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.trainable_weights]
        )
        nonTrainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.non_trainable_weights]
        )
        totalParams = trainableParams + nonTrainableParams

        model_checkpoint = tf_ModelCheckpoint(
            os.path.join(self.saving_path, "weights_best.h5"),
            monitor="val_loss" if self.model_name!="cddpm" else "val_n_loss",
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(model_checkpoint)

        # callback for early stopping
        earlystopper = EarlyStopping(
            monitor=self.config.model.optim.early_stop.loss,
            patience=self.config.model.optim.early_stop.patience,
            min_delta=0.005,
            mode=self.config.model.optim.early_stop.mode,
            verbose=self.verbose,
            restore_best_weights=True,
        )
        callbacks.append(earlystopper)

        # callback for saving the learning rate
        lr_observer = custom_callbacks_tf.LearningRateObserver()
        callbacks.append(lr_observer)

        for x, y in self.val_generator:
            x_val = x
            y_val = y
            break
        
        plt_saving_path = os.path.join(self.saving_path, "training_images")
        os.makedirs(plt_saving_path, exist_ok=True)
        plot_callback = custom_callbacks_tf.PerformancePlotCallback(
            x_val, y_val, plt_saving_path, frequency=10, is_cddpm=self.model_name=="cddpm"
        )
        callbacks.append(plot_callback)

        if self.verbose > 0:
            print("Model configuration:")
            print(f"\tModel_name: {self.model_name}")
            print(f"\tOptimizer: {self.optim}")
            print(f"\tLR scheduler: {lr_schedule}")
            print(f"\tLoss: {loss_funct}")
            print(f"\tEval: {eval_metric}")
            print(
                "Trainable parameteres: {} \nNon trainable parameters: {} \nTotal parameters: {}".format(
                    trainableParams, nonTrainableParams, totalParams
                )
            )
            # callbacks.append(custom_callbacks.CustomCallback())

        if self.model_name == "cddpm":
            # calculate mean and variance of training dataset for normalization
            model.normalizer.adapt(self.train_generator.map(lambda x, y: x))

        start = time.time()

        print("Training is going to start:")

        if self.data_on_memory:
            history = model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=self.num_epochs,
                steps_per_epoch=self.input_data_shape[0]//self.batch_size,
                validation_steps=self.val_input_data_shape[0]//self.batch_size,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=self.num_epochs,
                callbacks=callbacks,
            )

        dt = time.time() - start
        mins, sec = divmod(dt, 60)
        hour, mins = divmod(mins, 60)
        print(
            "\nTime elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)\n"
        )

        model.save_weights(os.path.join(self.saving_path, "weights_last.h5"))
        self.history = history

        os.makedirs(self.saving_path + "/train_metrics", exist_ok=True)

        for key in history.history:
            np.save(
                self.saving_path + "/train_metrics/" + key + ".npy",
                history.history[key],
            )
        np.save(self.saving_path + "/train_metrics/time.npy", np.array([dt]))
        np.save(self.saving_path + "/train_metrics/lr.npy", np.array(lr_observer.epoch_lrs))

    def predict_images(self, result_folder_name=""):

        utils.set_seed(self.seed)
        print(f'Using seed: {self.seed}')

        ground_truths = []
        widefields = []
        predictions = []
        print(f"Prediction of {self.model_name} is going to start:")

        

        for test_filename in self.test_filenames:
            
            utils.set_seed(self.seed)
            lr_images, hr_images, _ = datasets_tf.extract_random_patches_from_folder(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=[test_filename],
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                datagen_sampling_pdf=1,
            )

            hr_images = np.expand_dims(hr_images, axis=-1)
            lr_images = np.expand_dims(lr_images, axis=-1)

            io.imsave(
                os.path.join(self.saving_path, result_folder_name, "lr_img_from_dataset.tif"),
                lr_images[0]
            )
            io.imsave(
                os.path.join(self.saving_path, result_folder_name, "hr_img_from_dataset.tif"),
                hr_images[0]
            )

            ground_truths.append(hr_images[0, ...])
            widefields.append(lr_images[0, ...])
            
            if self.model_name == "unet" or self.model_name == "cddpm":
                if self.verbose > 0:
                    print("Padding will be added to the images.")
                    print("LR images before padding:")
                    print(
                        "LR images - shape:{} dtype:{}".format(
                            lr_images.shape, lr_images.dtype
                        )
                    )

                if self.model_name == "unet":
                    height_padding, width_padding = utils.calculate_pad_for_Unet(
                        lr_img_shape=lr_images[0].shape,
                        depth_Unet=self.config.used_model.depth,
                        is_pre=True,
                        scale=self.scale_factor,
                    )
                else:
                    height_padding, width_padding = utils.calculate_pad_for_Unet(
                        lr_img_shape=lr_images[0].shape,
                        depth_Unet=self.config.used_model.block_depth,
                        is_pre=True,
                        scale=self.scale_factor,
                    )
                    

                if self.verbose > 0 and (
                    height_padding == (0, 0) and width_padding == (0, 0)
                ):
                    print("No padding has been needed to be added.")

                lr_images = utils.add_padding_for_Unet(
                    lr_imgs=lr_images,
                    height_padding=height_padding,
                    width_padding=width_padding,
                )

                print(f'Before paddins: {hr_images.shape}')
                hr_images = utils.add_padding_for_Unet(
                    lr_imgs=hr_images,
                    height_padding=(height_padding[0]*self.scale_factor, height_padding[1]*self.scale_factor),
                    width_padding=(width_padding[0]*self.scale_factor, width_padding[1]*self.scale_factor),
                )
                print(f'After paddins: {hr_images.shape}')

                io.imsave(
                    os.path.join(self.saving_path, result_folder_name, "lr_img_after_padding.tif"),
                    lr_images[0]
                )
                io.imsave(
                    os.path.join(self.saving_path, result_folder_name, "hr_img_after_padding.tif"),
                    hr_images[0]
                )

            if self.verbose > 0:
                print(
                    "HR images - shape:{} dtype:{}".format(
                        hr_images.shape, hr_images.dtype
                    )
                )
                print(
                    "LR images - shape:{} dtype:{}".format(
                        lr_images.shape, lr_images.dtype
                    )
                )

            if self.config.model.others.positional_encoding:
                lr_images = utils.concatenate_encoding(
                    lr_images,
                    self.config.model.others.positional_encoding_channels,
                )

            optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
            )

            model = model_utils.select_model(
                model_name=self.model_name,
                input_shape=lr_images.shape,
                output_channels=hr_images.shape[-1],
                scale_factor=self.scale_factor,
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                model_configuration=self.config.used_model,
                verbose=self.verbose
            )

            if self.verbose > 0:
                print(model.summary())

            loss_funct = "mean_absolute_error"
            eval_metric = "mean_squared_error"

            model.compile(
                optimizer=optim, loss=loss_funct, metrics=[eval_metric, ssim_loss]
            )

            # Load old weights
            model.load_weights(os.path.join(self.saving_path, "weights_best.h5"))

            if self.model_name == "cddpm":
                aux_prediction = model.predict(lr_images, diffusion_steps=50, seed=self.seed)
            else:
                aux_prediction = model.predict(lr_images, batch_size=1)

            io.imsave(
                os.path.join(self.saving_path, result_folder_name, "aux_prediction.tif"),
                aux_prediction[0]
            )

            if self.model_name == "unet" or self.model_name == "cddpm":
                print(f'Before removing padding: {aux_prediction.shape}')
                aux_prediction = utils.remove_padding_for_Unet(
                    pad_hr_imgs=aux_prediction,
                    height_padding=height_padding,
                    width_padding=width_padding,
                    scale=self.scale_factor,
                )
                print(f'After removing padding: {aux_prediction.shape}')

            io.imsave(
                os.path.join(self.saving_path, result_folder_name, "aux_prediction_after_removing_padding.tif"),
                aux_prediction[0]
            )

            # aux_prediction = datasets.normalization(aux_prediction)
            aux_prediction = np.clip(aux_prediction, a_min=0.0, a_max=1.0)

            io.imsave(
                os.path.join(self.saving_path, result_folder_name, "aux_prediction_after_clip.tif"),
                aux_prediction[0]
            )

            if len(aux_prediction.shape) == 4:
                predictions.append(aux_prediction[0, ...])
            elif len(aux_prediction.shape) == 3:
                if aux_prediction.shape[-1] == 1:
                    predictions.append(aux_prediction)
                if aux_prediction.shape[0] == 1:
                    predictions.append(np.expand_dims(aux_prediction[0,:,:], -1))

        self.Y_test = ground_truths
        self.predictions = predictions
        self.X_test = widefields

        # assert (np.max(self.Y_test) <= 1.0).all and (np.max(self.predictions) <= 1.0).all and (np.max(self.X_test) <= 1.0).all
        # assert (np.min(self.Y_test) >= 0.0).all and (np.min(self.predictions) >= 0.0).all and (np.min(self.X_test) >= 0.0).all

        if self.verbose > 0:
            utils.print_info("predict_images() - Y_test", self.Y_test)
            utils.print_info("predict_images() - predictions", self.predictions)
            utils.print_info("predict_images() - X_test", self.X_test)

        # Save the predictions
        os.makedirs(os.path.join(self.saving_path, "predicted_images", result_folder_name), exist_ok=True)

        for i, image in enumerate(predictions):

            print(image.shape)

            io.imsave(
                os.path.join(self.saving_path, "predicted_images", result_folder_name, os.path.splitext(self.test_filenames[i])[0] + '.tif'),
                np.squeeze(image)
            )
        print(
            "Predicted images have been saved in: "
            + self.saving_path
            + "/predicted_images"
        )