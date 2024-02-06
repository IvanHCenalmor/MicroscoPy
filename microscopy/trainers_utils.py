import numpy as np
import os

from . import datasets_utils
from . import utils
from . import metrics

class ModelsTrainer:
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
        data_on_memory=0,
    ):
        self.data_name = config.dataset_name

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        train_extension_list = [
            os.path.splitext(e)[1] for e in os.listdir(self.train_hr_path)
        ]
        train_extension = max(set(train_extension_list), key=train_extension_list.count)
        self.train_filenames = sorted(
            [x for x in os.listdir(self.train_hr_path) if x.endswith(train_extension)]
        )

        self.validation_split = config.hyperparam.validation_split
        if val_hr_path is None or val_lr_path is None:
            self.val_lr_path = train_lr_path
            self.val_hr_path = train_hr_path

            self.val_filenames = self.train_filenames[
                int(len(self.train_filenames) * (1 - self.validation_split )) :
            ]
            self.train_filenames = self.train_filenames[
                : int(len(self.train_filenames) * (1 - self.validation_split))
            ]
        else:
            self.val_lr_path = val_lr_path
            self.val_hr_path = val_hr_path

            val_extension_list = [
                os.path.splitext(e)[1] for e in os.listdir(self.val_hr_path)
            ]
            val_extension = max(set(val_extension_list), key=val_extension_list.count)
            self.val_filenames = sorted(
                [x for x in os.listdir(self.val_hr_path) if x.endswith(val_extension)]
            )

        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        test_extension_list = [
            os.path.splitext(e)[1] for e in os.listdir(self.test_hr_path)
        ]
        test_extension = max(set(test_extension_list), key=test_extension_list.count)
        self.test_filenames = sorted(
            [x for x in os.listdir(self.test_hr_path) if x.endswith(test_extension)]
        )

        self.crappifier_method = config.used_dataset.crappifier
        self.scale_factor = config.used_dataset.scale
        self.lr_patch_size_x = config.used_dataset.patch_size_x
        self.lr_patch_size_y = config.used_dataset.patch_size_y
        self.datagen_sampling_pdf = config.hyperparam.datagen_sampling_pdf

        if "rotation" in config.hyperparam.data_augmentation:
            self.rotation = True
        if "horizontal_flip" in config.hyperparam.data_augmentation:
            self.horizontal_flip = True
        if "vertical_flip" in config.hyperparam.data_augmentation:
            self.vertical_flip = True
        if len(config.hyperparam.data_augmentation) != 0 and (
            not self.rotation or not self.horizontal_flip or not self.vertical_flip
        ):
            raise ValueError("Data augmentation values are not well defined.")

        self.model_name = config.model_name
        self.num_epochs = config.hyperparam.num_epochs
        self.batch_size = config.hyperparam.batch_size
        self.learning_rate = config.hyperparam.lr
        self.discriminator_learning_rate = config.hyperparam.discriminator_lr
        self.optimizer_name = config.hyperparam.optimizer
        self.discriminator_optimizer = config.hyperparam.discriminator_optimizer
        self.lr_scheduler_name = config.hyperparam.scheduler
        self.discriminator_lr_scheduler = config.hyperparam.discriminator_lr_scheduler

        self.test_metric_indexes = config.hyperparam.test_metric_indexes

        self.additional_folder = config.hyperparam.additional_folder
        self.seed = config.hyperparam.seed

        self.verbose = verbose
        self.data_on_memory = data_on_memory

        save_folder = "scale" + str(self.scale_factor)

        if self.additional_folder:
            save_folder += "_" + self.additional_folder

        self.saving_path = saving_path
        self.config = config

        os.makedirs(self.saving_path, exist_ok=True)
        utils.save_yaml(
            self.config,
            os.path.join(self.saving_path, "train_configuration.yaml"),
        )

        utils.set_seed(self.seed)
        # To calculate the input and output shape and the actual scale factor 
        # (
        #     _,
        #     train_input_shape,
        #     train_output_shape,
        #     actual_scale_factor,
        # ) = datasets_utils.TFDataset(
        #     filenames=self.train_filenames,
        #     hr_data_path=self.train_hr_path,
        #     lr_data_path=self.train_lr_path,
        #     scale_factor=self.scale_factor,
        #     crappifier_name=self.crappifier_method,
        #     lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
        #     datagen_sampling_pdf=self.datagen_sampling_pdf,
        #     validation_split=0.0,
        #     batch_size=self.batch_size,
        #     rotation=self.rotation,
        #     horizontal_flip=self.horizontal_flip,
        #     vertical_flip=self.vertical_flip,
        #     verbose=self.verbose
        # )

        train_input_shape = [64,64]
        train_output_shape = [64,64]
        actual_scale_factor = 2

        self.input_data_shape = train_input_shape
        self.output_data_shape = train_output_shape

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


        print("\n" + "-" * 10)
        print(
            "{} model will be trained with the next configuration".format(
                self.model_name
            )
        )
        print("Dataset: {}".format(self.data_name))
        print("\tTrain wf path: {}".format(train_lr_path))
        print("\tTrain gt path: {}".format(train_hr_path))
        print("\tVal wf path: {}".format(val_lr_path))
        print("\tVal gt path: {}".format(val_hr_path))
        print("\tTest wf path: {}".format(test_lr_path))
        print("\tTest gt path: {}".format(test_hr_path))
        print("Preprocessing info:")
        print("\tScale factor: {}".format(self.seed))
        print("\tScale factor: {}".format(self.scale_factor))
        print("\tCrappifier method: {}".format(self.crappifier_method))
        print("\tPatch size: {} x {}".format(self.lr_patch_size_x, self.lr_patch_size_y))
        print("Training info:")
        print("\tEpochs: {}".format(self.num_epochs))
        print("\tBatchsize: {}".format(self.batch_size))
        print("\tGen learning rate: {}".format(self.learning_rate))
        print("\tDisc learning rate: {}".format(self.discriminator_learning_rate))
        print("\tGen optimizer: {}".format(self.optimizer_name))
        print("\tDisc optimizer: {}".format(self.discriminator_optimizer))
        print("\tGen scheduler: {}".format(self.lr_scheduler_name))
        print("\tDisc scheduler: {}".format(self.discriminator_lr_scheduler))
        print("-" * 10)

    def prepare_data(self):
        raise NotImplementedError("prepare_data() not implemented.")

    def train_model(self):
        raise NotImplementedError("train_model() not implemented.")

    def predict_images(self, result_folder_name=""):
        raise NotImplementedError("predict_images() not implemented")

    def eval_model(self, result_folder_name=""):
        utils.set_seed(self.seed)

        if self.verbose  > 0:
            utils.print_info("eval_model() - self.Y_test", self.Y_test)
            utils.print_info("eval_model() - self.predictions", self.predictions)
            utils.print_info("eval_model() - self.X_test", self.X_test)

        print("The predictions will be evaluated:")
        metrics_dict = self.test_metrics 
        
        os.makedirs(os.path.join(self.saving_path, "test_metrics", result_folder_name), exist_ok=True)

        for key in metrics_dict.keys():
            if len(metrics_dict[key]) > 0:
                np.save(
                    os.path.join(self.saving_path, "test_metrics", result_folder_name, f"{key}.npy"),
                    metrics_dict[key],
                )