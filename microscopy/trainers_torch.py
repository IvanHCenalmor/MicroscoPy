import numpy as np
import time
import csv
import os

from skimage import io

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from . import datasets_torch
from . import datasets_utils
from . import utils
from . import model_utils

from .trainers_utils import ModelsTrainer

#######

class PytorchTrainer(ModelsTrainer):
    def __init__(
        self,
        data_name,
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
        super().__init__(
            data_name,
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

        self.library_name = "pytorch"

    def prepare_data(self):
        
        self.data_module = datasets_torch.PytorchDataModuler(
            lr_patch_size_x=self.lr_patch_size_x,
            lr_patch_size_y=self.lr_patch_size_y,
            batch_size=self.batch_size,
            scale_factor=self.scale_factor,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            train_hr_path=self.train_hr_path,
            train_lr_path=self.train_lr_path,
            train_filenames=self.train_filenames,
            val_hr_path=self.val_hr_path,
            val_lr_path=self.val_lr_path,
            val_filenames=self.val_filenames,
            test_hr_path=self.test_hr_path,
            test_lr_path=self.test_lr_path,
            test_filenames=self.test_filenames,
            crappifier_method=self.crappifier_method,
            verbose=self.verbose,
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

    def train_model(self):
        utils.set_seed(self.seed)

        self.data_len = self.input_data_shape[0] // self.batch_size + int(self.input_data_shape[0] % self.batch_size != 0)

        model = model_utils.select_model(
            model_name=self.model_name,
            input_shape=None,
            output_channels=None,
            scale_factor=self.scale_factor,
            batch_size=self.batch_size,
            data_len=self.data_len,
            lr_patch_size_x=self.lr_patch_size_x,
            lr_patch_size_y=self.lr_patch_size_y,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            learning_rate_g=self.learning_rate,
            learning_rate_d=self.discriminator_learning_rate,
            g_optimizer=self.optimizer_name,
            d_optimizer=self.discriminator_optimizer,
            g_scheduler=self.lr_scheduler_name,
            d_scheduler=self.discriminator_lr_scheduler,
            epochs=self.num_epochs,
            save_basedir=self.saving_path,
            train_hr_path=self.train_hr_path,
            train_lr_path=self.train_lr_path,
            train_filenames=self.train_filenames,
            val_hr_path=self.val_hr_path,
            val_lr_path=self.val_lr_path,
            val_filenames=self.val_filenames,
            crappifier_method=self.crappifier_method,
            model_configuration=self.config,
            verbose=self.verbose,
        )
        
        # Let's define the callbacks that will be used during training
        callbacks = [] 
        
        # First to monitor the LR en each epoch (for validating the scheduler and the optimizer)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Saving checkpoints during training based
        checkpoints = ModelCheckpoint(
            monitor="val_ssim",
            mode="max",
            save_top_k=1,
            every_n_train_steps=5,
            save_last=True,
            filename="{epoch:02d}-{val_ssim:.3f}",
        )
        callbacks.append(checkpoints)

        os.makedirs(self.saving_path + "/Quality Control", exist_ok=True)
        logger = CSVLogger(self.saving_path + "/Quality Control", name="Logger")

        trainer = L.Trainer(
            accelerator="gpu", 
            devices=-1,
            strategy="ddp_find_unused_parameters_true",
            max_epochs=self.num_epochs,
            logger=logger,
            callbacks=callbacks,
        )

        print('\n')
        print(trainer.accelerator)
        print(trainer.strategy)
        print('\n')
        
        print("Training is going to start:")
        start = time.time()

        trainer.fit(model, datamodule=self.data_module)

        # Displaying the time elapsed for training
        dt = time.time() - start
        mins, sec = divmod(dt, 60)
        hour, mins = divmod(mins, 60)
        print(
            "\nTime elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)\n"
        )

        logger_path = os.path.join(self.saving_path + "/Quality Control/Logger")
        all_logger_versions = [
            os.path.join(logger_path, dname) for dname in os.listdir(logger_path)
        ]
        last_logger = all_logger_versions[-1]

        train_csv_path = last_logger + "/metrics.csv"

        if not os.path.exists(train_csv_path):
            print(
                "The path does not contain a csv file containing the loss and validation evolution of the model"
            )
        else:
            with open(train_csv_path, "r") as csvfile:
                csvRead = csv.reader(csvfile, delimiter=",")
                keys = next(csvRead)
                step_idx = keys.index("step")
                keys.remove("step")

                # Initialize the dictionary with empty lists
                train_metrics = {}
                for k in keys:
                    train_metrics[k] = []

                # Fill the dictionary
                for row in csvRead:
                    step = int(row[step_idx])
                    row.pop(step_idx)
                    for i, row_value in enumerate(row):
                        if row_value:
                            train_metrics[keys[i]].append([step, float(row_value)])

                os.makedirs(self.saving_path + "/train_metrics", exist_ok=True)

                # Save the metrics
                for key in train_metrics:
                    values_to_save = np.array([e[1] for e in train_metrics[key]])
                    np.save(
                        self.saving_path + "/train_metrics/" + key + ".npy",
                        values_to_save,
                    )
                np.save(self.saving_path + "/train_metrics/time.npy", np.array([dt]))

        self.history = []
        print("Train information saved.")

    def predict_images(self, result_folder_name=""):
        utils.set_seed(self.seed)

        model = model_utils.select_model(
            model_name=self.model_name,
            scale_factor=self.scale_factor,
            batch_size=self.batch_size,
            save_basedir=self.saving_path,
            model_configuration=self.config,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            checkpoint=os.path.join(self.saving_path, "best_checkpoint.pth"),
            verbose=self.verbose,
            state='predict'
        )

        trainer = L.Trainer(
                            accelerator="gpu", 
                            devices=-1,
                            strategy="ddp_find_unused_parameters_true",
                        )

        print('\n')
        print(trainer.accelerator)
        print(trainer.strategy)
        print('\n')

        print("Prediction is going to start:")
        predictions = trainer.predict(model, dataloaders=self.data_module)
        print('prediction done')
        predictions = np.array(
            [
                np.expand_dims(np.squeeze(e.cpu().detach().numpy()), axis=-1)
                for e in predictions
            ]
        )

        os.makedirs(os.path.join(self.saving_path, "predicted_images"), exist_ok=True)

        for i, image in enumerate(predictions):
            io.imsave(
                self.saving_path + "/predicted_images/" + self.test_filenames[i],
                image
            )
        print(
            "Predicted images have been saved in: "
            + self.saving_path
            + "/predicted_images"
        )

        self.predictions = predictions

        lr_images, hr_images, _ = datasets_utils.extract_random_patches_from_folder(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                datagen_sampling_pdf=1,
            )

        self.Y_test = np.expand_dims(hr_images, axis=-1)
        self.X_test = np.expand_dims(lr_images, axis=-1)

        if self.verbose > 0:
            utils.print_info("predict_images() - self.Y_test", self.Y_test)
            utils.print_info("predict_images() - self.predictions", self.predictions)
            utils.print_info("predict_images() - self.X_test", self.X_test)
