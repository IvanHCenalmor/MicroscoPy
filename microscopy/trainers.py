PYTORCH_MODELS = ["wgan", "esrganplus", "srgan"]
TENSORFLOW_MODELS = ["rcan", "dfcan", "wdsr", "unet", "cddpm"]

def get_model_trainer(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):
    if config.model_name in PYTORCH_MODELS:
        from .trainers_torch import PytorchTrainer

        model_trainer = PytorchTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path,  val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    elif config.model_name in TENSORFLOW_MODELS:
        from .trainers_tf import TensorflowTrainer

        model_trainer = TensorflowTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path, val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    else:
        raise Exception("Not available model.")

    return model_trainer

def train_configuration(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):
    model_trainer = get_model_trainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path, val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )

        
    model_trainer.prepare_data()
    model_trainer.train_model()

    model_trainer.predict_images()
    model_trainer.eval_model()

    return model_trainer.history


def predict_configuration(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):
    model_trainer = get_model_trainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path, val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )

    model_trainer.prepare_data()
    model_trainer.predict_images()
    model_trainer.eval_model()