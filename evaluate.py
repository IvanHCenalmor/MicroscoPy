from microscopy.trainers import train_configuration
from omegaconf import DictConfig
import hydra
import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_path(dataset_root, dataset_name, folder):
    if folder is not None:
        return os.path.join(dataset_root, dataset_name, folder)
    else:
        return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    #'LiveFActinDataset', 'EM', 'MitoTracker_small', 'F-actin', 'ER', 'MT', 'MT-SMLM_all'
    for dataset_name in ["EM"]:  
        train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = cfg.used_dataset.data_paths

        dataset_root = "datasets" if os.path.exists("datasets") else "../datasets"
        train_lr_path = load_path(dataset_root, dataset_name, train_lr)
        train_hr_path = load_path(dataset_root, dataset_name, train_hr)
        val_lr_path = load_path(dataset_root, dataset_name, val_lr)
        val_hr_path = load_path(dataset_root, dataset_name, val_hr)
        test_lr_path = load_path(dataset_root, dataset_name, test_lr)
        test_hr_path = load_path(dataset_root, dataset_name, test_hr)

        # 'unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus', 'cddpm'
        for model_name in ["unet"]: 
            for batch_size in [4]:  
                for num_epochs in [1]:                  
                    for lr, discriminator_lr in [(0.001, 0.001)]:
                        cfg.model_name = model_name
                        cfg.hyperparam.batch_size = batch_size
                        cfg.hyperparam.num_epochs = num_epochs
                        cfg.hyperparam.lr = lr
                        cfg.hyperparam.discriminator_lr = discriminator_lr

                        save_folder = "scale" + str(cfg.used_dataset.scale)
                        if cfg.hyperparam.additional_folder is not None:
                            save_folder += "_" + cfg.hyperparam.additional_folder

                        saving_path = "./results/{}/{}/{}/epc{}_btch{}_lr{}_optim-{}_lrsched-{}_seed{}".format(
                            cfg.dataset_name,
                            cfg.model_name,
                            save_folder,
                            cfg.hyperparam.num_epochs,
                            cfg.hyperparam.batch_size,
                            cfg.hyperparam.lr,
                            cfg.hyperparam.optimizer,
                            cfg.hyperparam.scheduler,
                            cfg.hyperparam.seed
                        )

                        test_metric_path = os.path.join(saving_path, "test_metrics")
                        if (
                            os.path.exists(test_metric_path)
                            and len(os.listdir(test_metric_path)) > 0
                        ):
                            print(f"{saving_path} - model combination already trained.")
                        else:
                            try:
                                model = train_configuration(
                                    config=cfg,
                                    train_lr_path=train_lr_path,
                                    train_hr_path=train_hr_path,
                                    val_lr_path=val_lr_path,
                                    val_hr_path=val_hr_path,
                                    test_lr_path=test_lr_path,
                                    test_hr_path=test_hr_path,
                                    saving_path=saving_path,
                                    verbose=0
                                )
                                del model

                            except Exception as e:
                                print(
                                    f"\033[91mERROR\033[0m - In config {cfg.dataset_name} {cfg.model_name} {cfg.hyperparam.num_epochs} {cfg.hyperparam.lr}"
                                )
                                print(e)
                            gc.collect()
my_app()