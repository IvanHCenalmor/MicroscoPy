from pytorch_lightning.callbacks import Callback as pl_callback
from matplotlib import pyplot as plt
import numpy as np

from .utils import min_max_normalization as normalization

#####################################
#
# Pytorch Lightning Callbacks.


class PerformancePlotCallback_Pytorch(pl_callback):
    def __init__(self, x_test, y_test, img_saving_path, frequency=1):
        self.x_test = x_test
        self.y_test = y_test
        self.img_saving_path = img_saving_path
        self.frequency = frequency

    def on_train_epoch_end(self, trainer, pl_module):
        """
        At the end of each epoch (with a frequency) during training, an image of the plot with  
        the LR, HR and prediction is saved.
        
        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (Module): The PyTorch Lightning module.
        Returns:
            None
        """
        if pl_module.current_epoch % self.frequency == 0:
            cuda_x_test_img = self.x_test.to("cuda")
            y_pred = pl_module.forward(cuda_x_test_img).detach().cpu().numpy()

            print(f'y_pred: {y_pred.shape}')
            print(f'self.x_test: {self.x_test.shape}')
            print(f'self.x_test: {self.x_test.shape}')

            aux_y_pred = np.expand_dims(y_pred[:,0,:,:], axis=-1)
            aux_x_test = np.expand_dims(self.x_test[:,0,:,:], axis=-1)
            aux_y_test = np.expand_dims(self.y_test[:,0,:,:], axis=-1)

            print(f'aux_y_pred: {aux_y_pred.shape}')
            print(f'aux_y_test: {aux_y_test.shape}')
            print(f'aux_x_test: {aux_x_test.shape}')

            ssim = ssim_loss(aux_y_test[0], aux_y_pred[0])

            plt.switch_backend("agg")
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Input LR image")
            plt.imshow(aux_x_test[0], "gray")
            plt.subplot(1, 4, 2)
            plt.title("Ground truth")
            plt.imshow(aux_y_test[0], "gray")
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            plt.imshow(aux_y_pred[0], "gray")
            plt.subplot(1, 4, 4)
            plt.title(f"SSIM: {ssim.numpy():.3f}")
            plt.imshow(1 - normalization(aux_y_test[0] - aux_y_pred[0]), "inferno")

            plt.tight_layout()
            plt.savefig(f"{self.img_saving_path}/{pl_module.current_epoch}.png")
            plt.close()

#
#####################################
