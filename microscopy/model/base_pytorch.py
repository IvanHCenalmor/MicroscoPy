
import os
import numpy as np
import torch
import lightning as L
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from ..optimizer_scheduler_utils import select_optimizer, select_lr_schedule

class BaseModel(L.LightningModule):
    def __init__(
        self,
        data_len: int = 8,
        epochs: int = 151,
        scale_factor: int = 2,
        learning_rate_d: float = 0.0001,
        learning_rate_g: float = 0.0001,
        g_optimizer: str = None,
        d_optimizer: str = None,
        g_scheduler: str = None,
        d_scheduler: str = None,
        save_basedir: str = None,
        gen_checkpoint: str = None,
        additonal_configuration: dict = {},
        verbose: int = 0,
        k
    ):
        super(BaseModel, self).__init__()
        self.save_hyperparameters()

        print('self.hparams.verbose: {}'.format(self.hparams.verbose))

        if self.hparams.verbose > 0:
            print('\nVerbose: Model initialized (begining)\n')

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Free cuda memory
        torch.cuda.empty_cache()
    

        # Initialize generator and load the checkpoint in case is given
        self.generator = self.load_generator()
        self.best_valid_loss = self.load_best_loss()
        
        self.discriminator = self.load_discriminator()

        self.load_losses()

        self.data_len = self.hparams.data_len

        self.val_g_loss = []
        self.val_ssim = []

        if self.hparams.verbose > 0:
            print(
                "Generators parameters: {}".format(
                    sum(p.numel() for p in self.generator.parameters())
                )
            )
            print(
                "Discriminators parameters: {}".format(
                    sum(p.numel() for p in self.discriminator.parameters())
                )
            )
            print(
                "self.netF parameters: {}".format(
                    sum(p.numel() for p in self.netF.parameters())
                )
            )
        
        if self.hparams.verbose > 0:
            os.makedirs(f"{self.hparams.save_basedir}/training_images", exist_ok=True)


        self.step_schedulers = ['CosineDecay', 'OneCycle', 'MultiStepScheduler']
        self.epoch_schedulers = ['ReduceOnPlateau']

        if self.hparams.verbose > 0:
            print('\nVerbose: Model initialized (end)\n')

    def forward(self, x):
        if isinstance(x, dict):
            return self.generator(x["lr"])
        else:
            return self.generator(x)

    def training_step(self, batch, batch_idx):

        if self.hparams.verbose > 1:
            print('\nVerbose: Training step (begining)\n')

        lr, hr = batch["lr"], batch["hr"]

        # Extract the optimizers
        g_opt, d_opt = self.optimizers()

        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass

        # The generator is updated every self.hparams.n_critic_steps
        if (batch_idx + 1) % self.hparams.n_critic_steps == 0:
            if self.hparams.verbose > 0:
                print(f'Generator updated on step {batch_idx + 1}')

            # Optimize generator
            # toggle_optimizer(): Makes sure only the gradients of the current optimizer's parameters are calculated
            #                     in the training step to prevent dangling gradients in multiple-optimizer setup.
            self.toggle_optimizer(g_opt)

            final_g_loss = self.generator_step(lr, hr)

            # Optimize generator
            self.manual_backward(final_g_loss)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)
            
            if self.hparams.g_scheduler in self.step_schedulers:
                sched_g.step()

        # The discriminator is updated every step
        if (batch_idx + 1) % 1 == 0:
            if self.self.hparams.verbose:
                print(f'Discriminator  updated on step {batch_idx + 1}')
            # Optimize discriminator
            self.toggle_optimizer(d_opt)

            final_d_loss = self.discriminator_step(lr, hr)
 
            # Optimize discriminator/critic
            self.manual_backward(final_d_loss)
            d_opt.step()
            d_opt.zero_grad()
            self.untoggle_optimizer(d_opt)

            if self.hparams.d_scheduler in self.step_schedulers:
                sched_d.step()

        if self.verbose > 0:
            print('\nVerbose: Training step (end)\n')

    def validation_step(self, batch, batch_idx):
        
        if self.verbose > 0:
            print('\nVerbose: validation_step (begining)\n')
            
        # Right now used for just plotting, might want to change it later
        lr, hr = batch["lr"], batch["hr"]
        fake_hr = self.generator(lr)

        if self.verbose > 0:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
            print(f'generated.shape: {fake_hr.shape} generated.min: {fake_hr.min()} generated.max: {fake_hr.max()}')

        # Calculate the loss for the generator
        val_g_loss = self.val_generator_step(lr, hr)
        val_ssim = self.ssim(fake_hr, hr)
        val_psnr = self.psnr(fake_hr, hr)

        self.val_g_loss.append(val_g_loss.cpu().numpy())
        self.val_ssim.append(val_ssim.cpu().numpy())

        if self.verbose > 0:
            print('\nVerbose: validation_step (end)\n')
    
        self.log("val_g_loss", val_g_loss, prog_bar=True, on_epoch=True)
        self.log("val_ssim", val_ssim, prog_bar=True, on_epoch=True)
        self.log("val_psnr", val_psnr, prog_bar=False, on_epoch=True)

    def on_validation_epoch_end(self):

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (begining)\n')
        
        mean_val_g_loss = np.array(self.val_g_loss).mean()

        if self.verbose > 0:
            print(f'g_loss: {mean_val_g_loss}')
            print(f'self.best_valid_loss: {self.best_valid_loss}')

        if mean_val_g_loss < self.best_valid_loss:
            self.best_valid_loss = mean_val_g_loss
            self.save_model("best_checkpoint.pth")

        self.val_g_loss.clear() # free memory

        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass
        
        mean_val_ssim = np.array(self.val_ssim).mean()

        # Note that step should be called after validate()
        if self.hparams.d_scheduler in self.epoch_schedulers:
            sched_d.step(mean_val_ssim)
        if self.hparams.g_scheduler in self.epoch_schedulers:
            sched_g.step(mean_val_ssim)

        self.val_ssim.clear() # free memory

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (end)\n')

    def on_train_end(self):
        self.save_model("last_checkpoint.pth")

    def configure_optimizers(self):
        
        if self.verbose > 0:
            print('\nVerbose: configure_optimizers (begining)\n')
            print(f'Generator optimizer: {self.hparams.g_optimizer}')
            print(f'Discriminator optimizer: {self.hparams.d_optimizer}')
            print(f'Generator scheduler: {self.hparams.g_scheduler}')
            print(f'Discriminator scheduler: {self.hparams.d_scheduler}')

        self.opt_g = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.g_optimizer,
            learning_rate=self.hparams.learning_rate_g,
            check_point=self.hparams.gen_checkpoint,
            parameters=self.generator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        self.opt_d = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.d_optimizer,
            learning_rate=self.hparams.learning_rate_d,
            check_point=None,
            parameters=self.discriminator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_g = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.g_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_g,
            monitor_loss="val_g_loss",
            name="g_lr",
            optimizer=self.opt_g,
            frequency=self.hparams.n_critic_steps,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_d = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.d_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_d,
            monitor_loss="val_g_loss",
            name="d_lr",
            optimizer=self.opt_d,
            frequency=1,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        if sched_g is None and sched_d is None:
            scheduler_list = []
        else:
            scheduler_list = [sched_g, sched_d]

        return [self.opt_g, self.opt_d], scheduler_list
    
    
    def save_model(self, filename):
        if self.hparams.save_basedir is not None:
            torch.save(
                {
                    "model_state_dict": self.generator.state_dict(),
                    "optimizer_state_dict": self.opt_g.state_dict(),
                    "scale_factor": self.hparams.scale_factor,
                    "best_valid_loss": self.best_valid_loss,
                },
                self.hparams.save_basedir + "/" + filename,
            )
        else:
            raise Exception(
                "No save_basedir was specified in the construction of the WGAN object."
            )
        if self.verbose > 0:
            print('\nVerbose: Model initialized (begining)\n')

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Free cuda memory
        torch.cuda.empty_cache()
    
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio()

        self.data_len = self.hparams.data_len

        self.val_loss = []
        self.val_ssim = []

        if self.verbose > 0:
            os.makedirs(f"{self.hparams.save_basedir}/training_images", exist_ok=True)

        self.step_schedulers = ['CosineDecay', 'OneCycle', 'MultiStepScheduler']
        self.epoch_schedulers = ['ReduceOnPlateau']

        if self.verbose > 0:
            print('\nVerbose: Model initialized (end)\n')

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        raise NotImplementedError

    def on_train_end(self):
        self.save_model("last_checkpoint.pth")

    def configure_optimizers(self):
        
        if self.verbose > 0:
            print('\nVerbose: configure_optimizers (begining)\n')
            print(f'Generator optimizer: {self.hparams.g_optimizer}')
            print(f'Discriminator optimizer: {self.hparams.d_optimizer}')
            print(f'Generator scheduler: {self.hparams.g_scheduler}')
            print(f'Discriminator scheduler: {self.hparams.d_scheduler}')

        self.opt_g = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.g_optimizer,
            learning_rate=self.hparams.learning_rate_g,
            check_point=self.hparams.gen_checkpoint,
            parameters=self.generator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        self.opt_d = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.d_optimizer,
            learning_rate=self.hparams.learning_rate_d,
            check_point=None,
            parameters=self.discriminator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_g = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.g_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_g,
            monitor_loss="val_g_loss",
            name="g_lr",
            optimizer=self.opt_g,
            frequency=self.hparams.n_critic_steps,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_d = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.d_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_d,
            monitor_loss="val_g_loss",
            name="d_lr",
            optimizer=self.opt_d,
            frequency=1,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        if sched_g is None and sched_d is None:
            scheduler_list = []
        else:
            scheduler_list = [sched_g, sched_d]

        return [self.opt_g, self.opt_d], scheduler_list
    
    
    def save_model(self, filename):
        raise NotImplementedError