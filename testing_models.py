import os
import sys
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from trainers import *

dataset_config = {'EM': [None, 'train', None, None, None, 'test'],
                  'MitoTracker_small': [None, 'train', None, None, None, 'test'],
                  }

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

dataset_name = 'EM'

train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = dataset_config[dataset_name]

dataset_root = '../datasets'
train_lr_path = os.path.join(dataset_root, dataset_name, train_lr) if train_lr is not None else None
train_hr_path = os.path.join(dataset_root, dataset_name, train_hr) if train_hr is not None else None
val_lr_path = os.path.join(dataset_root, dataset_name, val_lr) if val_lr is not None else None
val_hr_path = os.path.join(dataset_root, dataset_name, val_hr) if val_hr is not None else None
test_lr_path = os.path.join(dataset_root, dataset_name, test_lr) if test_lr is not None else None
test_hr_path = os.path.join(dataset_root, dataset_name, test_hr) if test_hr is not None else None

scale = 4

model_configuration = {'optim': {'early_stop':{'loss':'val_ssim_loss','mode':'max', 'patience':10},
                                 'adam':{'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'adamax':{'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'adamW':{'decay':0.004,'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'sgd_momentum':0.9,
                                 'ReduceOnPlateau':{'monitor':'val_loss','factor':0.5,'patience':3},
                                 'MultiStepScheduler':{'lr_steps':[50000, 100000, 200000, 300000],
                                                       'lr_rate_decay':0.5}},
                       'rcan': {'num_filters':16,
                                'percp_coef': 1000},
                       'dfcan': {'n_ResGroup': 4, 'n_RCAB': 4},
                       'wdsr': {'num_res_blocks': 32},
                       'unet': {'init_channels': 16,
                                'depth': 4,
                                'upsample_method': 'SubpixelConv2D',
                                'maxpooling': False,
                                'percp_coef': 10},
                       'wgan': {'g_layers': 15,
                                'd_layers': 5,
                                'recloss': 100.0,
                                'lambda_gp':10},
                       'esrganplus': {'n_critic_steps':5},
                       'others': {'positional_encoding':False,
                                  'positional_encoding_channels':64}
                      }

test_metric_indexes = [69,  7, 36, 75, 74, 30, 12, 42, 87,  0]

optimizer = 'Adam'  #'Adam', 'Adamax', 'RMSprop', 'SGD'
discriminator_optimizer = 'Adam'  #'Adam', 'Adamax', 'RMSprop', 'SGD'
scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
discriminator_lr_scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'

crappifier_method = 'downsampleonly'

model_name = 'wgan' # ['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']
seed = 666
batch_size = 2
number_of_epochs = 5
lr = 0.0001
discriminator_lr = 0.0001
additional_folder = "prueba"

num_patches = 16
patch_size_x = 64
patch_size_y = 64
validation_split = 0.1
data_augmentation = ['rotation', 'horizontal_flip', 'vertical_flip']

model = train_configuration(
                 data_name=dataset_name, 
                 train_lr_path=train_lr_path, train_hr_path=train_hr_path, 
                 val_lr_path=val_lr_path, val_hr_path=val_hr_path,
                 test_lr_path=test_lr_path, test_hr_path=test_hr_path,
                 crappifier_method=crappifier_method, 
                 model_name=model_name, scale_factor=scale, 
                 number_of_epochs=number_of_epochs, batch_size=batch_size, 
                 learning_rate=lr, discriminator_learning_rate=discriminator_lr, 
                 optimizer_name=optimizer, lr_scheduler_name=scheduler, 
                 test_metric_indexes=test_metric_indexes, 
                 additional_folder=additional_folder, 
                 model_configuration=model_configuration, seed=seed,
                 num_patches=num_patches, patch_size_x=patch_size_x, patch_size_y=patch_size_y, 
                 validation_split=validation_split, data_augmentation=data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=0
                )
                 