
#!/usr/bin/env python
# coding: utf-8

from src.trainers import *

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";

from matplotlib import pyplot as plt
import numpy as np

def plot_images(img_list):
    for img in img_list:
        plt.figure(figsize=(10,10))
        plt.imshow(img, 'gray')
        plt.show()

def print_info(data):
    print('Shape: {}'.format(data.shape))
    print('dtype: {}'.format(data.dtype))
    print('Min: {}'.format(data.min()))    
    print('Min: {}'.format(data.max()))    
    print('Mean: {}'.format(data.mean()))    



train_path = '../datasets/TFM - dataset Electron Microscopy/train'
test_path = '../datasets/TFM - dataset Electron Microscopy/test'
drawn_test_path = './data_example/drawn_test'

train_filenames = sorted([os.path.join(filename) for filename in os.listdir(train_path)])
test_filenames = sorted([os.path.join(filename) for filename in os.listdir(test_path)])
drawn_test_filenames = sorted([os.path.join(filename) for filename in os.listdir(drawn_test_path)])

print(len(train_filenames))
print(len(test_filenames))
print(len(drawn_test_filenames))


# # See one image

from src.datasets import read_image
hr_img = read_image(os.path.join(train_path, train_filenames[0]))
test_img = read_image(os.path.join(test_path, test_filenames[0]))
drawn_test_img = read_image(os.path.join(drawn_test_path, drawn_test_filenames[0]))

print_info(drawn_test_img)


# # Test `prepare_data`

scale = 4

from src.datasets import extract_random_patches_from_folder

train_patches_wf, train_patches_gt = extract_random_patches_from_folder(
                                        hr_data_path=train_path, 
                                        lr_data_path=None, 
                                        filenames=train_filenames, 
                                        scale_factor=scale, 
                                        crappifier_name='em_crappify', 
                                        lr_patch_shape=(64, 64), 
                                        num_patches=16)

X_train = np.expand_dims(train_patches_wf, axis=-1)
Y_train = np.expand_dims(train_patches_gt, axis=-1)

print('X_train')
print_info(X_train)
print('\n')

print('Y_train')
print_info(Y_train)

input_data_shape = X_train.shape
output_data_shape = Y_train.shape


from src.datasets import get_train_val_generators 
batch_size = 8

train_generator, val_generator = get_train_val_generators(X_data=X_train,
                                                          Y_data=Y_train,
                                                          validation_split=0.1,
                                                          batch_size=batch_size,
                                                          show_examples=0,
                                                          rotation=True,
                                                          horizontal_flip=True,
                                                          vertical_flip=True)


dataset_config = {'EM': [None, 'train', None, None, None, 'test'],
                  'MitoTracker_small': [None, 'train', None, None, None, 'test'],
                  'F-actin': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt'],
                  'ER': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt/level_06'],
                  'MT': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt'],
                  'LiveFActinDataset': ['train_split/wf', 'train_split/gt', 'val_split/wf', 'val_split/gt', 'test_split/wf', 'test_split/gt']
                  }

crappifier_config = {'EM': 'em_crappify', 
                     'MitoTracker_small': 'fluo_crappify',
                     'F-actin': 'fluo_SP_AG_D_sameas_preprint',
                     'ER': 'fluo_SP_AG_D_sameas_preprint',
                     'MT': 'fluo_SP_AG_D_sameas_preprint',
                     'LiveFActinDataset': 'fluo_SP_AG_D_sameas_preprint'}

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

model_name = 'unet' # ['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']
seed = 666
batch_size = 8
number_of_epochs = 20
lr = 0.001
discriminator_lr = 0.001
additional_folder = "prueba"

scale = 4

num_patches = 16
patch_size_x = 64
patch_size_y = 64
validation_split = 0.1
data_augmentation = ['rotation', 'horizontal_flip', 'vertical_flip']

dataset_name = 'EM'
library_name = 'tensorflow'

train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = dataset_config[dataset_name]

dataset_root = '../datasets'
train_lr_path = os.path.join(dataset_root, dataset_name, train_lr) if train_lr is not None else None
train_hr_path = os.path.join(dataset_root, dataset_name, train_hr) if train_hr is not None else None
val_lr_path = os.path.join(dataset_root, dataset_name, val_lr) if val_lr is not None else None
val_hr_path = os.path.join(dataset_root, dataset_name, val_hr) if val_hr is not None else None
test_lr_path = os.path.join(dataset_root, dataset_name, test_lr) if test_lr is not None else None
test_hr_path = os.path.join(dataset_root, dataset_name, test_hr) if test_hr is not None else None

train_hr_path = '../datasets/TFM - dataset Electron Microscopy/train'
test_hr_path = '../datasets/TFM - dataset Electron Microscopy/test'

crappifier_method = crappifier_config[dataset_name]

model_trainer = TensorflowTrainer(library_name, 
                train_lr_path, train_hr_path, 
                val_lr_path, val_hr_path,
                test_lr_path, test_hr_path,
                crappifier_method, model_name, scale, 
                number_of_epochs, batch_size, 
                lr, discriminator_lr, 
                optimizer, scheduler, 
                test_metric_indexes, additional_folder, 
                model_configuration, seed,
                num_patches, patch_size_x, patch_size_y, 
                validation_split, data_augmentation,
                discriminator_optimizer=discriminator_optimizer, 
                discriminator_lr_scheduler=discriminator_lr_scheduler,
                verbose=0,
            )

model_trainer.train_generator=train_generator
model_trainer.val_generator=val_generator
model_trainer.input_data_shape = input_data_shape
model_trainer.output_data_shape = output_data_shape
    

model_trainer.train_model()


from src.datasets import extract_random_patches_from_folder

test_path = '../datasets/TFM - dataset Electron Microscopy/test'
drawn_test_path = './data_example/drawn_test'

test_filenames = sorted([os.path.join(filename) for filename in os.listdir(test_path)])[0:10]
drawn_test_filenames = sorted([os.path.join(filename) for filename in os.listdir(drawn_test_path)])[0:10]

lr_images, hr_images = extract_random_patches_from_folder(
                                hr_data_path=test_path, 
                                lr_data_path=None, 
                                filenames=test_filenames, 
                                scale_factor=scale, 
                                crappifier_name='em_crappify', 
                                lr_patch_shape=None, 
                                num_patches=1)

hr_images = np.expand_dims(hr_images, axis=-1)
lr_images = np.expand_dims(lr_images, axis=-1)

print('Test HR images')
print_info(hr_images)
print('\n')
print('Test LR images')
print_info(lr_images)
print('\n')



drawn_lr_images, drawn_hr_images = extract_random_patches_from_folder(
                                hr_data_path=drawn_test_path, 
                                lr_data_path=None, 
                                filenames=drawn_test_filenames, 
                                scale_factor=scale, 
                                crappifier_name='em_crappify', 
                                lr_patch_shape=None, 
                                num_patches=1)

drawn_hr_images = np.expand_dims(drawn_hr_images, axis=-1)
drawn_lr_images = np.expand_dims(drawn_lr_images, axis=-1)

print('Drawn HR images')
print_info(drawn_hr_images)
print('\n')
print('Drawn LR images')
print_info(drawn_lr_images)
print('\n')

from src.optimizer_scheduler_utils import select_optimizer, select_optimizer
from src.model_utils import select_model


optim = select_optimizer(library_name='tensorflow', optimizer_name=optimizer, 
                                learning_rate=0.001, check_point=None,
                                parameters=None, additional_configuration=model_configuration)

model = select_model(model_name=model_name, input_shape=lr_images.shape,  
                    output_channels=1, scale_factor=scale, model_configuration=model_configuration)

loss_funct = 'mean_absolute_error'
eval_metric = 'mean_squared_error'

model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])

# Load old weights
model.load_weights(os.path.join(model_trainer.saving_path, 'weights_best.h5') )   


test_predictions = model.predict(lr_images, batch_size=1)
print('Test predictions')
print_info(test_predictions)
print('\n')



drawn_test_predictions = model.predict(drawn_lr_images, batch_size=1)
print('Drawn predictions')
print_info(drawn_test_predictions)
print('\n')



plot_images([test_predictions[0], drawn_test_predictions[0]])