import numpy as np
from skimage import filters
from skimage.util import random_noise, img_as_float
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale
from matplotlib import pyplot as plt
from skimage import io

# Create corresponding training patches synthetically by adding noise
# and downsampling the images (see https://www.biorxiv.org/content/10.1101/740548v3)

def fluo_G_D(x, scale=4):
    mu, sigma = 0, 5
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    
    return npzoom(x, 1/scale, order=1)

def fluo_AG_D(x, scale=4):
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return npzoom(x, 1/scale, order=1)

def downsampleonly(x, scale=4):
    return npzoom(x, 1/scale, order=1)

def fluo_SP_D(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)

    return npzoom(x, 1/scale, order=1)

def fluo_SP_AG_D_sameas_preprint(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return npzoom(x, 1/scale, order=1)

def fluo_SP_AG_D_sameas_preprint_rescale(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

def em_AG_D_sameas_preprint(x, scale=4):
    lvar = filters.gaussian(x, sigma=3)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    
    return npzoom(x, 1/scale, order=1)

def em_G_D_001(x, scale=4):
    noise = np.random.normal(0, 3, x.shape)
    x = x + noise
    x = x - x.min()
    x = x/x.max()
    
    return npzoom(x, 1/scale, order=1)

def em_G_D_002(x, scale=4):
    mu, sigma = 0, 3
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def em_P_D_001(x, scale=4):
    x = random_noise(x, mode='poisson', seed=1)

    return npzoom(x, 1/scale, order=1)

def new_crap_AG_SP(x, scale=4):
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)

    return rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

def new_crap(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)
        
    return rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

def apply_crappifier(x, scale, crappifier_name):
    crappifier_dict = { 'downsampleonly':downsampleonly,
                        'fluo_G_D':fluo_G_D, 
                        'fluo_AG_D':fluo_AG_D,
                        'fluo_SP_D':fluo_SP_D,
                        'fluo_SP_AG_D_sameas_preprint':fluo_SP_AG_D_sameas_preprint,
                        'fluo_SP_AG_D_sameas_preprint_rescale':fluo_SP_AG_D_sameas_preprint_rescale,
                        'em_AG_D_sameas_preprint':em_AG_D_sameas_preprint,
                        'em_G_D_001':em_G_D_001,
                        'em_G_D_002':em_G_D_002,
                        'em_P_D_001':em_P_D_001,
                        'new_crap_AG_SP':new_crap_AG_SP,
                        'new_crap':new_crap}
    
    if crappifier_name in crappifier_dict:
        return crappifier_dict[crappifier_name](x, scale)
    else:
        raise ValueError('The selected crappifier_name is not in: {}'.format(crappifier_dict))