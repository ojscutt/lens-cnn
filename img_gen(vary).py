# ======================================================================================
# IMPORT RELEVANT PACKAGES
import numpy as np
import math
import lens_functions as lf
import matplotlib.pyplot as plt
from PIL import Image
import astropy.cosmology as cosmo
import lenstronomy.Util.util as util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.Profiles.interpolation import Interpol
from lenstronomy.Data.coord_transforms import Coordinates
from scipy.stats import truncnorm
from scipy.constants import pi
from scipy.constants import c
import random
import time
import os

# ======================================================================================
# USER INPUTS
set_size = 500 #set desired number of images in dataset
res_scale = 1 #set desired resolution scaling factor (60x60=1, 30x30=0.5 etc.)
im_set = 2 #define image set to create, 1 = train, 2 = test

# ======================================================================================
# MISC SETUP
start1 = time.time()

SB = np.zeros([set_size,3]) #array for lens params .txt file

## define gaussian source
x, y = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.3, 0.0
gauss_blob = (np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ))))*255

if not os.path.exists('dataset' + str(im_set)):
    os.makedirs('dataset' + str(im_set))
    
if not os.path.exists('dataset' + str(im_set) + '/images'):
    os.makedirs('dataset' + str(im_set) + '/images')

# ======================================================================================
# LENS MODELLER
for k in range(set_size):
    # ==================================================================================
    # DEFINE LENS MODEL PARAMETERS
    main_halo_type = 'SIE'

    ## define ellipticity
    lower_lim = 0.1
    upper_lim = 2
    mean = 0.78
    sigma = 0.12
    q = truncnorm.rvs((lower_lim-mean)/sigma,
                      (upper_lim - mean)/sigma,
                      loc=mean,
                      scale=sigma
                     )
    
    ## define orientation
    phi = np.random.uniform(0,2*np.pi)
    
    ## combine ellipticity and orientation into complex ellipticity
    e1 = (1-q**2)/(1+q**2) * math.cos(2*phi)
    e2 = (1-q**2)/(1+q**2) * math.sin(2*phi)
    
    ## define Einstein radius
    lower = 0.3
    upper = 2.7
    meanE = 1.5
    sigmaE = 0.4
    
    theta_E = truncnorm.rvs((lower - meanE)/sigma,
                            (upper - meanE)/sigmaE,
                            loc=meanE,
                            scale=sigmaE
                           )

    ## lens parameters Einstein rad, complex ellipticity, lens centre x and y
    kwargs_lens_main = {'theta_E': theta_E,
                        'e1': e1,
                        'e2': e2,
                        'center_x': np.random.uniform(-0.1,0.1),
                        'center_y':  np.random.uniform(-0.1,0.1)
                       }
    kwargs_shear = {'gamma1': 0.0, 'gamma2': 0}
    lens_model_list = [main_halo_type, 'SHEAR']
    kwargs_lens_list = [kwargs_lens_main, kwargs_shear]

    # ==================================================================================
    # SOURCE AND LENS REDSHIFTS
    
    data = gauss_blob #source object
    pixel_size = 0.395 #pixel size to be scaled
    z1 = 0.1 #lens redshift (average z value of SDSS data)

    ## source redshift
    meanZ = 1.77
    sigmaZ = 0.75
    Zlower = 1
    Zupper = 6

    z2 = truncnorm.rvs((Zlower-meanZ)/sigmaZ,
                       (Zupper - meanZ)/sigmaZ,
                       loc=meanZ,
                       scale=sigmaZ
                      )
    
    ## calculate angular size
    da1=cosmo.WMAP9.angular_diameter_distance(z1) #angular size at z1
    da2=cosmo.WMAP9.angular_diameter_distance(z2) #angular size at z2

    new_pixel_size =pixel_size * da1/da2/3

    # ==================================================================================
    # MASS OF LENS
    
    A = 1200
    alpha = 1/3
    hz = 0.8
    D_s = da2
    D_d = cosmo.WMAP9.angular_diameter_distance(1) #Lens at redshift 1
    D_ds = D_s - D_d *(1+1)/(1+z2)

    sigma_lens = math.sqrt(abs((kwargs_lens_main['theta_E'] * c**2) /4*pi * D_s/D_ds))
    mass_lens = 0.9*(A/sigma_lens)**alpha * 10**15/hz
    mass_limit = 0.1 * mass_lens #limit for substructures
    
    # ==================================================================================
    # SUBSTRUCTURE MASSES
    
    a = random.uniform(1.5,2.5) #range of substructre mass power law (negative)
    m = 4  #range of possible powers (e.g 10^8 -> 10^11 = 4)
    subs_min = 8 #minimum power of 10 (e.g 8 gives min of 10^8)

    ## create array of possible substructure masses following defined a
    power_distribution = lf.truncated_power_law(a=a, m=m)
    sample = np.int64(power_distribution.rvs(size=2500)) + (subs_min - 1)
    plaw_masses = 10**sample

    ## cut off array where cumulative total exceeds mass limit
    N = np.where(plaw_masses.cumsum() > mass_limit)[0][0]
    mass_sub_options = plaw_masses[0:N]

    ## define substructure morphology
    sigma =  A * abs((hz*mass_sub_options)/10**15)**alpha
    sub_ER = 4 * pi * (sigma*10**3)**2/c**2 * D_ds/D_s
    sub_ER = sub_ER * (3600*180)/pi #rad to arcsecond conversion
    
    # ==================================================================================
    # SUBSTRUCTURE POSITIONS

    subhalo_type = 'SIS'

    num_subhalo = N  #number of subhalos to be rendered
    
    max_r = 2*kwargs_lens_main['theta_E'] #max distance from centre (2*theta_E)
    
    ## generating substructure positions
    center_x_list = np.zeros(N)
    center_y_list = np.zeros(N)
    for n in range(N):
        rand_angle = 2 * math.pi * random.random()
        rand_rad = max_r * math.sqrt(random.random())

        center_x_list[n] = (rand_rad * math.cos(rand_angle))
        center_y_list[n] = (rand_rad * math.sin(rand_angle))

    ## combine morphology and position of substructures
    theta_E_list = sub_ER
    for j in range(num_subhalo):
        lens_model_list.append(subhalo_type)
        kwargs_lens_list.append({'theta_E': theta_E_list[j], 
                                 'center_x': center_x_list[j],
                                 'center_y': center_y_list[j]
                                })

    # ==================================================================================
    # RAY SHOOTING FOR IMAGE GENERATION

    ## set up coord grid for evalution of lensing
    lensModel = LensModel(lens_model_list)
    x_grid, y_grid = util.make_grid(numPix=60*res_scale, deltapix=0.1*res_scale)
    kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens_list)
    kappa = util.array2image(kappa)

    ## define high res grid for ray shooting
    numPix = 60*res_scale
    deltaPix = new_pixel_size
    res_factor = new_pixel_size/(0.1/res_scale)
    theta_x_high_res, theta_y_high_res = util.make_grid(numPix=numPix,
                                                        deltapix=deltaPix/res_factor
                                                       )
    
    ## ray-shoot the image plane coordinates to the source plane
    beta_x_high_res, beta_y_high_res = lensModel.ray_shooting(theta_x_high_res,
                                                              theta_y_high_res,
                                                              kwargs=kwargs_lens_list
                                                             )

    ## surface brightness computation using interpolation function
    kwargs_interp = {'image': data,
                     'center_x': kwargs_lens_main['center_x'],
                     'center_y': kwargs_lens_main['center_y'],
                     'scale': new_pixel_size,
                     'phi_G': 0
                    }

    interp_light = Interpol()
    source_lensed_interp = interp_light.function(beta_x_high_res,
                                                 beta_y_high_res,
                                                 **kwargs_interp
                                                )
    source_lensed_interp = util.array2image(source_lensed_interp) #map to image
    
    # ==================================================================================
    # NOISE AND PSF
    
    ## PSF
    image_blurred = lf.lens_blur(source_lensed_interp, 0.0722/res_scale)
    data_size=numPix*res_factor
    r=int(data_size)
    signal_annulus=lf.signal_to_noise(image_blurred,
                                      data_size,
                                      data_size,
                                      r)

    ## Noise
    s2n=300 #signal to noise ratio
    signal=signal_annulus['aperture_sum'][0]
    noise=signal_annulus['aperture_sum'][0]/s2n
    noise_sigma= np.abs(noise/(np.sqrt(np.pi*(r**2))))
    noise_image=lf.lens_noise(image_blurred, noise_sigma)

    min_im = np.min(noise_image)
    noise_image = noise_image + np.abs(min_im)+1

    max_im = np.max(noise_image)
    noise_image = noise_image/max_im

    # ==================================================================================
    # NORMALISE AND SAVE IMAGE
    
    ## convert to greyscale RGB (max 255)
    im = noise_image * 255
    im = Image.fromarray(im)
    im = im.convert("L")
    
    ## save image png
    im.save('dataset' + str(im_set) + '/images/SIE_im' + str(k) + '.png')
    
    ## plot image
    #plt.matshow(im, cmap="gray")
    #plt.show()

    # ==================================================================================
    # STORE LENS AND SUBSTRUCTURE PARAMS
    
    SB[k][0] = (k) #lens number
    SB[k][1] = num_subhalo #number of substructures
    SB[k][2] = a #substructure mass dist power law

    ## print lens params
    #print("{} {} {}".format(k, num_subhalo, a))
    
# ======================================================================================
# END OF IMAGE GEN LOOP

## save lens parameters in txt
np.savetxt('dataset' + str(im_set) + '/SB' + str(im_set) + '.txt',
           SB,
           delimiter=' ',
           fmt='%i %i %1.10f'
          )

end1 = time.time()
print(round(end1 - start1,2), 's')