###---multimedia project: geometric shapes detection by Adam Vedernikov & Inna Lopovok---###

import functions as func

dir_regular = "shapes_regular\\"
dir_low_noise = "shapes_noise_low\\"
dir_high_noise = "shapes_noise_high\\"

#img_index can be changed to any value between 1-50 to open different images
img_index = 13

##demonstrates shape detection on regular (non-noisy) images
func.demo_regular(dir_regular, img_num=img_index)

##demonstrates shape detection on low-noise images + median filter noise reduction
func.demo_low_noise(dir_low_noise, img_num=img_index)

##demonstrates shape detection on low-noise images + median filter+ morphological noise reductions
func.demo_high_noise(dir_high_noise, img_num=img_index)


###---UNCOMMENT THESE FUNCTION CALLS FOR MORE DEMONSTRATIONS---###

##same as high-noise image demonstration, but allows modifying the intensity of the salt & pepper noise
#func.demo_modifiable_noise(dir_regular, img_num=img_index, sp_probability=0.2)

##demonstrates the accuracy verification process. For each image in the image folders it displays the shape detection outputs for 3 versions of an image simultaneously: regular, low-noise, high-noise
#func.verify_accuracy(dir_regular, dir_low_noise, dir_high_noise)
