###---multimedia project: geometric shapes detection by Adam Vedernikov & Inna Lopovok---###

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

#The main function of the program, which converts an image to a grayscale format,then to binary using a threshold function.
#The binary image is then used to find the contours of the image, which allow us to find geometric shapes by counting the number of contours.
#Then the function uses the info provided by the contours to draw text (tagging the shape) & also to color them according to their category
def find_shapes(image_bgr):
    image_output = image_bgr.copy()
    width = image_bgr.shape[1]
    image_gray = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        if (cv2.arcLength(contour, True) <= 100): #to make sure small traces of noise don't get counted as ellipses
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(image_output, [approx], 0, (0, 0, 0), 3)
        x = approx.ravel()[0] - 45 #x,y are used to find the right coordinates to place the text over the shape
        y = approx.ravel()[1] + 80

        if len(approx) == 3:
            cv2.drawContours(image_output, [contour], 0, (255, 0, 0), -1)  # paint triangles blue
            cv2.putText(image_output, "Triangle", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        elif len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            if (w == width): # technically the whole image is a rectangle, so to avoid painting everything red we continue
                continue
            ratio = float(w)/h
            if ratio >= 0.95 and ratio <= 1.05:
                cv2.drawContours(image_output, [contour], 0, (205, 205, 0), -1)  # paint squares cyan
                cv2.putText(image_output, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            else:
                cv2.drawContours(image_output, [contour], 0, (0, 0, 255), -1)  # paint rectangles red
                cv2.putText(image_output, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        elif len(approx) == 5:
            cv2.drawContours(image_output, [contour], 0, (0, 140, 255), -1)  # paint pentagons orange
            cv2.putText(image_output, "Pentagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        elif len(approx) == 6:
            cv2.drawContours(image_output, [contour], 0, (219, 112, 147), -1)  # paint hexagons purple
            cv2.putText(image_output, "Hexagon", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        elif len(approx) == 10:
            cv2.drawContours(image_output, [contour], 0, (0, 255, 255), -1)  # paint stars yellow
            cv2.putText(image_output, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        else:
            cv2.drawContours(image_output, [contour], 0, (0, 255, 0), -1)  # paint ellipses green
            cv2.putText(image_output, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    return image_output, image_gray, image_binary

#displaying the regular image transformation from img->grey->binary->output img with detected shapes
def show_regular(input_img, grey_img, bin_img, output_img, title="figure1"):
    fig, _ = plt.subplots(figsize=(10, 10))
    fig.canvas.set_window_title(title)

    plt.subplot(221)
    plt.axis('off')
    plt.title("input")
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    plt.subplot(222)
    plt.axis('off')
    plt.title("greyscale")
    plt.imshow(cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB))

    plt.subplot(223)
    plt.axis('off')
    plt.title("binary")
    plt.imshow(cv2.cvtColor(bin_img, cv2.COLOR_BGR2RGB))

    plt.subplot(224)
    plt.axis('off')
    plt.title("output")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.show()

    return


##############################---NOISE FOCUSED FUNCTIONS---##############################

#since random gives us a number between 0-1, there's an equal chance between turning a pixel on or off,
#meaning the higher is 'probability's value the higher is the range of numbers for turning on/off a pixel
def add_salt_pepper_noise(image,probability=0.01):
    image_shape = image.shape
    num_of_rows = image_shape[0]
    num_of_cols = image_shape[1]
    img_noisy = image.copy()

    for row in range(num_of_rows):
        for column in range(num_of_cols):
            rand_num = np.random.random()

            if rand_num > 1 - probability:
                img_noisy[row][column] = 255
            elif rand_num < probability:
                img_noisy[row][column] = 0
    return img_noisy

#this function receives a source folder destination with images, and then adds 'salt and pepper' noise (intensity controlled by 'probability' variant),
#and saves the noisy variants in the destination folder with the chosen tag (i.e, img1.jpg -> img1_noisy.jpg)
def salt_pepper_folder(source, destination, probability, tag, image_type='.jpg'):
    imgs = os.listdir(source)

    for img in imgs:
        img_edited = img.replace(image_type,"")
        src_img = cv2.imread(source+'\\'+img)
        noisy_img = add_salt_pepper_noise(src_img, src_img.shape, probability)
        cv2.imwrite(destination+'\\'+img_edited+'_'+tag+image_type, noisy_img)
    return

#applies a median blur noise reduction to an image
def reduce_noise_median(img_noisy, krnl_size = 5):
    if ((krnl_size is not None) and (krnl_size % 2 == 1)):
        img_noisy = cv2.medianBlur(img_noisy, krnl_size)
    else:
        print("Wrong kernel input")
    return img_noisy

#applies morph operations OPEN+CLOSE noise reduction to an image
def reduce_noise_morph(img_noisy, krnl_size = 3):
    if ((krnl_size is not None) and (krnl_size % 2 == 1)):
        kernel = np.ones((krnl_size, krnl_size), np.uint8)
        img_noisy = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)
        img_noisy = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, kernel)
    else:
        print("Wrong kernel input")
    return img_noisy

#showing the process of denoising an image, with the option to skip morphological operations if the noise is low-mid
def show_denoise(noisy_img, median_img, morphed=None, title="figure1"):
    fig, _ = plt.subplots(figsize=(10, 10))
    fig.canvas.set_window_title(title)

    plt.subplot(131)
    plt.axis('off')
    plt.title("Input with noise")
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))

    plt.subplot(132)
    plt.axis('off')
    plt.title("Median")
    plt.imshow(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))

    if(morphed is not None):
        plt.subplot(133)
        plt.axis('off')
        plt.title("Morphed")
        plt.imshow(cv2.cvtColor(morphed, cv2.COLOR_BGR2RGB))
    plt.show()
    return

#a function which generates outputs of find_shapes together in order to compare the accuracy on different levels of noise
def verify_accuracy(dir_regular, dir_low_noise, dir_high_noise, image_type='.jpg'):
    shape_detect_title = "Shape detection verification (loops through all of the 50 images)"
    imgs = os.listdir(dir_regular)

    for img in imgs:
        img_edited = img.replace(image_type,"")
        img_regular = cv2.imread(dir_regular+"\\"+img)
        img_low_noise = cv2.imread(dir_low_noise+"\\"+img_edited+"_low_noise"+image_type)
        img_high_noise = cv2.imread(dir_high_noise+"\\"+img_edited+"_high_noise"+image_type)

        regular_res,_,_ = find_shapes(img_regular)

        img_low_noise = reduce_noise_median(img_low_noise)
        low_noise_res,_,_ = find_shapes(img_low_noise)

        img_high_noise = reduce_noise_median(img_high_noise)
        img_high_noise = reduce_noise_morph(img_high_noise)
        high_noise_res,_,_ = find_shapes(img_high_noise)

        fig, _ = plt.subplots(figsize=(7, 13))
        fig.canvas.set_window_title(shape_detect_title)

        plt.subplot(311)
        plt.axis('off')
        plt.title(img_edited)
        plt.imshow(cv2.cvtColor(regular_res, cv2.COLOR_BGR2RGB))

        plt.subplot(312)
        plt.axis('off')
        plt.title("low_noise")
        plt.imshow(cv2.cvtColor(low_noise_res, cv2.COLOR_BGR2RGB))

        plt.subplot(313)
        plt.axis('off')
        plt.title("high_noise")
        plt.imshow(cv2.cvtColor(high_noise_res, cv2.COLOR_BGR2RGB))
        plt.show()
    return

#############################---DEMOS---##################################

#non-noisy image: shape detection demonstration
def demo_regular(dir_regular, img_num):
    shape_detect_title = "Shape detection for a regular (non-noisy) image"
    input_image = cv2.imread(dir_regular + 'shapes'+str(img_num)+'.jpg')
    output_image, grey_img, bin_img = find_shapes(input_image)
    show_regular(input_image, grey_img, bin_img, output_image, title=shape_detect_title)
    return

#low-noise image: denoising+shape detection demonstration
def demo_low_noise(dir_low_noise, img_num):
    shape_detect_title = "Shape detection for a low noise image"
    noise_reduce_title = "Noise reduction for a low noise image noisy->median"
    input_image = cv2.imread(dir_low_noise + 'shapes'+str(img_num)+'_low_noise.jpg')

    median = reduce_noise_median(input_image)
    show_denoise(input_image, median, title=noise_reduce_title)

    output_image, grey_img, bin_img = find_shapes(median)
    show_regular(median, grey_img, bin_img, output_image, title=shape_detect_title)
    return

#high-noise image: denoising+shape detection demonstration
def demo_high_noise(dir_high_noise, img_num):
    shape_detect_title = "Shape detection for a high noise image"
    noise_reduce_title = "Noise reduction for a high noise image noisy->median->morph"
    input_image = cv2.imread(dir_high_noise + 'shapes' + str(img_num) + '_high_noise.jpg')

    median = reduce_noise_median(input_image)
    morph = reduce_noise_morph(median)
    show_denoise(input_image, median, morph, noise_reduce_title)

    output_image, grey_img, bin_img = find_shapes(morph)
    show_regular(morph, grey_img, bin_img, output_image, title=shape_detect_title)
    return

#modified-noise level image: denoising+shape detection demonstration
def demo_modifiable_noise(dir_regular, img_num, sp_probability):
    shape_detect_title = "Shape detection for a modifiable noise image"
    noise_reduce_title = "Noise reduction for a modifiable noise image noisy->median->morph"

    input_image = cv2.imread(dir_regular + 'shapes'+str(img_num)+'.jpg')
    input_image = add_salt_pepper_noise(input_image, sp_probability)
    median = reduce_noise_median(input_image)
    morph = reduce_noise_morph(median)
    show_denoise(input_image, median, morph, title = noise_reduce_title)
    output_image, grey_img, bin_img = find_shapes(morph)
    show_regular(morph, grey_img, bin_img, output_image, title=shape_detect_title)
    return




