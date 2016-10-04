from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.misc, scipy.io as sio
from random import randint

""" Histogram equalization of a grayscale image. """ 
def histogram_equalization(im,nbr_bins=256):
    # get image histogram 
    imhist,bins = np.histogram(im.flatten(), nbr_bins, normed=True) 
    cdf = imhist.cumsum() 
    # cumulative distribution function 
    cdf = 255 * cdf / cdf[-1] 
    # normalize 
    # use linear interpolation of cdf to find new pixel values 
    im2 = np.interp(im.flatten(),bins[:-1],cdf) 
    return im2.reshape(im.shape), cdf, bins

def histogram_matching(im, g_cdf, bins):
    output_image = np.interp(im.flatten(),g_cdf,bins[:-1]) 
    return output_image.reshape(im.shape)


def get_axes():
    ax = []

    ax.append(plt.subplot2grid((4,2), (0,0)))
    ax.append(plt.subplot2grid((4,2), (0,1)))
    ax.append(plt.subplot2grid((4,2), (1,0)))
    ax.append(plt.subplot2grid((4,2), (1,1)))
    ax.append(plt.subplot2grid((4,2), (2,0), colspan=2))
    ax.append(plt.subplot2grid((4,2), (3,0)))
    ax.append(plt.subplot2grid((4,2), (3,1)))
    
    return ax

def RunAssignment():
    ax = get_axes()
    plt.gray()
    
    bins_count = 256
    
    # Drawing input image with its histogram.
    input_image = Image.open("ImageFolder/dolphins.jpg").convert('L')
    input_image_data = np.array(input_image)
    ax[0].axis('off')
    ax[0].imshow(input_image)
    ax[2].hist(input_image_data.flatten(), bins_count)
    
    # Getting random normal distribution.
    normal_distribution = np.random.normal(127, 32, 10000)
    # Desired probability density function.
    ax[4].hist(normal_distribution, bins_count)
    
    # Getting T(x).
    output_uniform, cdf1, bins1 = histogram_equalization(input_image_data, bins_count)
    # Getting G(x).
    output_random, cdf2, bins2 = histogram_equalization(normal_distribution, bins_count)
    
    # Transform function for the input image.
    ax[5].plot(cdf1)
    # Transform function for getting uniform from the normal.
    ax[6].plot(cdf2)
    
    # Getting resulting image using matching algorithm.
    output_image = histogram_matching(output_uniform, cdf2, bins2)
    output_image_data = np.array(output_image)
    ax[1].axis('off')
    ax[1].imshow(output_image)
    # Drawing real output image histogram.
    ax[3].hist(output_image_data.flatten(), bins_count)
    
    plt.gray()
    plt.show()
    
if __name__ == '__main__':
    RunAssignment()