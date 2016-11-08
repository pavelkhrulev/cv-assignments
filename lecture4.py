from PIL import Image
from numpy import *
from scipy.ndimage import filters
import scipy.misc
import matplotlib.pyplot as plt

def get_magnitude(input_image, sigma):
    # Pre-smooth
    blured = filters.gaussian_filter(input_image,sigma)
    # Compute gradient
    imx = zeros(blured.shape)
    filters.gaussian_filter(blured, sigma, [1, 0], output=imx)
    imy = zeros(blured.shape)
    filters.gaussian_filter(blured, sigma, [0, 1], output=imy)
    # Compute magnitude
    return sqrt(imx**2 + imy**2)

def plot_red_stars(magnitude, axis):
    top5border = amax(magnitude)*0.95
    for index, x in ndenumerate(magnitude):
        if x > top5border:
            axis.plot(index[1], index[0], 'r*', markersize = 15)

def run_assignment():
    input_image = array(Image.open('ImageFolder/cat_small.jpg').convert('L'))

    fig = plt.figure()
    plt.gray()
    ax1 = plt.subplot2grid((4,1), (0,0)) #input
    ax2 = plt.subplot2grid((4,1), (1,0)) #im1
    ax3 = plt.subplot2grid((4,1), (2,0)) #im5
    ax4 = plt.subplot2grid((4,1), (3,0)) #im10

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    
    magnitude_1 = get_magnitude(input_image, 1)
    magnitude_5 = get_magnitude(input_image, 5)
    magnitude_10 = get_magnitude(input_image, 10)
    
    ax1.imshow(input_image)
    ax2.imshow(magnitude_1)
    ax3.imshow(magnitude_5)
    ax4.imshow(magnitude_10)
    
    plot_red_stars(magnitude_1, ax2)
    plot_red_stars(magnitude_5, ax3)
    plot_red_stars(magnitude_10, ax4)
    
    plt.draw()
    plt.show()
    fig.savefig('ImageFolder/lecture4_result.jpg')
    
    print 'Done'
    
if __name__ == '__main__':
    run_assignment()
