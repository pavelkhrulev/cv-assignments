import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage

low_sigma_count = 0
high_sigma_count = 0

def subplot((place1,place2,place3),img, title):
    plt.subplot(place1,place2,place3)
    plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    
def image_to_frequency_domain(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

def low_pass_filter(image, sigma):
    return ndimage.gaussian_filter(image, sigma)
    
def high_pass_filter(image, sigma):
    return image - low_pass_filter(image, sigma)

def create_hybrid_image(image_to_low_pass, image_to_high_pass,
                        low_sigma, high_sigma, plot = True):
    
    low_pass_image = low_pass_filter(image_to_low_pass, low_sigma)
    high_pass_image = high_pass_filter(image_to_high_pass, high_sigma)
    
    hybrid_image = low_pass_image + high_pass_image
    
    if plot is False:
        return hybrid_image

    # Plot images after filtering
    subplot((4, 2, 5), low_pass_image,'Image 1 after low pass filtering')
    subplot((4, 2, 6), high_pass_image,'Image 2 after high pass filtering')
    
    # Go to frequency domain and plot images
    low_pass_image_freq_domain = image_to_frequency_domain(low_pass_image)
    high_pass_image_freq_domain = image_to_frequency_domain(high_pass_image)
    subplot((4, 2, 7), low_pass_image_freq_domain,'Image 1 after LPF in frequency domain')
    subplot((4, 2, 8), high_pass_image_freq_domain,'Image 2 after HPF in frequency domain')
    
    plt.savefig('helper_images.jpg')
    
    plt.figure(2)
    plt.xticks([])
    plt.yticks([])
    plt.gray()
    plt.imshow(hybrid_image)
    plt.savefig('hybrid_image.jpg')
    plt.show()
        
def create_helper_video(low_filtered_image, high_filtered_image,
                        low_sigma_limit, high_sigma_limit):
    print 'start video creating'
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Creating current iteration label with sigma values 
    iteration_label = ax.text(0, -30, '0  0')
    
    # Initialization
    im = ax.imshow(create_hybrid_image(low_filtered_image, high_filtered_image, 0, 0, plot=False), cmap='gray')

    def update_frame(n, low_sigma_limit, high_sigma_limit):
        global low_sigma_count, high_sigma_count
        
        # Updating sigma counters
        if high_sigma_count == high_sigma_limit or low_sigma_count == 0:
            high_sigma_count = 0
            low_sigma_count += 1
            
        high_sigma_count += 1
        
        # Getting new hybrid image
        hybrid_image = create_hybrid_image(low_filtered_image, high_filtered_image,
                                           low_sigma_count, high_sigma_count, plot=False)
        im.set_data(hybrid_image)
        
        # Updating iteration label
        iteration_label.set_text('low sigma count: %d, high sigma count: %d'
                                 %(low_sigma_count,high_sigma_count))
        
        return im

    ani = animation.FuncAnimation(fig, update_frame, low_sigma_limit*high_sigma_limit - 1, 
                                  interval=1000, fargs=(low_sigma_limit, high_sigma_limit) )
    writer = animation.writers['ffmpeg'](fps=1)

    ani.save('hybrid_video.mp4',writer=writer,dpi=100)
    print 'end of video creating'
    
def main_task(low_pass_image, high_pass_image):
    # Go to frequency domain
    low_pass_image_freq_domain = image_to_frequency_domain(low_pass_image)
    high_pass_image_freq_domain = image_to_frequency_domain(high_pass_image)
    
    # Plot input images
    subplot((4,2,1), low_pass_image, 'Input Image 1')
    subplot((4,2,3), low_pass_image_freq_domain, 'Image 1 in frequency domain')
    subplot((4,2,2), high_pass_image,'Input Image 2')
    subplot((4,2,4), high_pass_image_freq_domain, 'Image 2 in frequency domain')
    
    ''' 
    Video helped to determine the best sigma values for image sets.
    Set 1 (2->1): {5,2}, {3,3}, {4,2}
    Set 2 (1->2): {3,3}
    Set 3 (2->1): {6,5}, {7,4}, {4,6}
    '''

    # Creating hybrid image
    create_hybrid_image(low_pass_image, high_pass_image, 4, 2, plot=True)
    
    plt.show()
    
def helper_task(low_pass_image, high_pass_image):
    create_helper_video(low_pass_image, high_pass_image, 10, 10)

if __name__ == "__main__":
    low_pass_image = ndimage.imread("./ProjectImages/set1_2.jpg", flatten=True)
    high_pass_image = ndimage.imread("./ProjectImages/set1_1.jpg", flatten=True)
    
    '''
    It's hard to find proper sigma values to create hybrid image immediately.
    You can use helper task for creating video that includes many frames
    with different sigma values and choose the one.
    '''
    #helper_task()
    
    main_task(low_pass_image, high_pass_image)
    