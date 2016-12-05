from scipy.ndimage import filters 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PIL import Image
from scipy.misc.pilutil import imrotate, imresize

class ImagePoint:
    Flat = 1
    Corner = 2
    Edge = 3

def compute_harris_response(im, k = 0.15, sigma=2):
    # derivatives 
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx) 
    imy = np.zeros(im.shape) 
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy) 

    # compute components of the Harris matrix 
    Wxx = filters.gaussian_filter(imx*imx,sigma) 
    Wxy = filters.gaussian_filter(imx*imy,sigma) 
    Wyy = filters.gaussian_filter(imy*imy,sigma) 
    
    # determinant and trace 
    Wdet = Wxx*Wyy - Wxy**2 
    Wtr = Wxx + Wyy 
    return Wdet-k*(Wtr**2)

def calculate_s_function(image, x0, y0, u_max, v_max, axis):
    S = [[0 for u in range(0, u_max*2)] for v in range(0, v_max*2)]
    
    x_count = 0
    y_count = 0
    for u in range(-u_max, u_max):
        for v in range(-v_max, v_max):
            for u_current in range(-u_max, u_max):
                for v_current in range (-v_max, v_max):
                    test = (image[x0 + u_current + u][y0 + v_current + v] - image[x0 + u_current][y0 + v_current])**2
                    S[x_count][y_count] = S[x_count][y_count] + test
            y_count += 1
        y_count = 0
        x_count += 1
                    
    
    X = np.arange(-u_max, u_max, 1)
    Y = np.arange(-v_max, v_max, 1)
    X, Y = np.meshgrid(X, Y)
    
    Z = S
                    
    axis.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)

def get_harris_points(harrisim, threshold=0.65): 
    # find top corner candidates above a threshold 
    harrisim_t = (harrisim > harrisim.max()*threshold)
    # get coordinates of candidates    
    coords = np.array(harrisim_t.nonzero()).T 
    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords] 
    # sort candidates
    index = np.argsort(candidate_values) 
    # save the points
    filtered_coords = []
    for i in index: 
        filtered_coords.append(coords[i]) 
    return filtered_coords

def get_filtered_coords(values):
    coords = np.array(values.nonzero()).T 
    candidate_values = [values[c[0], c[1]] for c in coords] 
    # sort candidates
    index = np.argsort(candidate_values) 
    # save the points
    filtered_coords = []
    for i in index: 
        filtered_coords.append(coords[i])
    return filtered_coords
        
def get_point_of_interest(harrisim, point,threshold=0.65):
    top = harrisim.max()*threshold
     
    if point is ImagePoint.Corner:
        corners = (harrisim >= top)
        return get_filtered_coords(corners)[1]
    if point is ImagePoint.Flat:
        flats = (harrisim <= 10)
        flats = (flats >= 0)
        return get_filtered_coords(flats)[1]
    if point is ImagePoint.Edge:
        edges = (harrisim <= -100)
        return get_filtered_coords(edges)[1]

def plot_harris_points(image,filtered_coords,axis):
    axis.imshow(image)
    axis.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'r*') 
    axis.axis('off')

def start_harris_detection(image, axis):
    harrisim = compute_harris_response(image) 
    filtered_coords = get_harris_points(harrisim)    
    plot_harris_points(image, filtered_coords, axis)
        

def task_1():
    im = np.array(Image.open('pRect.png').convert('L'))
    rotated_im = imrotate(im, 45)
    resized_im = imresize(im, 20)

    fig, axis = plt.subplots(3, 1)
    plt.gray()
    start_harris_detection(im, axis[0])
    start_harris_detection(rotated_im, axis[1])
    start_harris_detection(resized_im, axis[2])
    
    plt.show()
    
def task_2():
    im = np.array(Image.open('pRect.png').convert('L'))
    
    harrisim = compute_harris_response(im)
    
    ax = []

    ax.append(plt.subplot2grid((3,1), (0,0), projection="3d"))
    ax.append(plt.subplot2grid((3,1), (1,0), projection="3d"))
    ax.append(plt.subplot2grid((3,1), (2,0), projection="3d"))
    
    ax[0].set_title('Corner')
    ax[0].set_axis_off()
    ax[1].set_title('Edge')
    ax[1].set_axis_off()
    ax[2].set_title('Flat')
    ax[2].set_axis_off()


    point = get_point_of_interest(harrisim, ImagePoint.Corner)
    calculate_s_function(im, point[0],point[1], 3,3, ax[0])
    
    point = get_point_of_interest(harrisim, ImagePoint.Edge)
    calculate_s_function(im, point[0],point[1], 3,3, ax[1])
    
    point = get_point_of_interest(harrisim, ImagePoint.Flat)
    calculate_s_function(im, point[0],point[1], 3,3, ax[2])
    
    plt.gray()
    
    plt.show()

if __name__ == '__main__':
        
    task_1()
    task_2()
    