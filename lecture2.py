from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.misc, scipy.io as sio
from random import randint

# Use this class instead of enumeration.
class ImageOrientation:
    Horizontal = 1
    Vertical = 2

# Images should be opened already.
def stack_images(images, orientation=ImageOrientation.Horizontal):
    widths, heights = zip(*(i.size for i in images))
    
    if orientation is ImageOrientation.Horizontal:
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for current_image in images:
            # Bad proportions resizing, but now it's ok.
            current_image = current_image.resize((current_image.size[0], max_height))
            new_image.paste(current_image, (x_offset,0))
            x_offset += current_image.size[0]
        return new_image
    
    if orientation is ImageOrientation.Vertical:
        raise NotImplementedError

def get_random_image_part(image, square_side):
    width = image.size[0]
    height = image.size[1]
    left = randint(0, width - square_side)
    top = randint(0, height - square_side)
    image_part = Image.new('RGB', (square_side, square_side))
    box = (left, top, left+square_side, top+square_side)
    region = image.crop(box)
    image_part.paste(region)
    drawer = ImageDraw.Draw(image)
    drawer.rectangle(box, 'red', 'red')
    # Writing mean value.
    image_part_data = np.array(image_part)
    mean = round(image_part_data[:,:].mean(), 2)
    font = ImageFont.truetype('Fonts/test.ttf', 40)
    drawer.text((left, top), str(mean), (255,255,255), font)
    
    return image_part

def get_axes():
    ax = []

    ax.append(plt.subplot2grid((3,5), (0,0), rowspan=3,colspan=3))
    ax.append(plt.subplot2grid((3,5), (0,3)))
    ax.append(plt.subplot2grid((3,5), (0,4)))
    ax.append(plt.subplot2grid((3,5), (1,3)))
    ax.append(plt.subplot2grid((3,5), (1,4)))
    ax.append(plt.subplot2grid((3,5), (2,3)))
    ax.append(plt.subplot2grid((3,5), (2,4)))
    return ax

def reduce_image_twice(image):
    reduced_image = np.array(image)
    reduced_image = image.resize((reduced_image.shape[1] / 2, reduced_image.shape[0] / 2))
    return reduced_image

def RunAssignment():
    ax = get_axes()
    
    face = Image.fromarray(scipy.misc.face()) 
    miet = Image.open('ImageFolder/miet.jpeg')
    
    main_image = stack_images([miet, face])
    
    # Images may be overlayed due to random.
    image_part_1 = get_random_image_part(main_image, 300)
    image_part_2 = get_random_image_part(main_image, 200)
    
    ax[0].axis('off')
    ax[0].imshow(main_image)
    ax[1].axis('off')
    ax[1].imshow(image_part_1)
    ax[2].axis('off')
    ax[2].imshow(image_part_2)
    
    reduced_image = main_image
    for x in range(3, 7):
        reduced_image = reduce_image_twice(reduced_image);
        ax[x].axis('off')
        ax[x].imshow(reduced_image)
    
    # Saving picture as an image.
    plt.savefig('image.jpg')
    
    # Saving the main image, its mean value and the other images.
    with open('image_backup.pkl','wb') as backup_file:
        main_image_data = np.array(main_image)
        main_image_mean = main_image_data[:,:].mean()
        pickle.dump([main_image, main_image_mean, image_part_1, image_part_2], backup_file)
    
    # Saving data into mat file.
    sio.savemat('data_backup.mat', {'main_image_data': main_image_data, 
                                    'main_image_mean': main_image_mean})
    
    # Just for testing pickle loading.
    # with open('image_backup.pkl','rb') as backup_file:
    #     [main_image, main_image_mean, image_part_1, image_part_2] = pickle.load(backup_file)
    #         
    # ax[0].axis('off')
    # ax[0].imshow(main_image)
    # ax[1].axis('off')
    # ax[1].imshow(image_part_1)
    # ax[2].axis('off')
    # ax[2].imshow(image_part_2)
    
    plt.show()
    
if __name__ == '__main__':
    RunAssignment()