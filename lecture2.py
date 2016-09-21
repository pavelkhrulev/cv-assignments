from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.misc, scipy.io

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

def get_random_image_part(image):
    raise NotImplementedError

def getFig():
    fig = plt.figure()
    ax = []

    ax.append(plt.subplot2grid((3,5), (0,0), rowspan=3,colspan=3))
    ax.append(plt.subplot2grid((3,5), (0,3)))
    ax.append(plt.subplot2grid((3,5), (0,4)))
    ax.append(plt.subplot2grid((3,5), (1,3)))
    ax.append(plt.subplot2grid((3,5), (1,4)))
    ax.append(plt.subplot2grid((3,5), (2,3)))
    ax.append(plt.subplot2grid((3,5), (2,4)))
    return fig,ax

def doubleImageRecuder(im):
    image = np.array(im)
    print image.shape

    image = im.resize((image.shape[1] / 2, image.shape[0] / 2))
    return image

def RunAssignment():
    fig, ax = getFig()
    
    face = Image.fromarray(scipy.misc.face()) 
    miet = Image.open('miet.jpeg')
    
    main_image = stack_images([miet, face])

    square_side = 200
    im1_x = 0
    im1_y = -400
    im2_x = -1000
    im2_y = 0
    
    im1 = Image.new('RGB', (square_side, square_side))
    im1.paste(main_image, (im1_x,im1_y))
    im2 = Image.new('RGB', (square_side, square_side))
    im2.paste(main_image, (im2_x,im2_y))
    
    dr = ImageDraw.Draw(main_image)
    
    dr.rectangle([(im1_x, im1_y*(-1)) ,(im1_x+square_side, im1_y*(-1)+square_side)], 'red', 'red')
    dr.rectangle([(im2_x*(-1), im2_y) ,(im2_x*(-1)+square_side, im2_y+square_side)], 'red', 'red')
    
    font = ImageFont.truetype('test.ttf', 40)
    
    im1_arr = np.array(im1)
    im2_arr = np.array(im2)

    mean1 = im1_arr[:,:].mean()
    mean2 = im2_arr[:,:].mean()
    
    dr.text((im1_x, im1_y*(-1)), str(mean1) , (255,255,255), font)    
    dr.text((im2_x*(-1), im2_y), str(mean2) , (255,255,255), font)    
    
    
    im3 = doubleImageRecuder(main_image);
    im4 = doubleImageRecuder(im3);
    im5 = doubleImageRecuder(im4);
    im6 = doubleImageRecuder(im5);
    
    file = open('image_backup.pkl','rb')
    im = pickle.load(file)
    file.close()
        
    ax[0].imshow(main_image)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[1].imshow(im1)
    ax[2].axis('off')
    ax[2].imshow(im2)
    ax[3].axis('off')
    ax[3].imshow(im3)
    ax[4].axis('off')
    ax[4].imshow(im4)
    ax[5].axis('off')
    ax[5].imshow(im5)
    ax[6].axis('off')
    ax[6].imshow(im6)
    
    backup_objects = [main_image, mean1, mean2, im1, im2]
        
    file = open('image_backup.pkl','wb')
    pickle.dump(backup_objects, file)
    file.close()
    
        
    plt.savefig("image.jpg")
    
    plt.show()
    
if __name__ == '__main__':
    RunAssignment()