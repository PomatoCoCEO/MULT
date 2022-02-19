import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr
import numpy as np


def ex1():
    images= ['barn_mountains','logo','peppers']
    qualities = [75,50,25]
    
    for s in images:
        img = Image.open(f"imagens/{s}.bmp")
        rgb_img = img.convert('RGB')
        for q in qualities:
            rgb_img.save(f"ex1/{s}_{q}.jpeg", quality=q)
            im2 = Image.open(f"ex1/{s}_{q}.jpeg")
            plt.figure()
            plt.title(f'{s}_{q}.jpeg')
            plt.imshow(im2)

def separate_rgb(image):
    return (image[:,:,0], image[:,:,1], image[:,:,2])

def join_rbg(r,g,b):
    return  np.dstack((r,g,b))

def color_map(color_map_name, min_color=(0,0,0), max_color = (1,1,1)) :
    return clr.LinearSegmentedColormap.from_list(color_map_name, [min_color,max_color], 256)

def view_image(img, color_map):
    plt.figure()
    plt.imshow(img, color_map)
    plt.show(block=False)

def image_padding(img):
    or_shape = img.shape
    last_line = img[len(img),:,:]
    while(len(img)%16!=0):
        img = np.vstack((img, [last_line]))
    last_col = img[:, len(img[0])-1, :]
    sh= img.shape
    while( sh[1]%16!=0):
        img = np.hstack((img, [last_col]))
        sh = img.shape
    return img, or_shape

def image_remove_padding(img, shape):
    h,c = shape[0], shape[1]
    return img[:h,:c,:]


def encode(img_name):
    img= plt.imread(img_name)
    plt.figure()
    plt.imshow(img)
    
    print(img.shape)  # dimensions
    #R= img[:,:,0] # red channel
    #print(R.shape) 
    print(img.dtype)  # data typr

    img_padded , original_shape = image_padding(img)

    cmRed = color_map('myRed', (0,0,0),(1,0,0))
    cmGreen = color_map('myGreen', (0,0,0),(0,1,0) )
    cmBlue = color_map( 'myBlue', (0,0,0),(0,0,1))
    cmGray = color_map('myGray', (0,0,0),(1,1,1) )

    r,g,b = separate_rgb(img_padded)
    #joined= join_rbg(r,g,b)  # to decode later
    #plt.figure()
    #plt.imshow(joined)
    d = {'red':cmRed, 'green': cmGreen, 'blue': cmBlue }
    e = {'red': r, 'green':g, 'blue':b}
    for col in d.keys():
        view_image(e[col],d[col])

    


def decode(encoded):
    pass


def main():
    plt.close('all')
    
    encode('imagens/peppers.bmp')


    a=input()






if __name__ == "__main__":
    # ex1()
    main()