import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr
import numpy as np
RGB2YCBCR=np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
YCBCR2RGB=np.linalg.inv(RGB2YCBCR)


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

def separate_3channels(image):
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
    sh = or_shape = img.shape
    last_line = img[len(img)-1,:,:]
    if (len(img)%16!=0):   
        arr_to_add = np.tile(last_line, (16-len(img)%16,1)).reshape(16-len(img)%16,or_shape[1],3)
        img = np.vstack((img, arr_to_add))

    last_col = np.array([img[:, len(img[0])-1, :]])
    sh= img.shape
    if(sh[1]%16!=0):
        arr_to_add = np.tile(last_col, (1,16-sh[1]%16)).reshape(sh[0], 16-sh[1]%16,3)
        img = np.hstack((img, arr_to_add))

    return img, or_shape

def image_remove_padding(img, shape):
    h,c = shape[0], shape[1]
    return img[:h,:c,:]

# def rgb_to_ycbcr(img):
    trans_matrix= np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])

def ycbcr_to_rgb(img):
    img[:,:,1:3] -= 128
    recovered = img.dot(YCBCR2RGB.T)
    recovered[recovered < 0]=0
    recovered[recovered > 255]=255
    recovered= np.round(recovered)
    return recovered.astype(np.uint8)

def rbg2ycbcr(img):
  trans_matrix= np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
  ycc= img.dot(trans_matrix.T)
  ycc[:,:,1:3] += 128
  # ycc[ycc<0]=0
  # ycc[ycc>255]=255;
  return ycc;

def encode(img_name):
    img= plt.imread(img_name)
    plt.figure()
    plt.imshow(img)
    
    print(img.shape)  # dimensions
    print("image: ",img)
    #R= img[:,:,0] # red channel
    #print(R.shape) 
    print(img.dtype)  # data typr

    img_padded , original_shape = image_padding(img)

    cmRed = color_map('myRed', (0,0,0),(1,0,0))
    cmGreen = color_map('myGreen', (0,0,0),(0,1,0) )
    cmBlue = color_map( 'myBlue', (0,0,0),(0,0,1))
    cmGray = color_map('myGray', (0,0,0),(1,1,1) )
    # min_cb = [x/255 for x in np.dot(YCBCR2RGB, [0,0,0])]
    # max_cb = [x/255 for x in np.dot(YCBCR2RGB, [0,127,0])]
    # min_cr = [x/255 for x in np.dot(YCBCR2RGB, [0,0,0])]
    # max_cr = [x/255 for x in np.dot(YCBCR2RGB, [0,0,127])]
    min_cb = (0.5,0.5,0)
    max_cb = (0.5,0.5,1)
    min_cr = (0,0.5,0.5)
    max_cr = (1,0.5,0.5)
    cmChromBlue = color_map('myCb', tuple(min_cb),  tuple(max_cb) )
    cmChromRed = color_map('myCr', tuple(min_cr),  tuple(max_cr) )
    r,g,b = separate_3channels(img_padded)
    chromin_image = rbg2ycbcr(img_padded)
    y, cr,cb = separate_3channels(chromin_image)

    d = {'red':cmRed, 'green': cmGreen, 'blue': cmBlue, 'gray': cmGray, 'chromBlue':cmGray, 'chromRed':cmGray }
    e = {'red': r, 'green':g, 'blue':b, 'gray': y, 'chromBlue':cr, 'chromRed':cb}
    for col in d.keys():
        view_image(e[col],d[col])
    print("coisas a acontecer aqui")


    return chromin_image, original_shape
    

    


def decode(encoded, shape):
    inverse_chromin = ycbcr_to_rgb(encoded)
    plt.figure()
    plt.title('depois de ycbcr e da inversão')
    plt.imshow(inverse_chromin)
    plt.show(block=False)

    img = image_remove_padding(inverse_chromin, shape)
    plt.figure()
    plt.title('sem padding')
    plt.imshow(img)
    plt.show(block=False)

    return img



def main():
    plt.close('all')
    encoded , original_shape = encode('imagens/barn_mountains_2.bmp')
    decoded= decode(encoded, original_shape)
    
    a=input()






if __name__ == "__main__":
    # ex1()
    main()