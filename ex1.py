import matplotlib
import os
from scipy.fftpack import dct, idct
from math import log
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr
import numpy as np
from tabulate import tabulate
RGB2YCBCR=np.array([[0.299,0.587,0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
YCBCR2RGB=np.linalg.inv(RGB2YCBCR)


def ex1():
    images=['logo']# ['peppers'] # ['barn_mountains']#,'logo',
    qualities = [75,50,25]
    rates=[]
    for s in images:
        img = Image.open(f"imagens/{s}.bmp")
        rgb_img = img.convert('RGB')
        arr=[s]
        plt.figure()
        plt.title(f'{s}.bmp')
        plt.imshow(img)
        original_size = os.path.getsize(f"imagens/{s}.bmp")

        for q in qualities:
            rgb_img.save(f"ex1/{s}_{q}.jpeg", quality=q)
            im2 = Image.open(f"ex1/{s}_{q}.jpeg")
            plt.figure()
            plt.title(f'{s}_{q}.jpeg')
            plt.imshow(im2)
            plt.show(block=False)
            compressed_size = os.path.getsize(f"ex1/{s}_{q}.jpeg")
            comp_rate = "%.1f:1" %(original_size/compressed_size)
            arr.append(comp_rate)
        rates.append(arr)

    # print(46*"_")
    print(tabulate(rates, headers = ['Image\\Quality',"75%","50%","25%"] ))
            # print(f"{s} with quality {q}: compression rate {comp_rate}%")

def separate_3channels(image):
    return (image[:,:,0], image[:,:,1], image[:,:,2])

def join_3channels(r,g,b):
    return  np.dstack((r,g,b))

def color_map(color_map_name, min_color=(0,0,0), max_color = (1,1,1)) :
    return clr.LinearSegmentedColormap.from_list(color_map_name, [min_color,max_color], 256)

def view_image(img, color_map,title="<untitled>"):
    plt.figure()
    plt.title(title)
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

def ycbcr2rgb(img):
    img[:,:,1:3] -= 128
    recovered = img.dot(YCBCR2RGB.T)
    recovered[recovered < 0]=0
    recovered[recovered > 255]=255
    recovered= np.round(recovered)
    return recovered.astype(np.uint8)

def rbg2ycbcr(img):
    ycc= img.dot(RGB2YCBCR.T)
    ycc[:,:,1:3] += 128
    return ycc

def cv2_downsample(y,cr,cb, comp_ratio):
    sh_cr= cr.shape
    sh_cb= cb.shape
    if comp_ratio[2]!= 0: #horizontal only
        cr_ratio= comp_ratio[0]//comp_ratio[1]
        cb_ratio= comp_ratio[0]//comp_ratio[2]
        sh_cb[1] = sh_cb[1]/cb_ratio
        sh_cr[1] = sh_cr[1]/cr_ratio
        cr_d = cv2.resize(cr, sh_cr, interpolation = cv2.INTER_AREA)
        cb_d = cv2.resize(cb, sh_cb, interpolation = cv2.INTER_AREA)
        
    else:
        cb_ratio=cr_ratio=comp_ratio[0]//comp_ratio[1]
        sh_cb[0] = sh_cb[0]/cb_ratio
        sh_cb[1] = sh_cb[1]/cb_ratio
        sh_cr[0] = sh_cr[0]/cb_ratio
        sh_cr[1] = sh_cr[1]/cb_ratio
    
        cr_d = cv2.resize(cr, sh_cr, interpolation = cv2.INTER_AREA)
        cb_d = cv2.resize(cb, sh_cb, interpolation = cv2.INTER_AREA)
    return y, cr_d, cb_d

def cv2_upsampling(y_d, cr_d, cb_d, comp_ratio):
    sh_y= y.shape
    cr = cv2.resize(cr_d, sh_y, interpolation = cv2.INTER_AREA)
    cb = cv2.resize(cb_d, sh_y, interpolation = cv2.INTER_AREA)
    return y_d, cr,cb

def ycrcb_downsampling(y,cr,cb, comp_ratio): # comp_ratio is a tuple with 3 values, such as (4,2,2)
    cr_d=cb_d=np.array([])
    
    if comp_ratio[2]!= 0: #horizontal only
        cr_ratio= comp_ratio[0]//comp_ratio[1]
        cr_d = cr[:,::cr_ratio]
        cb_ratio= comp_ratio[0]//comp_ratio[2]
        cb_d = cb[:,::cb_ratio]
    else:
        cb_ratio=cr_ratio=comp_ratio[0]//comp_ratio[1]
        cr_d = cr[::cr_ratio,::cr_ratio]
        cb_d = cb[:: cb_ratio,::cb_ratio]
    return y, cr_d, cb_d

def ycrcb_upsampling(y_d:np.array,cr_d:np.array, cb_d:np.array, comp_ratio: tuple)-> np.array:
    cb_shape= cb_d.shape
    cr_shape= cr_d.shape
    cr=cb=np.array([])
    if comp_ratio[2]!= 0: #horizontal only
        cr_ratio= comp_ratio[0]//comp_ratio[1]
        cb_ratio= comp_ratio[0]//comp_ratio[2]
        print("cb_ratio: ",cb_ratio)
        print("cr_ratio: ",cr_ratio)
        cb= np.repeat(cb_d, cb_ratio).reshape(cb_shape[0], cb_shape[1]*cb_ratio)
        #cb[:,::cb_ratio]= cb_d[:,:]
        cr= np.repeat(cr_d, cr_ratio).reshape(cr_shape[0], cr_shape[1]*cr_ratio)
        return y_d, cr,cb

    else:
        cb_ratio = cr_ratio = comp_ratio[0]//comp_ratio[1]
        print("cb_ratio: ",cb_ratio)
        print("cr_ratio: ",cr_ratio)
        cb= np.repeat(cb_d, cb_ratio).reshape(cb_shape[0], cb_shape[1]*cb_ratio)
        cb_shape= cb.shape
        cb= np.repeat(cb.T, cb_ratio).reshape(cb_shape[1], cb_shape[0]*cb_ratio).T
        
        cr= np.repeat(cr_d, cr_ratio).reshape(cr_shape[0], cr_shape[1]*cr_ratio)
        cr_shape= cr.shape
        cr= np.repeat(cr.T, cr_ratio).reshape(cr_shape[1], cr_shape[0]*cr_ratio).T
        return y_d, cr,cb

def dct_channel(channel):
    return dct(dct(channel, norm="ortho").T, norm='ortho').T

def idct_channel(dct_channel_arr):
    return idct(idct(dct_channel_arr, norm="ortho").T, norm="ortho").T



def dct_8x8(channel, block_size):
    n_array= np.zeros(channel.shape)

    n_array[]

def encode(img_name, ds_rate: tuple) -> None:
    img= plt.imread(img_name)
    # plt.figure()
    # plt.imshow(img)
    
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
    
    min_cb = (0.5,0.5,0)
    max_cb = (0.5,0.5,1)
    min_cr = (0,0.5,0.5)
    max_cr = (1,0.5,0.5)
    cmChromBlue = color_map('myCb', tuple(min_cb),  tuple(max_cb) )
    cmChromRed = color_map('myCr', tuple(min_cr),  tuple(max_cr) )
    r,g,b = separate_3channels(img_padded)
    chromin_image = rbg2ycbcr(img_padded)
    y, cb,cr = separate_3channels(chromin_image)

    d = {'red':cmRed, 'green': cmGreen, 'blue': cmBlue, 'gray': cmGray,  'chromBlue':cmChromBlue, 'chromRed':cmChromRed }
    e = {'red': r, 'green':g, 'blue':b, 'gray': y, 'chromBlue':cb, 'chromRed':cr}
    # for col in d.keys():
    #     view_image(e[col],d[col])
    print("coisas a acontecer aqui")

    # ds_rate = (4,2,0)
    (y_d,cr_d,cb_d) = ycrcb_downsampling(y,cr,cb, ds_rate)

    
    
    d = {'gray': cmGray, 'chromBlue':cmGray, 'chromRed':cmGray }
    e = { 'gray': y, 'chromBlue':cr_d, 'chromRed':cb_d}

    for col in d.keys():
        view_image(e[col],d[col])

    # return chromin_image, original_shape
    dct_y = dct_channel(y_d)
    dct_cb = dct_channel(cb_d)
    dct_cr = dct_channel(cr_d)
    dcts= {"y":dct_y,"cb":dct_cb,"cr":dct_cr}
    # for name, channel in dcts.items():
    #     fig = plt.figure()
    #     plt.title(f"{name} dct - log(x+0.0001)")
    #     # cax = plt.axes()
    #     # a = np.linspace(np.min(channel),np.max(channel),6)
    #     sh = plt.imshow(np.log(np.abs(channel) + 0.0001))
    #     fig.colorbar(sh)
    #     plt.show(block=False)
       
    return dct_y,dct_cb,dct_cr, original_shape

def decode(dct_y,dct_cb,dct_cr, ds_ratio, original_shape):
    y_d = idct_channel(dct_y) 
    cb_d = idct_channel(dct_cb) 
    cr_d = idct_channel(dct_cr) 
    dcts= {"y":y_d,"cb":cb_d,"cr":cr_d}
    min_cb = (0.5,0.5,0)
    max_cb = (0.5,0.5,1)
    min_cr = (0,0.5,0.5)
    max_cr = (1,0.5,0.5)
    cmGray = color_map('myGray', (0,0,0),(1,1,1) )
    cmChromBlue = color_map('myCb', tuple(min_cb),  tuple(max_cb) )
    cmChromRed = color_map('myCr', tuple(min_cr),  tuple(max_cr) )
    for name, channel in dcts.items():
        fig = plt.figure()
        plt.title(f"{name} downslampled restored")
        # cax = plt.axes()
        # a = np.linspace(np.min(channel),np.max(channel),6)
        plt.imshow(channel, cmGray)

        plt.show(block=False)
    
    y_u,cr_u, cb_u = ycrcb_upsampling(y_d,cr_d, cb_d, ds_ratio)
    print("cr_u.shape= ",cr_u.shape)
    print("cb_u.shape= ",cb_u.shape)
    d = {'chromBlue':cmChromBlue, 'chromRed':cmChromRed }
    e = {'chromBlue':cb_d, 'chromRed':cr_d}
    # for col in d.keys():
    #     view_image(e[col],d[col])
    d = {'gray': cmGray, 'chromBlue':cmChromBlue, 'chromRed':cmChromRed }
    e = { 'gray': y_u, 'chromBlue':cb_u, 'chromRed':cr_u}
    # for col in d.keys():
    #     view_image(e[col],d[col])
    encoded= join_3channels(y_u,cb_u,cr_u)

    inverse_chromin = ycbcr2rgb(encoded)
    plt.figure()
    plt.title('depois de ycbcr e da inversão')
    plt.imshow(inverse_chromin)
    plt.show(block=False)

    img = image_remove_padding(inverse_chromin, original_shape)
    plt.figure()
    plt.title('sem padding')
    plt.imshow(img)
    plt.show(block=False)

    return img



def main():
    # plt.close('all')
    ex1()
    ds_ratio = (4,2,2)
    dct_y, dct_cb, dct_cr, original_shape = encode('imagens/barn_mountains_2.bmp', ds_ratio)
    decoded= decode(dct_y, dct_cb, dct_cr, ds_ratio, original_shape)
    
    a=input()






if __name__ == "__main__":
    ex1()
    a=input()
    # main()