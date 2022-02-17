import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr

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

def main():
    plt.close('all')
    img= plt.imread('imagens/peppers.bmp')
    plt.figure()
    plt.imshow(img)

    print(img.shape)  # dimensions
    R= img[:,:,0] # red channel
    print(R.shape) 
    print(img.dtype)  # data typr

    cmRed = clr.LinearSegmentedColormap.from_list('myRed', [(0,0,0),(1,0,0)], 256)
    cmGreen = clr.LinearSegmentedColormap.from_list('myGreen', [(0,0,0),(0,1,0)], 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('myBlue', [(0,0,0),(0,0,1)], 256)
    cmGray = clr.LinearSegmentedColormap.from_list('myGray', [(0,0,0),(1,1,1)], 256)

    d = {'red':cmRed, 'green': cmGreen, 'blue': cmBlue }
    e = {'red': img[:,:,0], 'green':img[:,:,1], 'blue': img[:,:,2]}
    for col in d.keys():
        plt.figure()
        plt.imshow(e[col], d[col])
        plt.show(block=False)


    a=input()






if __name__ == "__main__":
    # ex1()
    main()