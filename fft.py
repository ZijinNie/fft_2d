import sys
import getopt 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from image_dft import *
import csv
import os
from PIL import Image

def main(argv):
# option handler   
#------------------------------------------------------------------------------
    mode = 1
    image = 'moonlanding.png'
    if len(argv) == 0:
        doDefault(image)

    try:
        opts,args = getopt.getopt(argv,'m:i')
    except getopt.GetoptError:
        print("ERROR\tWRONG COMMAND")
        print("python fft.py [-m mode] [-i image]")
        sys.exit(2)


    try: 

        for opt,arg in opts:
            if opt == '-m':
                mode = int(arg)
            elif opt == '-i':
                image = str(arg)

            else:
                print("ERROR\tWRONG COMMAND")
                print("python fft.py [-m mode] [-i image]")
                sys.exit(2)
    except ValueError:
        print("Incomplete Command!")
        print("python fft.py [-m mode] [-i image]")
        sys.exit(2)
    if(mode == 1):
        doDefault(image)
    elif(mode == 2):
        doDenoise(image)
    elif(mode == 3):
        doCompression(image)
    elif(mode == 4):
        showPlot()
    else:
        print("Incomplete Command!")
        print("python fft.py [-m mode] [-i image]")
        sys.exit(2)

def doDefault(dataPath):
    img = image.imread(dataPath)
    # (M,N) = img.shape
    img = img * 255
    img_padded = padding_zeros(img)
    # (X,Y) = img_padded.shape
    my_fft_padded = abs(FFT_SHIFT(FFT2D(img_padded)))
    # my_fft = my_fft_padded[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2]
    # np_sol = abs(np.fft.fftshift(np.fft.fft2(img)))
    plt.subplot(1,2,1)
    plt.title('origin')
    plt.imshow(img_padded, plt.cm.gray)
    plt.subplot(1,2,2)
    plt.title('my FFT2D')
    plt.imshow(np.log(1+my_fft_padded), plt.cm.gray)
    plt.show()

def FFT2D_n(img):
    return np.fft.fft2(img)

def doDenoise(dataPath):
    image_ml = image.imread(dataPath)
    M,N = image_ml.shape
    image_ml = image_ml * 255
    image_padded = padding_zeros(image_ml)

    my_fft = FFT_SHIFT(FFT2D(image_padded))
    denoised_padded = denoise(my_fft)
    denoised_padded = np.abs(IFFT2D(denoised_padded))
    X,Y = denoised_padded.shape
    denoised = denoised_padded[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2]

    plt.subplot(1,2,1)
    plt.title('origin')
    plt.imshow(image_ml, plt.cm.gray)
    plt.subplot(1,2,2)
    plt.title('denoised')
    plt.imshow(denoised,plt.cm.gray)
    plt.show()

def doCompression(dataPath):
    imageName = dataPath
    img = image.imread(dataPath)
    img = img * 255  
    M,N = img.shape

    img = padding_zeros(img)
    X,Y = img.shape

    img = FFT_SHIFT(FFT2D(img))

    result_0 = compress(img,imageName,0)
    path = 'compressing_{}_0.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_0, fmt = '%.3e', delimiter=",")
    result_0 = np.abs(IFFT2D(result_0)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_0.max() * (result_0 - result_0.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_0.png'.format(imageName))

    result_20 = compress(img,imageName,0.2)
    path = 'compressing_{}_20.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_20, fmt = '%.3e', delimiter=",")
    result_20 = np.abs(IFFT2D(result_20)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_20.max() * (result_20 - result_20.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_20.png'.format(imageName))

    result_40 = compress(img,imageName,0.4)
    path = 'compressing_{}_40.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_40, fmt = '%.3e', delimiter=",")
    result_40 = np.abs(IFFT2D(result_40)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_40.max() * (result_40 - result_40.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_40.png'.format(imageName))

    result_60 = compress(img,imageName,0.6)
    path = 'compressing_{}_60.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_60, fmt = '%.3e', delimiter=",")
    result_60 = np.abs(IFFT2D(result_60)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_60.max() * (result_60 - result_60.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_60.png'.format(imageName))

    result_80 = compress(img,imageName,0.8)
    path = 'compressing_{}_80.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_80, fmt = '%.3e', delimiter=",")
    result_80 = np.abs(IFFT2D(result_80)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_80.max() * (result_80 - result_80.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_80.png'.format(imageName))

    result_95 = compress(img,imageName,0.95)
    path = 'compressing_{}_95.csv'.format(imageName)
    with open(path, 'w') as csvfile:
        pass
    np.savetxt(path, result_95, fmt = '%.3e', delimiter=",")
    result_95 = np.abs(IFFT2D(result_95)[X//2-M//2 : X//2+M//2 , Y//2 - N//2: Y//2+N//2])
    rescaled = (255.0 / result_95.max() * (result_95 - result_95.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save('compressed_{}_95.png'.format(imageName))

    plt.subplot(2,3,1)
    plt.title('origin')
    plt.imshow(result_0,plt.cm.gray)
    plt.subplot(2,3,2)
    plt.title('20%')
    plt.imshow(result_20,plt.cm.gray)
    plt.subplot(2,3,3)
    plt.title('40%')
    plt.imshow(result_40,plt.cm.gray)
    plt.subplot(2,3,4)
    plt.title('60%')
    plt.imshow(result_60,plt.cm.gray)
    plt.subplot(2,3,5)
    plt.title('80%')
    plt.imshow(result_80,plt.cm.gray)
    plt.subplot(2,3,6)
    plt.title('95%')
    plt.imshow(result_95,plt.cm.gray)
    plt.show()

    
def showPlot():
    return 0

if __name__=="__main__":
    main(sys.argv[1:])


