import sys
import getopt 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from image_dft import *

def main(argv):
    print(argv)
 
# option handler   
#------------------------------------------------------------------------------
    if len(argv) == 0:
        doDefault()

    try:
        opts,args = getopt.getopt(argv,'1:2:3:4:n:')
    except getopt.GetoptError:
        print("ERROR\tWRONG COMMAND")
        print("python DnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name")
        sys.exit(2)
    try:    
        for opt,arg in opts:
            if opt == '-t':
                timeoutin = int(arg)
            elif opt == '-r':
                timeout_limit = int(arg)
            elif opt == '-p':
                portin = int(arg)
            elif opt == '-m' and arg == 'x': 
                typein = 'MX'
                break
            elif opt == '-n' and arg =='s':
                typein = "NS"
                break
            else:
                print("ERROR\tWRONG COMMAND")
                print("python DnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name")
                sys.exit(2)
    except ValueError:
        print("Incomplete Command!")
        print("python DnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name")

        sys.exit(2)

def doDefault():
    print('123')
    dataPath = "moonlanding.png"
    img = image.imread(dataPath)
    img = img * 255
    img = padding_zeros(img)
    # cv2.imshow('image',img)
    my_fft = abs(FFT_SHIFT(FFT2D(img)))

    plt.subplot(1,2,1)
    plt.title('origin')
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title('my FFT2D')
    plt.imshow(np.log(1+my_fft))
    plt.show()
if __name__=="__main__":
    main(sys.argv[1:])


