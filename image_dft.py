import numpy as np
import matplotlib.pyplot as plt

def signal(sinuses):
    ''' This function creates a function f(x) = sin(a*pi*2)+sin(b*p*2)+...
     plots the graph of f(x) and saves it
    Args:
        sinuses(Float) : The list of constants a,b,c,...
    Returns:
        t (numpy array): X axis
        s (numpy array): Y axis
    '''
    # Create the singal as a sum of different sinuses
    t = np.linspace(0, 0.5, 800)
    s=0
    for i in range(len(sinuses)):
        s += np.sin(sinuses[i] * 2 * np.pi * t)

    # Plot the signal
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)

    ax.plot(t, s, label = r'$y=f(x)$')

    ax.set_title(" Signal ", fontsize = 20)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time [s]")

    ax.legend(loc='best')
    ax.grid(True)
    plt.show()
    fig.savefig('signal.png')
    return s, t

def Fourier(s, t, alg = "False"):
    '''This function performs the DFT in the function f(x) defined in signal function, plots and saves the figure
    Args:
        t (numpy array): X axis
        s (numpy array): Y axis
        alg (string): Variable to determine whether to use my DFT or numpy's
    '''

    #Perform the Fourier Transform
    if alg == "True":
        fft = DFT(s)
    else:
        fft = np.fft.fft(s)

    T = t[1] - t[0]  # sample rate
    N = s.size

    # 1/T = frequency
    f = np.linspace(0, 1 / T, N)

    # Plot the signal Decomposition
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)

    ax.set_title(" Decomposed Signal ", fontsize = 20)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency [Hz]")

    # ax.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
    ax.bar(f[:N // 2], np.abs(fft)[:N // 2] , width=1.5)  # 1 
    ax.grid(False)
    plt.show()
    fig.savefig("Decomposed_signal.png")


def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-1j*2*np.pi*v*y/N) for v in range(N)] for y in range(N)])
    return sig.dot(V)

def IDFT(sig):
    N = sig.size
    V = np.array([[np.exp(1j*2*np.pi*v*y/N) for v in range(N)] for y in range(N)]) 
    return sig.dot(V) / N
    # return sig.dot(V)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
  
    if N % 2 > 0:
      raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
      return DFT(x)
    else:
      X_even = FFT(x[::2])
      X_odd = FFT(x[1::2])
      factor = np.exp(-2j * np.pi * np.arange(N) / N)
      print(N)
      return np.concatenate([X_even + factor[:N // 2] * X_odd,
                            X_even + factor[N // 2:] * X_odd])

def IFFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
  
    if N % 2 > 0:
      raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
      return IDFT(x)
    else:
      X_even = IFFT(x[::2])
      X_odd = IFFT(x[1::2])
      factor = np.exp(2j * np.pi * np.arange(N) / N)
      print(N)
      return np.concatenate([X_even + factor[:N // 2] * X_odd,
                           X_even + factor[N // 2:] * X_odd])


def checkValidInput(x):
    x.reshape((x.shape[0],1))
    N = x.shape[0]
    l = 1
    r = 0
    while(N > l):
      l = l*2 
      r = r+1
    if(N != l):
      arr = np.zeros((l,1))
      arr = np.asarray(arr, dtype = float)
      arr = np.transpose(arr)
      arr= arr[0]
      arr[0:N] = x
      return arr
    return x

def twoDDFT(x):
    result = np.empty([x.shape[0],x.shape[1]], np.complex)

    x = np.asarray(x, dtype=float)
    M = x.shape[0]

    N = x.shape[1]

    for k in range(M):
      for j in range(N):
          sum = 0
          for n in range (N):
            for m in range (M):
              sum += x[m,n] * np.exp(-2j * np.pi * (float(k * m) / M + float(j * n)/N ))
          
          result[k][j] = sum

    return result
    


def FFT2(x):
    N = x.shape[1] # 只需考虑第二个维度，然后在第一个维度循环
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized
        return np.array([DFT(x[i,:]) for i in range(x.shape[0])])
    else:
        X_even = FFT2(x[:,::2])
        X_odd = FFT2(x[:,1::2])
        factor = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:,:int(N/2)],X_odd),
                               X_even + np.multiply(factor[:,int(N/2):],X_odd)])
def IFFT2(x):
    N = x.shape[1] # 只需考虑第二个维度，然后在第一个维度循环
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized

        row_element_size = x[1,:].size
        V = np.array([[np.exp(1j*2*np.pi*v*y/row_element_size) for v in range(N)] for y in range(N)]) 

        return np.array([x[i,:].dot(V) for i in range(x.shape[0])])
    else:
        X_even = IFFT2(x[:,::2])
        X_odd = IFFT2(x[:,1::2])
        factor = np.array([np.exp(2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:,:int(N/2)],X_odd),
                               X_even + np.multiply(factor[:,int(N/2):],X_odd)])

def FFT2D(img):
    return FFT2(FFT2(img).T).T

def IFFT2D(img):
    return IFFT2(IFFT2(img).T).T / img.shape[0] / img.shape[1]

def FFT_SHIFT(img):
    M,N = img.shape
    M = int(M/2)
    N = int(N/2)
    return np.vstack((np.hstack((img[M:,N:],img[M:,:N])),np.hstack((img[:M,N:],img[:M,:N]))))

def denoise(img, percentile = 0.25):

    (M, N) = img.shape
    transformed  = img.copy()
    for i in range(M):
      for j in range(N):
        if i < M * (0.5 - percentile /2 ) or i > M * (0.5 + percentile /2) :
          transformed[i][j] = 0
        else:
          if  j < N * (0.5 - percentile /2 ) or j > N * (0.5 + percentile /2) :
            transformed[i][j] = 0
    return transformed

def padding_zeros(img):
    M,N = img.shape
    L = 1
    l = 0
    R = 1
    r = 0
    while(M > L):
      L = L*2 
      l = l+1
    while(N > R):
      R = R * 2
      r = r +1
    if(M == L & N == R ):
      return img
    else:
      result = np.zeros((L,R))
      for i in range(M) :
        for j in range(N):
          result[(L-M)//2 + i][(R - N)//2 + j] = img[i][j]
      return result