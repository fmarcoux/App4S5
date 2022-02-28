import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

PHOTOREPOSITORY = "photos\\"



def savePNG(data,name):
    plt.plot(data)
    plt.savefig(PHOTOREPOSITORY+name)
    plt.close()


def abberation():
    pass
    zero = [-0.9*np.exp(np.pi/2j),-0.9*np.exp(-np.pi/2j),-0.95*np.exp(7*np.pi/8j),-0.9*np.exp(-7*np.pi/8j)]
    pole = [-1,-0.98,0,0.8]
    #
    num = np.poly(pole)
    den = np.poly(zero)

    zplane(num,den)

    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    savePNG(amp,"Amplitude Filtre aberation")
    savePNG(angle,"Angle Filtre aberation")

    plt.gray()
    img = np.load("goldhill_aberrations.npy")

    row = len(img)
    column = len(img[0])
    newImg = np.zeros((row, column))
    for i in range(0, len(img)):
        originale = img[i]
        filteredLine = signal.lfilter(num, den, originale)
        newImg[i] = filteredLine
    return newImg

    matplotlib.image.imsave("abberation.png", arr=imageFiltrer)


def rotation():
    pass

def filtreHauteFrequence():
    pass

def compression():
    pass


if __name__ == "__main__":
    abberation()
    rotation()
    filtreHauteFrequence()
    compression()
