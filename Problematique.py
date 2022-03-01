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

    #zplane(num,den)

    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    #savePNG(amp,"Amplitude Filtre aberation")
    #savePNG(angle,"Angle Filtre aberation")

    plt.gray()
    img = np.load("goldhillInit\\goldhill_aberrations.npy")

    row = len(img)
    column = len(img[0])
    newImg = np.zeros((row, column))
    for i in range(0, len(img)):
        originale = img[i]
        filteredLine = signal.lfilter(num, den, originale)
        newImg[i] = filteredLine

    matplotlib.image.imsave("goldhillFinale\\abberationEnleve.png", arr=newImg)
    return newImg

def rotation(img):
    nbRow = len(img)
    nbColumn = len(img[0])
    newImg = np.zeros((nbColumn, nbRow)) #puisquon tourne de 90 degree a droite,
    # nbLigne = column et nbColumn = ligne
    matriceRotation = [[0, 1], [-1, 0]]
    for i in range(0,nbRow):
        for j in range(0,nbColumn):
            coordonnex = -nbRow+1+i # le point (0,0)  de l'image revien au coordonées (-N,M) dans le plan E (N = nbColonne M =nombre de ligne)
            coordonney = nbColumn-1-j
            coordonne = np.transpose([coordonnex,coordonney])
            x,y = np.matmul(matriceRotation,coordonne) #calcul des nouvelles coordonnées avec les coordonnés transformées
            newImg[x][y] = img[i][j] #on utilise le point i,j de l'image puisque les coordonnées transformées
            # sont utiles seulement pour calculer les nouvelles coordonnées
    plt.gray()
    matplotlib.image.imsave("goldhillFinale\\rotation.png", arr=newImg)
    return newImg

def filtreHauteFrequence():
    pass

def compression():
    pass


if __name__ == "__main__":
    #abberation()
    #rotation(matplotlib.image.imread("goldhillInit\\goldhill.png"))
    filtreHauteFrequence()
    compression()
