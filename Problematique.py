import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

PHOTOREPOSITORY = "photos\\"


def filterImage(num,den,img):
    row = len(img)
    column = len(img[0])
    newImg = np.zeros((row, column))
    for i in range(0, len(img)):
        originale = img[i]
        filteredLine = signal.lfilter(num, den, originale)
        newImg[i] = filteredLine
    return newImg

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

    newImg = filterImage(num,den,img)

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
    #a la main avec wc = 4789
    num = np.array([1, 2,1])  #z^2 +2z +1
    den = np.array([2.39146 , -1.107 , 0.5015]) #2.39146z^2 -1.107z + 0.5015
    imageFiltrerMain = filterImage(num,den,np.load("goldhillInit\\goldhill_bruit.npy"))
    matplotlib.image.imsave("goldhillFinale\\ImgFiltreMain.png", arr=imageFiltrerMain)
    #zplane(num,den)
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    #savePNG(amp,"Amplitude Filtre haute frequence a la main")
    #savePNG(angle,"Angle Filtre haute frequence a la main")
    ordre,fc,type = KeepLowerOrdre()
    print(f"Le type de filtre retenu est  : {type} avec une ordre de {ordre} et une frequence critique de {fc}")
    num, den = signal.ellip(N=ordre, Wn=fc, fs=1600, rp=0.2, rs=60)
    imageFiltrerPython = filterImage(num, den, np.load("goldhillInit\\goldhill_bruit.npy"))
    matplotlib.image.imsave("goldhillFinale\\ImgFiltreElliptique.png", arr=imageFiltrerPython)
    #zplane(num,den)
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    #savePNG(amp, "Amplitude Filtre haute frequence eliptique")
    #savePNG(angle, "Angle Filtre haute frequence eliptique")
    return imageFiltrerPython

def compression():
    pass

def KeepLowerOrdre():
    fs = 1600
    ws = 750
    wp = 500
    gpass = 0.2
    gstop = 60
    maxripple = 0.2
    ordremin = 999
    fc = 999
    type = ""
    ordre, frequenceCritique = signal.buttord(fs=fs, wp=wp, gpass=gpass, gstop=gstop, ws=ws)
    if(ordre < ordremin):
        ordremin=ordre
        fc = frequenceCritique
        type = "butterworth"
    print("\nButter : ", ordre)
    ordre, frequenceCritique = signal.cheb1ord(fs=fs, wp=wp, gpass=gpass, gstop=gstop, ws=ws)
    if (ordre < ordremin):
        ordremin = ordre
        fc = frequenceCritique
        type = "cheb type 1"
    print("\nCheb1 : ", ordre)
    ordre, frequenceCritique = signal.cheb2ord(fs=fs, wp=wp, gpass=gpass, gstop=gstop, ws=ws)
    if (ordre < ordremin):
        ordremin = ordre
        fc = frequenceCritique
        type = "cheb type 2"
    print("\nCheb2 : ", ordre)
    ordre, frequenceCritique = signal.ellipord(fs=fs, wp=wp, gpass=gpass, gstop=gstop, ws=ws)
    if (ordre < ordremin):
        ordremin = ordre
        fc = frequenceCritique
        type = "elliptique"
    print("\nelliptique : ", ordre)
    return ordremin,fc,type


if __name__ == "__main__":
    #abberation()
    #rotation(matplotlib.image.imread("goldhillInit\\goldhill.png"))
    filtreHauteFrequence()
    compression()
