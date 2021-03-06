###Francois Marcoux marf2910
###Emmanuel Bolduc bole2202
import os

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane

PHOTOREPOSITORY = "photos\\"

def saveFilterResponse(num,den,type,show=False):
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    savePNG(amp,w, f"Amplitude {type}",angle = False,show=True)
    savePNG(angle,w, f"Angle {type}",angle=True,show=False)

def filterImage(num,den,img):
    row = len(img)
    column = len(img[0])
    newImg = np.zeros((row, column))
    for i in range(0, len(img)):
        originale = img[i]
        filteredLine = signal.lfilter(num, den, originale)
        newImg[i] = filteredLine
    return newImg

def savePNG(data,xaxis,name,angle,show =False,):
    plt.plot(xaxis,data)
    plt.savefig(PHOTOREPOSITORY+name)
    if (angle):
        plt.ylabel("Angle (rad)")
    else:
        plt.ylabel("Gain (dB)")
    if(show):
        plt.show()
    plt.close()


def abberation(img):
    pass
    zero = [-0.9*np.exp(np.pi/2j),-0.9*np.exp(-np.pi/2j),-0.95*np.exp(7*np.pi/8j),-0.9*np.exp(-7*np.pi/8j)]
    pole = [-1,-0.98,0,0.8]

    num = np.poly(pole)
    den = np.poly(zero)

    saveFilterResponse(num,den,"Filtre abbération")

    #zplane(num,den,filename="photos\\ZplaneFiltreAbberation.png")
    zplane(num,den)

    newImg = filterImage(num,den,img)
    matplotlib.image.imsave("goldhillFinale\\abberationEnleve.png", arr=newImg)

    plt.imshow(img)
    plt.title("Image avec aberrations")
    plt.show()
    plt.imshow(newImg)
    plt.title("Image sans abbberations")
    plt.show()

    return newImg

def rotation(img):

    nbRow = len(img)
    nbColumn = len(img[0])
    newImg = np.zeros((nbColumn, nbRow)) #le nombre de ligne  est le nombre de colonnes après la rotation
    matriceRotation = [[0, 1], [-1, 0]]
    for i in range(0,nbRow):
        for j in range(0,nbColumn):
            coordonnex = -nbRow+1+i # le point (0,0)  de l'image revient au coordonées (-N,M) dans le plan E (N = nbColonne M =nombre de ligne)
            coordonney = nbColumn-1-j
            coordonne = np.transpose([coordonnex,coordonney])
            x,y = np.matmul(matriceRotation,coordonne) #calcul des nouvelles coordonnées avec les coordonnés transformées
            newImg[x][y] = img[coordonnex][coordonney]
    plt.gray()
    matplotlib.image.imsave("goldhillFinale\\rotation.png", arr=newImg)

    plt.imshow(img)
    plt.title("Image avant rotation")
    plt.show()
    plt.imshow(newImg)
    plt.title("Image après rotation")
    plt.show()

    return newImg

def filtreHauteFrequence(img):
    #a la main avec wc = 4789
    # utilise ce lien la pour voir la demarche wolfram alpha : https://www.wolframalpha.com/input?i=1%2F+%28%28%28+3200+*+%28z-1%29%2F%28z%2B1%29+%29%5E2%29%2F%284789.14%29%5E2+%2B+%28sqrt%282%29%2F%284789.14%29+*%283200+*+%28z-1%29%2F%28z%2B1%29%29%29+%2B+1%29
    num = np.array([1, 2,1])  #z^2 +2z +1
    den = np.array([2.3914 , -1.107 , 0.5015]) #2.3914z^2 -1.107z + 0.5015

    # zplane(num,den,filename="photos\\ZplaneHauteFreqBilineaire")
    zplane(num, den)

    saveFilterResponse(num,den,"Filtre haute frequence bilineaire",show=True)

    imageFiltrerMain = filterImage(num,den,img)
    matplotlib.image.imsave("goldhillFinale\\ImgFiltreBilinéaire.png", arr=imageFiltrerMain)


    ordre,fc,type = KeepLowerOrdre()
    print(f"Le type de filtre retenu est  : {type} avec une ordre de {ordre} et une frequence critique de {fc}")
    num1, den1 = signal.ellip(N=ordre, Wn=fc, fs=1600, rp=0.2, rs=60)

    # zplane(num1,den1,filename="photos\\ZplaneHauteFreqPython")
    zplane(num1, den1)

    saveFilterResponse(num1,den1,"Filtre haute frequence élliptique",show=True)

    imageFiltrerPython = filterImage(num1, den1, img)
    matplotlib.image.imsave("goldhillFinale\\ImgFiltreElliptique.png", arr=imageFiltrerPython)

    plt.imshow(imageFiltrerMain)
    plt.title("Image filtrée, methode bilinéaire")
    plt.show()
    plt.imshow(imageFiltrerPython)
    plt.title("Image filtrée, filtre elliptique")
    plt.show()

    return imageFiltrerPython

def compression(img, percent):
    covariance = np.cov(img)
    values, vector = np.linalg.eig(covariance)
    passage = np.transpose(vector)
    img_passee = np.matmul(passage,img)
    for lines in range(0,len(img)):
        if lines>(len(img)*(1-percent)):
            for j in range(0,len(img)):
                img_passee[lines,j]=0

    matplotlib.image.imsave("goldhillFinale\\compresse.png", arr=img_passee)
    output = np.matmul(np.linalg.inv(passage),img_passee)
    matplotlib.image.imsave("goldhillFinale\\decompresse.png", arr=output)
    plt.imshow(output)
    plt.title(f"Image compressée de {percent*100}%")
    plt.show()
    return

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
    try:
        os.mkdir("photos")
        os.mkdir("goldhillFinale")
    finally:
        plt.gray()
        img= np.load("goldhillInit\\image_complete.npy")
        img = abberation(img)
        img = rotation(img)
        img=filtreHauteFrequence(img)
        compression(img,0.5)
        compression(img,0.7)
