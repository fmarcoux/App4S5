import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane


# Press the green button in the gutter to run the script.
def p1():
    K = 1
    num = np.poly([0.8j,-0.8j])
    den = np.poly([0.95*np.exp(np.pi/8j),0.95*np.exp(-np.pi/8j)])
    Hz = signal.TransferFunction(num,den)
    #zplane(num,den)

    #b oui le filtre est stable

    #c
    w,H = signal.freqz(num,den)
    print(H)
    amp = 20*np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence de la fonction de transfert")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Angle de la fonction de transfert")
    plt.close()

    #d
    impulsion = np.zeros(100)
    one = np.ones(1)
    impulsion = np.concatenate([impulsion,one,impulsion])
    signalFiltre = signal.lfilter(num,den,impulsion)

    reponseFrequence = np.fft.fft(signalFiltre)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\Reponse en f de limpulsion FFT")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Angle de la l'impulsion FFT")
    plt.close()


   #e le filtre qui annule lautre est l'inverse (inverse num et denominateur)

    signal2B = signal.lfilter(den,num,signalFiltre)
    reponseFrequence = np.fft.fft(signal2B)
    plt.plot(signal2B)
    plt.savefig("photos\\signal 2 b")

def p2():
    num = np.poly([np.exp(-np.pi/16j),np.exp(np.pi/16j)])
    den = np.poly([ 0.95*np.exp(-np.pi/16j),  0.95* np.exp(np.pi/16j)])

    zplane(num,den)
    w,H = signal.freqz(num,den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence de la fonction de transfert p2")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Angle de la fonction de transfert p2")
    plt.close()

    n = a=np.arange(0,600)

    sin16 = np.sin(np.pi*n/16)
    sin32 = np.sin(np.pi*n/32)
    sins = sin16+sin32

    signalfiltre = signal.lfilter(num,den,sins)
    plt.plot(signalfiltre)
    plt.show()

def p3():


    ordre,frequenceCritique = signal.buttord(fs= 48000,wp=2500,gpass=0.2,gstop=40,ws=3500)
    num,den = signal.butter(ordre,frequenceCritique,fs=48000)
    w,H = signal.freqz(num,den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence filtre butterworth")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Anglefiltre butterworth ")
    plt.close()


    order,frequenceCritique = signal.cheb1ord(fs= 48000,wp=2500,gpass=0.2,gstop=40,ws=3500)
    num, den = signal.cheby1(N=order, Wn=frequenceCritique, fs=48000,rp=0.1)
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence filtre cheb1")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Anglefiltre cheb1 ")
    plt.close()

    order, frequenceCritique = signal.cheb2ord(fs=48000, wp=2500, gpass=0.2, gstop=40, ws=3500)
    num, den = signal.cheby2(N=order, Wn=frequenceCritique, fs=48000,rs=40)
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence filtre cheb2")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Anglefiltre cheb2 ")
    plt.close()

    order, frequenceCritique = signal.ellipord(fs=48000, wp=2500, gpass=0.2, gstop=40, ws=3500)
    num, den = signal.ellip(N=order, Wn=frequenceCritique, fs=48000, rp=0.1,rs=40)
    w, H = signal.freqz(num, den)
    amp = 20 * np.log10(np.abs(H))
    angle = np.angle(H)
    plt.plot(amp)
    plt.savefig("photos\\ReponseFrequence filtre ellip")
    plt.close()
    plt.plot(angle)
    plt.savefig("photos\\Anglefiltre ellip ")
    plt.close()


def p4():
    #etirer une image dun facteur 2 et ecraser de 1/2 sur y
    img = matplotlib.image.imread("goldhill.png")
    newImg = []
    matriceTransformation = [[None]*2048]*512
    print(len(img[0]),len(img))

    for i in range(0,512):
        for j in range(0,2048):
            if j==i*4:
                matriceTransformation[i][j]=1
            else:
                matriceTransformation[i][j]=0

    new = np.matmul(img,matriceTransformation)
    matplotlib.image.imsave("newGoldhill.png",new)

    new=np.zeros((512,2048))
    matriceTransformation = [[1,0],[0,4]]
    for i in range(0,512):

        for j in range(0,512):
            coordonne = np.transpose([i,j])
            x,y =np.matmul(matriceTransformation,coordonne)
            new[round(x)][round(y)] = img[i][j]

    matplotlib.pyplot.gray()
    matplotlib.image.imsave("newGoldhill.png", new)


if __name__ == '__main__':
    #p1()
    #p2()
    #p3()
    p4()

