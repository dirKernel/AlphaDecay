#import glob
import os
import Chn
import matplotlib.pyplot as plt

def chnsum(folderName):
    chnfiles = os.listdir(folderName)
#    chnfiles = glob.glob("*.Chn")
    chnfiles.sort()
    
    for i in range(len(chnfiles)):
        current = Chn.Chn(folderName+'/'+chnfiles[i])
        if 0 == i:
            spectrum = current.spectrum
        else:
            spectrum = spectrum + current.spectrum
    
    
#    fig1 = plt.figure()
#    plt.plot(spectrum)
#    #fig2 = plt.figure()
#    #plt.plot(spectrum[1000:-1])
#    plt.xlabel("Channel")
#    plt.ylabel("Count")
#    plt.show()
    
    return spectrum