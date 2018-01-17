#import glob
import os
import Chn
import matplotlib.pyplot as matplot

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
    
#    matplot.plot(spectrum)
#    matplot.xlabel("Channel")
#    matplot.ylabel("Count")
#    matplot.show()
    
    return spectrum