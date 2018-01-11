import glob
import Chn
import matplotlib.pyplot as matplot

chnfiles = glob.glob("*.Chn")
chnfiles.sort()

for i in range(len(chnfiles)):
    current = Chn.Chn(chnfiles[i])
    if 0 == i:
        spectrum = current.spectrum
    else:
        spectrum = spectrum + current.spectrum

matplot.plot(spectrum)
matplot.xlabel("Channel")
matplot.ylabel("Count")
matplot.show()