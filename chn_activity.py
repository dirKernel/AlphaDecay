# Appears to work in both Python 2.7 and 3.5

import glob
import Chn
import numpy
import matplotlib.pyplot as matplot

chnfiles = glob.glob("*.Chn")
chnfiles.sort()

time = []
activity = []
uncertainty = []

for i in range(len(chnfiles)):
    current = Chn.Chn(chnfiles[i])
    spectra = current.spectrum
    print(spectra.shape)
    time.append(current.mid_time())
    total = spectra[500:1800].sum() # Adjust as necessary
    activity.append(total/current.live_time)
    uncertainty.append(numpy.sqrt(total)/current.live_time)

# convert the lists to NumPy arrays
time = numpy.array(time)
activity = numpy.array(activity)
uncertainty = numpy.array(uncertainty)

#matplot.plot(time-time.min(),activity)
matplot.errorbar(time-time.min(),activity,uncertainty,fmt="+")
matplot.xlabel("Time (s)")
matplot.ylabel("Activity (counts/s)")
matplot.show()