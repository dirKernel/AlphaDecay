import spinmob as sm
import numpy as np
import Chn
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp





ch1 = Chn.Chn("Calibration/Pulse_2V_0111.chn")
y = ch1.spectrum[0:500]
x = np.arange(len(y))[0:500]


################################# Transform from channel number data to energy ###########################################

def convertChannelToEnergy(channelData,m,b):
    energyData = m*channelData + b*np.ones(len(channelData))
    return energyData

########################### Fit bismuth activity data in order to extract lead and bismuth half-lives ##########################

def extractHalfLives(tdata, Adata):
    #assume t is in seconds
    if len(tdata)!= len(Adata):
        print("ERROR: Time and Activity data are not of the same length.")
        return
    f = sm.data.fitter()
    f.set_functions(f='N0*lambda1*lambda2*(np.exp(-lambda1*x)-np.exp(-lambda2*x))/(lambda2-lambda1)+N1*np.exp(-lambda2*t)',
                    p='lambda1=1.81e-5,lambda2=1.9e-4,N0=1.0e5,N1=1.0e2')
    params = f.results[0]

    T1=np.log(2)/params[0]
    T2=np.log(2)/params[1]

    print(f.results[1])

    return np.asarray([T1,T2])



def gaus(x, a, x0, sigma):
    return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))


########################## Fit energy histogram to extract peak energy of the alpha particle ################################

def guassianFit(x, y):
# x variable is the energy bin and the y variable is the number of events in that bin
#     if len(x)!= len(y):
#         print("ERROR: Time and Activity data are not of the same length.")
#         return
#     print(np.pi)
#
#     print(x)
#     print(y)
#     f = sm.data.fitter('a*exp(-(x-x0)**2/(2*sig**2))','sig=2,x0=1000,a=900')
#     f.set_data(x,y)
#     f.fit()
#     #f.ginput()
#     params = f.results[0]
#     return params

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction



    ind = np.argmax(y)
    x0 = x[indß]
    popt, pcov = curve_fit(gaus, x, y, p0=[100, x0, 5])

    # plt.plot(x, y, 'b+:', label='data')
    # plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
    # plt.legend()
    # plt.title('Fig. 3 - Fit for Time Constant')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Voltage (V)')
    #plt.show()
    print(popt[1])
    print(popt[2])
    #plt.show()

    return popt[1],popt[2]




print(guassianFit(x,y))

mean, sigma = [], []

for n in range(1,8):
    ch = Chn.Chn("Calibration/Pulse_"+str(n)+"V_0111.chn")
    y = ch.spectrum
    x = np.arange(len(y))
    mean_temp, sigma_temp = guassianFit(x, y)
    mean.append(mean_temp)
    sigma.append(sigma_temp)
#plt.clear()
plt.errorbar(np.arange(1,8),mean,sigma)
plt.show()


#def removeAsymmetry(Edata, Ndata):


########################### Determine stopping power as a function of distance ###############################################

