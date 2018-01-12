#import spinmob as sm
import numpy as np
import Chn
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
import scipy as sp
import math


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
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


########################## Fit energy histogram to extract peak energy of the alpha particle ################################

def guassianFit(x, y):

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction

    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    popt, pcov = curve_fit(gaus, x, y, p0=[100, x0, 5]) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5

    plt.plot(x, y, 'b+:', label='data', linestyle='None')
    plt.plot(x, gaus(x, *popt), 'ro:', label='fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    print('Mean: %f'%popt[1])
    print('Sigma: %f'%popt[2])
    #plt.show()

    return popt[1],popt[2]


def calibrate_linear():
    mean, sigma = [], []

    for n in range(1,8):
        ch = Chn.Chn("Calibration/Pulse_"+str(n)+"V_0111.chn")
        y = ch.spectrum[0:1200]
        x = np.arange(len(y))
        mean_temp, sigma_temp = guassianFit(x, y)
        mean.append(mean_temp)
        sigma.append(sigma_temp)
    x = np.arange(1, 8)
    plt.errorbar(x,mean,yerr=sigma,color='red',markersize='100',linestyle='None')
    axes = plt.gca()
    axes.set_xlim([0,8])

    m,b = np.polyfit(x,mean,1)
    xx = np.linspace(0,7,1000)
    plt.plot(xx, m*xx+b*np.ones(len(xx)))
    print('Intercept: %f'%b)
    print('Slope: %f'%m)
    return b
    #plt.show()
    
def gaus_exp(x, a, b, c, x0, sigma):
    return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b * exp(c * x)

calibrate_linear()
am = Chn.Chn('Calibration/Am_0111_1.Chn')
am_y = am.spectrum
am_x = np.argmax(am_y)
print('Channel of peak for Am: %d'%am_x)


def exGauss(x, l, s, m):
    return l/2*np.exp(1/2*(2*x+l*s*s-2*m))*(1-sp.special.erfc((x+l*s*s-m)/(math.sqrt(2)*s)));    
    
def fitExGauss():
    ch = Chn.Chn("Calibration/Am_0111_1.chn")
    y = ch.spectrum
    x = np.arange(len(y))
    popt, pcov = curve_fit(exGauss, x, y, p0=[1, 1, 1200], maxfev=500000)
    plt.plot(x, y, 'b+:', label='data', linestyle='None')
    plt.plot(x, exGauss(x, *popt), 'ro:', label='fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    print('Mean: %f'%popt[2])
    print('Sigma: %f'%popt[1])
    print('Lambda: %f'%popt[0])
    #plt.show()

    return popt[2]















calibrate()
#fitExGauss()

#def removeAsymmetry(Edata, Ndata):


########################### Determine stopping power as a function of distance ###############################################

