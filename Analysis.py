import numpy as np
import Chn
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sp
import math
import spinmob as s
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

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

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def guassianFit(x, y):
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    popt, pcov = curve_fit(gauss, x, y, p0=[100, x0, 5]) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    plt.plot(x, y, 'b+:', label='data', linestyle='None')
    plt.plot(x, gauss(x, *popt), 'ro:', label='fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    print('Mean: %f'%popt[1])
    print('Sigma: %f'%popt[2])
    plt.show()

    return popt[1],popt[2]

def calibrateLinearFit():
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
    plt.show()

def expGauss(x, A, l, s, m):
    return A*l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))    
    
def expGaussFit_scipy(x, y, yerr, p0, res_tick=[-3,0,3]):
    plt.figure()
    popt, pcov = curve_fit(expGauss, x, y,p0=p0, maxfev=50000)
    plt.errorbar(x, y, yerr=yerr,fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3,color='k')
    plt.plot(x, expGauss(x, *popt), '-r', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    perr = np.sqrt(np.diag(pcov))
    print('\nMean: %f with error %f'%(popt[3], perr[3]))
    print('Sigma: %f with error %f'%(popt[2], perr[2]))
    print('Lambda: %f with error %f'%(popt[1], perr[1]))
    print('A: %f with error %f'%(popt[0], perr[0]))
    
    # Plot residuals
    difference = expGauss(x,*popt)-y
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0)
    axes.figure.add_axes(axes2)
    axes2.set_xticks([])
    axes2.set_yticks(res_tick)
    axes2.axhline(y=0, color='r', linestyle='-')
    axes2.plot(x,difference,'k+',markersize=3)

    plt.show()
    
    func = 'A*l/2*exp(l/2*(2*x-2*m+l*s^2))*(1-erf((x+l*s^2-m)/(s*sqrt(2))))'
    func = func.replace('A',str(round(popt[0],1)))
    func = func.replace('l',str(round(popt[1],2)))
    func = func.replace('s',str(round(popt[2],1)),3)
    func = func.replace('m',str(round(popt[3],1)))
    print('Fit='+func)
    
    return popt[3], perr[3], func # return the mean channel values

def emg(x,m,s,l):
    return l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(np.sqrt(2)*s)))

def simple(x,A,m,s,l):
    return A*emg(x,m,s,l)

def expGaussFit_spinmob():
    # not used
    my_fitter = s.data.fitter()
    my_fitter.set_functions(f=simple, p='A=200,m=250,s=0.1,l=0.1')
    ch = Chn.Chn("Calibration/Am_0111_1.chn")
    y = ch.spectrum[1000:1300]
    x = np.arange(len(y))
    my_fitter.set_data(x,y)
    my_fitter.fit()
    
def fitAlphaPeak(filepath, p0, left=100, right=100, res_tick=[-3,0,3]):
    '''
    This is the function to fit Alpha Peak
    filepath: full path to the file; p0: list of initial guess; left: how much away
    to the left from peak channel; right: how much away to the right from peak channel
    '''
    ch = Chn.Chn(filepath)
    y = ch.spectrum
    x = np.arange(len(y))
    ind = np.argmax(y)
    x0 = x[ind]
    y = y[x0-left:x0+right]
    x = np.arange(len(y))
    yerr = []
    for ele in y:
        yerr.append(math.sqrt(ele))
    mean, mean_err, func = expGaussFit_scipy(x, y, yerr, p0, res_tick)
    print('On Plot Mean: %f with error %f'%(mean, mean_err)+'\n')
    print('None Scaled Mean: %f with error %f'%(mean+x0-left, mean_err)+'\n')
    
    return mean, mean_err, x0, func

# For the pressure part
def pressureData():
    peak_means, peak_means_e, fitfunc = [], [], []
    data = ['0mBar_0112.Chn',
            '50mBar_0112.Chn',
            '100mBar_0112.Chn',
            '150mBar_0112.Chn',
            '200mBar_0112.Chn',
            '250mBar_0112.Chn',
            '300mBar_0112.Chn',
            '350mBar_0112.Chn',
            '400mBar_0112.Chn',
            '450mBar_0112.Chn',
            '470mBar_0112.Chn']
    for file in data:
        m, m_err, x0, func = fitAlphaPeak('pressure/'+file, [250, 0.05, 5, 100])
        peak_means.append(m+x0-100)
        peak_means_e.append(m_err)
        fitfunc.append(func)
    plt.figure()
    x = [0,50,100,150,200,250,300,350,400,450,470]
    m,b = np.polyfit(x,peak_means,1)
    xx = np.linspace(x[0],x[-1])
    plt.plot(xx, m*xx+b*np.ones(len(xx)),label='Linear Fit')
    print('Intercept: %f'%b)
    print('Slope: %f'%m)
    plt.plot(x, peak_means, 'kx',label='Data')
    plt.xlabel('Pressure (mBar)')
    plt.ylabel('Mean Channel')
    plt.legend()
    plt.show()
        
#calibrateLinearFit()
print(os.listdir('pressure'))
pressureData()
AmChannel = fitAlphaPeak("Calibration/Am_0111_1.chn", \
                         [150, 0.1, 0.1, 250], left=100, right=40, res_tick=[-10,0,10])



########################## Fit energy histogram to extract peak energy of the alpha particle ################################
########################### Determine stopping power as a function of distance ###############################################