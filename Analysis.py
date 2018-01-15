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

def reducedChiSquare(x,y,fx,yerr,n):
    """
    :param x: x vector
    :param y: y vector
    :param fx: vector of f(x), where f is the fitting function
    :param n: degrees of freedom = number of parameters describing the fit
    :return: Reduced chi^2 of the fit. Ideally should be 1.
    """
    if len(x)!=len(y) or len(y)!=len(fx) or len(fx)!=len(x):
        print("ERROR: x,y and fx should all have the same length.")
        return
    toReturn = 0.0
    for n in range(len(x)):
        toReturn += (y[n]-fx[n])**2/(yerr[n])**2
    return toReturn / n


def linearFit(x, y, yerr):
    """
    To perform linear fit; x: x data; y: y data; yerr: error on y data;
    Output: popt: list of parameter value; perr: list of parameter err
    """
    fit = np.polyfit(x, y, 1, w=yerr, cov=True)
    popt = fit[0]
    pcov = fit[1]
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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

def gaussianFit(x, y, yerr):
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    print(x0)
    yy = y[x0-20:x0+20]
    xx = np.arange(len(yy))
    #plt.plot(xx,yy)
    popt, pcov = curve_fit(gauss, xx, yy, p0=[200, 20, 1], sigma=yerr) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    perr = np.diag(pcov)
    plt.plot(xx+x0-20, yy, 'b+:', label='data', linestyle='None')
    plt.plot(xx+x0-20, gauss(xx, *popt), 'ro:', label='fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    print('Mean: %f'%popt[1])
    print('Sigma: %f'%popt[2])
    print('Error on the Mean: %f'%perr[1])
    RealMean = popt[1]+x0-20
    plt.show()

    return RealMean,perr[1]

def fitAlphaPeak(filepath, p0, left=100, right=100, res_tick=[-3,0,3]):
    """
    This is the function to fit Alpha Peak
    filepath: full path to the file; p0: list of initial guess; left: how much away
    to the left from peak channel; right: how much away to the right from peak channel
    """
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


#def gaussianFit(x, y, left=50, right=50):
#    ind = np.argmax(y) #to get the peak value x-coord
#    x0 = x[ind] #x0 is the peak value x-coord (channel number)
#    y = y[x0-left:x0+right]
#    popt, pcov = curve_fit(gauss, x, y, p0=[50, x0, 5]) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
#    perr = np.diag(pcov)
#    plt.plot(x, y, 'b+:', label='data', linestyle='None')
#    plt.plot(x, gauss(x, *popt), 'ro:', label='fit')
#    plt.legend()
#    plt.xlabel('Channels')
#    plt.ylabel('Counts')
#    print('Mean: %f'%popt[1])
#    print('Mean Error: %f'%perr[1])
#    print('Sigma: %f'%popt[2])
#    plt.show()
#    RealMean = popt[1]+x0-left
#
#    return RealMean, popt[2] 
#

def calibratePulses():
    mean, sigma= [], []
    i = np.zeros(15)
    for n in range(1,16):
        i[n-1] = 1000+500*(n-1)
        print(str(int(i[n-1])))
        ch = Chn.Chn("Calibration_4/"+str(int(i[n-1]))+"mV_0115.chn")
        y = ch.spectrum
        x = np.arange(len(y))
        yerr = np.sqrt(y)
        print('!!!!')
        print(yerr)
        mean_temp, sigma_temp = gaussianFit(x, y, yerr)
        mean.append(mean_temp)
        sigma.append(sigma_temp)
    x = i/1000
    plt.errorbar(x,mean,yerr=sigma,color='red',markersize='100',linestyle='None')
    axes = plt.gca()
    axes.set_xlim([0,8])

    res = np.polyfit(x,mean,1, w=sigma,cov=True)[0]
    m = res[0]
    b = res[1]
    cov = np.polyfit(x,mean,1,cov=True)[1]
    merr = np.sqrt(cov[0][0])
    berr = np.sqrt(cov[1][1])
    xx = np.linspace(0,7,1000)
    print(berr)
    print(merr)
    plt.plot()
    plt.plot(xx, m*xx+b*np.ones(len(xx)))
    print('Intercept: %f'%b)
    print('Slope: %f'%m)
    plt.show()
    
    return m, b, merr, berr

#def calibrateLinearFit():
#    mean, sigma = [], []
#    for n in range(1,8):
#        ch = Chn.Chn("Calibration_3/"+str(n)+"V_0112_2.chn")
#        y = ch.spectrum
#        x = np.arange(len(y))
#        mean_temp, sigma_temp = guassianFit(x, y)
#        mean.append(mean_temp)
#        sigma.append(sigma_temp)
#    x = np.arange(1,8)
#    plt.errorbar(x,mean,yerr=sigma,color='red',markersize='100',linestyle='None')
#    axes = plt.gca()
#    axes.set_xlim([0,8])
#
#    print(np.polyfit(x,mean,1,cov=True))
#    print('Intercept: %f'%b)
#    print('Intercept Error: %f'%errm)
#    print('Slope: %f'%m)
#    print('Slope Error: %f'%errb)
#    plt.show()





    


# For the pressure part
def pressureData(folderName):
    """
    This function reads all pressure varied data from the directory and fit linearly;
    folderName: string of the folder name under root directory
    """
    peak_means, peak_means_e, fitfunc = [], [], []
    data = os.listdir(folderName)
    pressure = []
    
    for d in data:
        p = d.split('m')[0]
        pressure.append(int(p))
        
    for file in data:
        m, m_err, x0, func = fitAlphaPeak('pressure/'+file, [250, 0.05, 5, 100])
        peak_means.append(m+x0-100)
        peak_means_e.append(m_err)
        fitfunc.append(func)
        
    plt.figure()
    x = pressure
    y = peak_means
    yerr = peak_means_e
    popt, perr = linearFit(x, y, yerr)
    m = popt[0]
    b = popt[1]
    m_e = perr[0]
    b_e = perr[1]
    xx = np.linspace(min(x), max(x))
    plt.plot(xx, m*xx+b*np.ones(len(xx)),label='Linear Fit')
    print('Intercept: %f $\pm$ %f'%(b,b_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    plt.plot(x, y, 'kx', label='Data')
    plt.xlabel('Pressure (mBar)')
    plt.ylabel('Mean Channel')
    plt.legend()
    plt.show()
    
    return popt, perr 
        
calibratePulses()
pressureData('pressure')
AmChannel = fitAlphaPeak("Calibration/Am_0111_1.chn", \
                         [150, 0.1, 0.1, 250], left=100, right=40, res_tick=[-10,0,10])



























##### Backup ######3

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

########################## Fit energy histogram to extract peak energy of the alpha particle ################################
########################### Determine stopping power as a function of distance ###############################################

def locallyDifferentiate(x,y,xerr,yerr):
    """
    :param x:
    :param y:
    :param xerr:
    :param yerr:
    :return:
    """
    X=[]
    Xerr=[]
    for n in len(x):
        X.append((x[n+1]+x[n])/2)
        Xerr.append((xerr[n+1]+xerr[n])/2)
    print(len(X))
    Y=[]
    for n in len(x):
        Y.append((y[n+1]-y[n])/(x[n+1]-x[n]))
    print(len(Y))
    return X,Y,
