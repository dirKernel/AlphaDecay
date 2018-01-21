import numpy as np
import Chn
import pylab as plb
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.3 #set the value globally
from scipy.optimize import curve_fit
import scipy as sp
from scipy.stats import chisquare
import math
import spinmob as s
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from chnsum import chnsum
import uncertainties as unc  
import uncertainties.unumpy as unp  

global E0 # americium energy needed for calibration
global E0Err
global calibIntercept
global calibInterceptErr
global N0
global N0Err
global slope
global slopeErr
global intercept
global interceptErr

E0 = 5.485
E0Err = 0.002

def reducedChiSquare(y,fx,yerr,m):
#    """
#    :param y: y vector
#    :param fx: vector of f(x), where f is the fitting function
#    :param m: the number of fitted parameters
#    :return: Reduced chi^2 of the fit. Ideally should be 1.
#    """
#
#    toReturn, count = 0.0, 0
#    print(y)
#    print(fx)
#    print(yerr)
#    for i in range(len(y)):
#        if yerr[i]<=-100000:
#            continue
#        else:
#            toReturn += (y[i]-fx[i])**2/fx[i]
#            count += 1
#    dof = count-m
    
#    return toReturn/dof, dof

    chisq, p = chisquare(y, f_exp=fx, ddof=m-1, axis=0)
    
    return chisq/(len(y)-m), len(y)-m


def linearFit(x, y, yerr):
    """
    To perform linear fit; x: x data; y: y data; yerr: error on y data;
    Output: popt: list of parameter value; perr: list of parameter err
    """
#    x = np.asarray(x)
    fit = np.polyfit(x, y, 1, w=yerr, cov=True)
    popt = fit[0]
    pcov = fit[1]
    perr = np.sqrt(np.diag(pcov))
    d = []
    for i in range(len(y)):
        d.append(y[i]-popt[0]*x[i]-popt[1])
#    print(d)
#    print(len(d))
#    print(y)
#    print(popt[0])
#    print(x)
#    print(popt[1])
#    print(type(x[1]))
#    print(type(popt[0]))
#    popt[0] = np.float64(popt[0])
    #print(popt[0]*x)
#    d = y-(x*popt[0]+popt[1]*np.ones(15))
#    np.asarray([2500.0, 5500.0, 4500.0, 3500.0, 1500.0, \
#         6500.0, 7500.0, 7000.0, 8000.0, 1000.0, \
#         6000.0, 4000.0, 3000.0, 2000.0, 5000.0])
#    print(d)
    
    return popt, perr

#def linear(x, m, b):
#    return m*x+b
#
#def linearFit(x, y, yerr):
#    popt, pcov = curve_fit(linear, x, y, p0=[0.1, -10], sigma=yerr)
#    perr = np.sqrt(np.diag(pcov))
##    print(type(x[1]))
##    print(type(y[1]))
#    #d = y-linear(x, *popt)
#    d = []
#    for i in range(len(y)):
#        d.append(y[i]-popt[0]*x[i]+popt[1])
#    
#    return popt, perr, d
    

def gauss(x, a, mean, sigma):
    return a*np.exp(-(x-mean)**2/(2*sigma**2))

def expGauss(x, A, l, s, m):
    return A*l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))    

def expGaussFit(filePathtobeSaved, x, y, yerr, p0, x0, left, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=(8, 6))
    popt, pcov = curve_fit(expGauss, x, y, p0=p0, maxfev=50000)
    npara = 4
    rchi, dof = reducedChiSquare(y, expGauss(x, *popt), yerr, npara)
    plt.errorbar(x+x0-left, y, yerr=yerr,fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3,color='k')
    plt.plot(x+x0-left, expGauss(x, *popt), '-r', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    perr = np.sqrt(np.diag(pcov))
    print('\nMean (Scaled): %f $\pm$ %f'%(popt[3], perr[3]))
    print('Sigma: %f $\pm$ %f'%(popt[2], perr[2]))
    print('Lambda: %f $\pm$ %f'%(popt[1], perr[1]))
    print('A: %f $\pm$ %f'%(popt[0], perr[0]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    
    # Plot residuals
    d = y-expGauss(x,*popt)
    stu_d = d/np.std(d, ddof=npara-1) # studentized residual
    stu_d_err = yerr/np.std(d, ddof=npara-1)
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.05)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks(res_tick)
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(x+x0-left, stu_d, yerr=stu_d_err, fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3,color='k')
    
    func = 'A*l/2*exp(l/2*(2*x-2*mu+l*s^2))*\n(1-erf((x+l*s^2-mu)/(s*sqrt(2))))'
    func = func.replace('A','('+str(int(popt[0]))+'$\pm$'+str(int(perr[0]))+')')
    func = func.replace('l','('+str(round(popt[1],3))+'$\pm$'+str(round(perr[1],3))+')')
    func = func.replace('s','('+str(round(popt[2],1))+'$\pm$'+str(round(perr[2],1))+')',3)
    func = func.replace('mu','('+str(round(popt[3],1))+'$\pm$'+str(round(perr[3],1))+')')
    print('Fit='+func)
    
    textstr = '$\chi^2$=%.2f\tDOF=%d\nFit=%s'%(rchi, dof, func)
    plt.text(0.1, 0.9, textstr, fontsize=8, transform=plt.gcf().transFigure)

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000)
    
    return popt, perr, rchi, dof, func # return the mean channel values  

def gaussianFit(filePathtobeSaved, x, y, yerr, p0=[300, 20, 2.5], left=20, right=20, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=[5.5,3.5])
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = yerr[x0-left:x0+right]
    popt, pcov = curve_fit(gauss, xx, yy, p0=p0, maxfev=50000) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    npara = 3
    rchi, dof = reducedChiSquare(yy, gauss(xx, *popt), yerr, npara)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(xx+x0-left, yy, yerr=yerr,fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3, color='k')
    xxx = np.linspace(min(xx),max(xx),1000)
    plt.plot(xxx+x0-left, gauss(xxx, *popt), 'r-', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    
    # Plot residuals
    d = yy-gauss(xx,*popt)
    stu_d = d/np.std(d, ddof=npara-1)
    stu_d_err = yerr/np.std(d, ddof=npara-1)
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.05)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks(res_tick)
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(xx+x0-left, stu_d, yerr=stu_d_err, fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3,color='k')
    
    func = 'A*exp(-(x-mean)^2/(2*sigma^2))'
    func = func.replace('A','('+str(int(popt[0]))+'$\pm$'+str(int(perr[0]))+')')
    func = func.replace('mean','('+str(round(popt[1],3))+'$\pm$'+str(round(perr[1],3))+')')
    func = func.replace('sigma','('+str(round(popt[2],1))+'$\pm$'+str(round(perr[2],1))+')',3)
    
    textstr = '$\chi^2$=%.2f\tDOF=%d\nFit=%s'%(rchi, dof, func)
    plt.text(0.1, 0.9, textstr, fontsize=8, transform=plt.gcf().transFigure)
    
    popt[1] = popt[1]+x0-left
    print('A: %f $\pm$ %f'%(popt[0], perr[0]))
    print('Mean: %f $\pm$ %f'%(popt[1], perr[1]))
    print('Sigma: %f $\pm$ %f'%(popt[2], perr[2]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    #plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000)

    return popt, perr

def fitAlphaPeak(filePathtobeSaved, filePath, p0, left=100, right=100, res_tick=[-3,0,3]):
    """
    This is the function to fit Alpha Peak
    filepath: full path to the file; p0: list of initial guess; left: how much away
    to the left from peak channel; right: how much away to the right from peak channel
    res_tick: the residual plot y axis ticks
    """
    ch = Chn.Chn(filePath)
    y = ch.spectrum
    #print('Real time: %d'%ch.real_time)
    x = np.arange(len(y))
    ind = np.argmax(y)
    x0 = x[ind]
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = np.sqrt(yy)
    popt, perr, rchi, dof, func = expGaussFit(filePathtobeSaved, xx, yy, yerr, p0, x0, left, res_tick)
    popt[3] += x0-left
    #print('On Plot Mean: %f with error %f'%(mean, mean_e)+'\n')
    print('Mean (Not Scaled): %f $\pm$ %f'%(popt[3], perr[3])+'\n')
    
    return popt, perr, rchi, dof, func

################################# Transform from channel number data to energy ###########################################




def convertChannelToEnergy(channelData):


    E0 = 5.485
    
#    popt_am, perr_am, func_am = fitAlphaPeak("Americium/Am_0111_1.chn", \
#                             [200, 1, 1, 100], left=100, right=50, res_tick=[-10,0,10])
#    m_am, m_am_e = popt_am[3], perr_am[3]
#    print('Amerisium Calibration: Mean channel = %f $\pm$ %f\nFit function = %s'%\
#          (m_am, m_am_e, func_am))
#
#    N0 = fitAlphaPeak("Calibration/Am_0111_1.chn",[500, 0.1, 0.1, 250])[0]

    # ^ Americium reference energy and recorded channel number

    #a = calibratePulses()[0]

    #c = calibratePulses()[1]


    energyData = m*channelData + b*np.ones(len(channelData))

    return energyData

########################### Fit bismuth activity data in order to extract lead and bismuth half-lives ##########################

def activityFitFunc(x, lambda1, lambda2, N0, N1):
    return 'N0*lambda1*lambda2*(np.exp(-lambda1*x)-np.exp(-lambda2*x))/(lambda2-lambda1)+N1*np.exp(-lambda2*t)'

def activityFit(x, y, yerr):
    popt, pcov = curve_fit(activityFitFunc, x, y, p0=[1.81e-5,1.9e-4,1.0e5,1.0e2], maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(x, y, yerr=yerr,fmt='x', elinewidth=0.5 ,capsize=1, ecolor='k', \
                 label='Data', linestyle='None', markersize=3,color='k')
    xx = np.linspace(min(x), max(x))
    plt.plot(xx, activityFitFunc(xx, *popt), '-r', label='Fit')
    plt.show()

    return popt, perr

def calibratePulses(folderName):
    mean, sigma, mean_e, vol = [], [], [], []
    data = os.listdir(folderName)
    for d in data:
        filePathtobeSaved = 'Figures/Calibration/'+d.split('.')[0]
        ch = Chn.Chn(folderName+'/'+d)
        y = ch.spectrum
        x = np.arange(len(y))
        yerr = np.sqrt(y)
        popt, perr = gaussianFit(filePathtobeSaved, x, y, yerr, p0=[300, 20, 2.5], res_tick=[-3,0,3])
        mean.append(popt[1])
        print(type(popt[1]))
        sigma.append(popt[2])
        mean_e.append(perr[1])
        vol.append(np.asarray(int(d.split('_')[0])))
    x = vol 
    y = mean
    yerr = mean_e
    
    filePathtobeSaved = 'Figures/Calibration/pulseLinear'
    fig = plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='x', elinewidth=0.5 ,capsize=3, ecolor='k', \
                 label='Data', linestyle='None', markersize=3, color='k')
    popt, perr = linearFit(x, y, yerr)
    print(d)
    m = popt[0]
    b = popt[1]
    m_e = perr[0]
    b_e = perr[1]
    npara = 2
    x = np.asarray(x) 
    y = np.asarray(y)
    yerr = np.asarray(yerr) 
    print(yerr)
    rchi, dof = reducedChiSquare(y, m*x+b*np.ones(len(x)), yerr, npara)
    xx = np.linspace(0, max(x))
    plt.plot(xx, m*xx+b*np.ones(len(xx)), color='r', label='Fit')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Mean Channel')
    print('Intercept: %f $\pm$ %f'%(b,b_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    plt.legend()
    
    x = np.asarray(x) 
    # this line of code saved my life, and it may save your life with this error
    # 'numpy.float64' object cannot be interpreted as an integer - Alvin
    d = y-(x*popt[0]+popt[1]*np.ones(len(x)))
    stu_d = d/np.std(d, ddof=npara-1)
    stu_d_err = yerr/np.std(d, ddof=npara-1)
    print(stu_d_err)
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.05)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks([-2,0,2])
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
#    labels = [item.get_text() for item in axes2.get_xticklabels()]
#    empty_string_labels = ['']*len(labels)
#    axes2.set_xticklabels(empty_string_labels)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes2.errorbar(x, stu_d, yerr=stu_d_err, fmt='x', elinewidth=0.5 ,capsize=3, ecolor='k', \
                 label='Stud', linestyle='None', markersize=3,color='k')
    
    func = 'slope*x+b'
    func = func.replace('b','('+str(int(round(popt[1],0)))+'$\pm$'+str(int(round(perr[1],0)))+')')
    func = func.replace('slope','('+str(round(popt[0],4))+'$\pm$'+str(round(perr[0],4))+')')
    print('Fit='+func)
    
    textstr = '$\chi^2$=%.2f\tDOF=%d\nFit=%s'%(rchi, dof, func)
    plt.text(0.1, 0.9, textstr, fontsize=10, transform=plt.gcf().transFigure)
    
    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000)


    ########## Define global variables which parameterize the conversion between channel number and energy ##################
    calibIntercept = b
    calibInterceptErr = b_e
    
    popt_am, perr_am, rchi_am, dof_am, func_am = fitAlphaPeak("Figures/Calibration/Am_0111_1", "Americium/Am_0111_1.chn", \
                         [200, 1, 1, 100], left=100, right=50, res_tick=[-2,0,2])
    N0, N0Err = popt_am[3], perr_am[3]
    print('Amerisium Calibration: Mean channel = %f $\pm$ %f\nFit function = %s'%\
      (N0, N0Err, func_am))

    slope = E0/(N0-calibIntercept)
    slopeErr = slope*np.sqrt((E0Err/E0)**2+(1/(N0-calibIntercept))**2*(calibInterceptErr**2+N0Err**2))

    intercept = E0*calibIntercept/(calibIntercept-N0)
    interceptErr = intercept*np.sqrt((E0Err/E0)**2+(N0*calibInterceptErr/(calibIntercept*(calibIntercept-N0)))**2+(N0Err/(calibIntercept-N0))**2)
    
    return m, b, m_e, b_e

def pressureData(folderName):
    """
    This function reads all pressure varied data from the directory and fit linearly;
    folderName: string of the folder name under root directory
    """
    peak_means, peak_means_e, fitfunc = [], [], []
    data = os.listdir(folderName)
    pressure = []
    
    for d in data:
        p = d.split('_')[0]
        pressure.append(np.asarray(int(p)))
        
    for file in data:
        popt, perr, func = fitAlphaPeak(folderName+'/'+file, p0=[600, 0.02, 7, 100],\
                                          right=80)
        peak_means.append(popt[3])
        peak_means_e.append(perr[3])
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
    plt.plot(xx, m*xx+b*np.ones(len(xx)),label='Fit')
    print('Intercept: %f $\pm$ %f'%(b,b_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    plt.plot(x, y, 'kx', label='Data')
    plt.xlabel('Pressure (mBar)')
    plt.ylabel('Mean Channel')
    plt.legend()
    plt.show()
    
    return m, m_e, b, b_e  
    

def halflifeMeasurement(folderName):
    files = os.listdir(folderName)
    files.sort()
    activity, elapse = [], []
    for f in files:
        ch = Chn.Chn(folderName+'/'+f)
        spectrum = ch.spectrum
        activity.append(sum(spectrum))
        fileInd = int(f.split('_')[0])
        elapse.append(fileInd*10*60) #10 minutes, starts at 0
    
    x = elapse
    y = activity
    yerr = np.sqrt(activity)
    
    popt, perr = activityFit(x, y, yerr)
    
    l1, l2 = popt[0], popt[1]
    l1_e, l2_e = perr[0], perr[1]
    
    T1=np.log(2)/l1
    T2=np.log(2)/l2
    # We need to do a propagation of error to T1 and T2
    
    return T1, T2

    

######################## Function Calling Area ##################################
    
m_calib, m_calib_e, b_calib, b_calib_e = calibratePulses('Calibration_4')
#m_press, m_press_e, b_press, b_press_e = pressureData('Pressure_2')

#popt_am, perr_am, func_am = fitAlphaPeak("Americium/Am_0111_1.chn", \
#                         [200, 1, 1, 100], left=100, right=50, res_tick=[-10,0,10])
#m_am, m_am_e = popt_am[3], perr_am[3]
#print('Amerisium Calibration: Mean channel = %f $\pm$ %f\nFit function = %s'%\
#      (m_am, m_am_e, func_am))

#halflifeMeasurement('Decay_1')



























########################## Fit energy histogram to extract peak energy of the alpha particle ################################
########################### Determine stopping power as a function of distance ###############################################

def locallyDifferentiate(x,y,xerr,yerr):
    """
    :param x: x data vector
    :param y: y data vector
    :param xerr: vector of error on x data
    :param yerr: vector of error on y data
    :return: (X,Y) are the local derivative of the (x,y) data set. Ie for each adjacent data points in the input data set, a straight line is drawn between them
    and the slope of the line is added to the vector Y. The vector X is compromised of the midpoints between the adjacent x's. The errors on X,Y are computed in terms
     of xerr and yerr
    """
    X=[]
    Xerr=[]
    for n in len(x):
        X.append((x[n+1]+x[n])/2)
        Xerr.append((xerr[n+1]+xerr[n])/2)
    print(len(X))
    Y=[]
    Yerr=[]
    for n in len(x):
        Y.append((y[n+1]-y[n])/(x[n+1]-x[n]))
        Yerr.append(Y[n]*np.sqrt(((xerr[n+1]**2+xerr[n]**2)/(x[n+1]-x[n])**2)+((yerr[n+1]**2+yerr[n]**2)/(y[n+1]-y[n])**2)))
    print(len(Y))

    return X,Y,Xerr,Yerr



