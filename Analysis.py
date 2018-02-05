import numpy as np
import Chn
#import pylab as plb
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
mpl.rcParams.update({'font.size': 15})
mpl.rcParams['axes.labelsize'] = 18
from scipy.optimize import curve_fit
from scipy.integrate import quad
import scipy as sp
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from chnsum import chnsum
#import uncertainties as unc  
#import uncertainties.unumpy as unp  

global E0 # americium energy needed for calibration
global E0Err
global calibIntercept #intercept on channel number versus voltage. beta in our notes.
global calibInterceptErr
global N0
global N0Err
global slope #slope on energy versus channel number
global slopeErr
global intercept #intercept on energy versus channel number
global interceptErr
global TOTAL
global TOTALErr

slope = 0.00457880799792
slopeErr = 3.35945387177e-06
intercept = 0.0661289090633
interceptErr = 0.00397196078318

#E0 = 5.48556
#E0Err = 0.00012

####################################################################################################################
################################################### Fitters ########################################################
####################################################################################################################

def plotStudRes(ax, d, xx , yerr, res_tick, x0=0, left=0):
    stu_d = d/yerr
    stu_d_err = np.ones(len(d))
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("top", size="20%", pad=0.1)
    ax.figure.add_axes(ax2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_yticks(res_tick)
    ax.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    ax2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    ax2.set_ylabel('Studentized\nResiduals', color='k', fontsize=12)
    from matplotlib.ticker import AutoLocator, AutoMinorLocator
    ax2.get_yaxis().set_major_locator(AutoLocator())
#    ax2.get_yaxis().set_minor_locator(AutoMinorLocator())
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=2)
    ax.tick_params(axis='both', direction='in')
    ax2.tick_params(axis='both', direction='in')
    ax2.errorbar(xx+x0-left, stu_d, yerr=stu_d_err, fmt='+', elinewidth=1.5 ,capsize=3, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k')

def reducedChiSquare(y,fx,yerr, npara):
    """
    :param y: y vector
    :param fx: vector of f(x), where f is the fitting function
    :param m: the number of fitted parameters
    :return: Reduced chi^2 of the fit. Ideally should be 1, dof the degree of freedom.
    """
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
    
    toReturn, count = 0.0, 0
    for i in range(len(y)):
        if yerr[i]==0:
            continue
        else:
            toReturn += (y[i]-fx[i])**2/yerr[i]**2
            count += 1
    dof = count-npara
    
    return toReturn/dof, dof

def linFitXIntercept(x, m, h):
    return m*x-m*h
    
def LinearFit_xIntercept(x, y, yerr):
    yerr = np.asarray(yerr)
    x = np.asarray(x)
    y = np.asarray(y)
    popt, pcov = curve_fit(linFitXIntercept, x, y, p0=[0.01, -20], maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr
    
def linearFit(x, y, yerr):
    """
    To perform linear fit; x: x data; y: y data; yerr: error on y data;
    Output: popt: list of parameter value; perr: list of parameter err
    """
    yerr = np.asarray(yerr)
    x = np.asarray(x)
    y = np.asarray(y)
    fit = np.polyfit(x, y, 1, w=1/yerr, cov=True)
    popt = fit[0]
    pcov = fit[1]
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

def gauss(x, a, mean, sigma):
    return a*np.exp(-(x-mean)**2/(2*sigma**2))

def gaussMul(x, *params, sigmaFixed=True):
    if not sigmaFixed:
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            a = params[i]
            mean = params[i+1]
            sigma = params[i+2]
            y = y + a*np.exp(-(x-mean)**2/(2*sigma**2))
        return y
    else:
        y = np.zeros_like(x)
        for i in range(0, len(params)-1, 2):
            a = params[i]
            mean = params[i+1]
            sigma = params[-1]
            y = y + a*np.exp(-(x-mean)**2/(2*sigma**2))
            
    return y

def expGauss(x, A, l, s, m):
    return A*l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))    

def expGaussMul(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        A = params[i]
        l = params[i+1]
        s = params[i+2]
        m = params[i+3]
        y = y + A*l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))
    return y

def expGaussFitMul(filePathtobeSaved, x, y, yerr, p0, x0, left, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=(8, 6))
    popt, pcov = curve_fit(expGaussMul, x, y, p0=p0, maxfev=5000000)
    npara = len(p0)
    rchi, dof = reducedChiSquare(y, expGaussMul(x, *popt), yerr, npara)
    xx, xx_e = convertChannelToEnergy(x)
    x0, x0_e = convertChannelToEnergy(x0)
    left, l_e = convertChannelToEnergy(left)
    plt.errorbar(xx+x0-left, y, yerr=yerr, xerr=xx_e+x0_e-l_e, fmt='o', elinewidth=1.5 ,capsize=3, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k')
    plt.plot(xx+x0-left, expGaussMul(x, *popt), '-r', label='Fit', linewidth=2)
#    plt.legend(loc=2)
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'$^{212}$Bi Decay to $^{208}$Tl'+'\nChannel Counts')
#    plt.ylim((-100,3400))
    perr = np.sqrt(np.diag(pcov))
    
    for i in range(0,len(popt),4):
        print('A %d: %f $\pm$ %f'%(i/4+1, popt[i], perr[i]))
        print('Lambda %d: %f $\pm$ %f'%(i/4+1, popt[i+1], perr[i+1]))
        print('Sigma %d: %f $\pm$ %f'%(i/4+1, popt[i+2], perr[i+2]))
        print('Mean %d (Scaled): %f $\pm$ %f\n'%(i/4+1, popt[i+3], perr[i+3]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
            
    # Plot residuals
    d = y-expGaussMul(x,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, xx, yerr, res_tick=res_tick, x0=x0, left=left)  

#    ax = fig.add_subplot(111)
#    ax.annotate('#1', xy=(convertChannelToEnergy(popt[3]-2, noErr=True), 200), xytext=(convertChannelToEnergy(popt[3]-2, noErr=True), 200+400),\
#            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
#    ax.annotate('#2', xy=(convertChannelToEnergy(popt[7]-2, noErr=True), 200), xytext=(convertChannelToEnergy(popt[7]-2, noErr=True), 200+400),\
#            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
#    ax.annotate('#3', xy=(convertChannelToEnergy(popt[11]-2, noErr=True), 2700), xytext=(convertChannelToEnergy(popt[11]-2, noErr=True), 2700+400),\
#            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
#    ax.annotate('#4', xy=(convertChannelToEnergy(popt[15]-1, noErr=True), 1200), xytext=(convertChannelToEnergy(popt[15]-1, noErr=True), 1200+400),\
#            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    return popt, perr, rchi, dof # return the mean channel values  

def expGaussFit(filePathtobeSaved, x, y, yerr, p0, x0, left, res_tick=[-3,0,3], xerr=0):
    fig = plt.figure(figsize=(8, 6))
    popt, pcov = curve_fit(expGauss, x, y, p0=p0, maxfev=50000)
    npara = len(p0)
    rchi, dof = reducedChiSquare(y, expGauss(x, *popt), yerr, npara)
    plt.errorbar(x+x0-left, y, yerr=yerr, xerr=xerr,fmt='o', elinewidth=1.5 ,capsize=3, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k')
    plt.plot(x+x0-left, expGauss(x, *popt), '-r', label='Fit')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Counts')
    perr = np.sqrt(np.diag(pcov))
    print('\nMean 1 (Scaled): %f $\pm$ %f'%(popt[3], perr[3]))
    print('Sigma 1: %f $\pm$ %f'%(popt[2], perr[2]))
    print('Lambda 1: %f $\pm$ %f'%(popt[1], perr[1]))
    print('A 1: %f $\pm$ %f'%(popt[0], perr[0]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
            
    # Plot residuals
    d = y-expGauss(x,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, x, yerr, res_tick=res_tick, x0=x0, left=left)

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    return popt, perr, rchi, dof # return the mean channel values  

def gaussianFit(filePathtobeSaved, x, y, yerr, p0=[300, 20, 2.5], left=15, right=15, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=(8,6))
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = yerr[x0-left:x0+right]
    popt, pcov = curve_fit(gauss, xx, yy, p0=p0, maxfev=50000) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    npara = len(p0)
    rchi, dof = reducedChiSquare(yy, gauss(xx, *popt), yerr, npara)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(xx+x0-left, yy, yerr=yerr, fmt='o', elinewidth=1.5 ,capsize=3, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k') 
#    fmt='+', elinewidth=1 ,capsize=2, ecolor='b', \
#                 label='Data', linestyle='None', markersize=4,color='b'
                 
    xxx = np.linspace(min(xx),max(xx),1000)
    plt.plot(xxx+x0-left, gauss(xxx, *popt), 'r-', label='Fit', linewidth=2)
    plt.xlabel('MCA Channel Number')
    plt.ylabel('Pulse Channel Counts')
            
    # Plot residuals
    d = yy-gauss(xx,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, xx, yerr, res_tick=res_tick, x0=x0, left=left)
    
    popt[1] = popt[1]+x0-left
    print('A: %f $\pm$ %f'%(popt[0], perr[0]))
    print('Mean: %f\pm%f'%(popt[1], perr[1]))
    print('Sigma: %f $\pm$ %f'%(popt[2], perr[2]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)

    return popt, perr, rchi

def gaussianFitMul(filePathtobeSaved, x, y, yerr, p0, left=15, right=15, res_tick=[-3,0,3], sigmaFixed=True):
    fig = plt.figure(figsize=(8, 6))
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = yerr[x0-left:x0+right]
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
    popt, pcov = curve_fit(gaussMul, xx, yy, p0=p0, maxfev=500000) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    npara = len(p0)
    rchi, dof = reducedChiSquare(yy, gaussMul(xx, *popt), yerr, npara)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(xx+x0-left, yy, yerr=yerr,fmt='o', elinewidth=1.5 ,capsize=3, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k')
    xxx = np.linspace(min(xx),max(xx),1000)
    plt.plot(xxx+x0-left, gaussMul(xxx, *popt), 'r-', label='Fit', linewidth=2)
    #plt.legend()
    plt.xlabel('MCA Channel Number')
    plt.ylabel('Alpha Decay Channel Counts')
    
    # Plot residuals
    d = yy-gaussMul(xx,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, xx, yerr, res_tick=res_tick, x0=x0, left=left)
    
    if not sigmaFixed:
        a1, a1_e = popt[0], perr[0]
        m1, m1_e = popt[1], perr[1]
        s1, s1_e = popt[2], perr[2]
        a2, a2_e = popt[3], perr[3]
        m2, m2_e = popt[4], perr[4]
        s2, s2_e = popt[5], perr[5]
        a3, a3_e = popt[6], perr[6]
        m3, m3_e = popt[7], perr[7]
        s3, s3_e = popt[8], perr[8]
    else:
        a1, a1_e = popt[0], perr[0]
        m1, m1_e = popt[1], perr[1]
        s1, s1_e = popt[-1], perr[-1]
        a2, a2_e = popt[2], perr[2]
        m2, m2_e = popt[3], perr[3]
        s2, s2_e = popt[-1], perr[-1]
        a3, a3_e = popt[4], perr[4]
        m3, m3_e = popt[5], perr[5]
        s3, s3_e = popt[-1], perr[-1]
        
    print('A 1: %f $\pm$ %f'%(a1, a1_e))
    print('Mean 1: %f\pm%f'%(m1, m1_e))
    print('Sigma 1: %f $\pm$ %f'%(s1, s1_e))
    print('A 2: %f $\pm$ %f'%(a2, a2_e))
    print('Mean 2: %f\pm%f'%(m2, m2_e))
    print('Sigma 2: %f $\pm$ %f'%(s2, s2_e))
    print('A 3: %f $\pm$ %f'%(a3, a3_e))
    print('Mean 3: %f\pm%f'%(m3, m3_e))
    print('Sigma 3: %f $\pm$ %f'%(s3, s3_e))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    
    ax = fig.add_subplot(111)
    ax.annotate('Peak 1', xy=(m1+x0-left, a1+20), xytext=(m1+x0-left, a1+100),\
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
    ax.annotate('Peak 2', xy=(m2+x0-left, a2+20), xytext=(m2+x0-left, a2+100),\
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
    ax.annotate('Peak 3', xy=(m3+x0-left-3, a3), xytext=(m3+x0-left-17, a3),\
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),horizontalalignment='center')
    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)

    return popt, perr, rchi, dof

def fitAlphaPeaks(filePathtobeSaved, filePath, p0, left=100, right=100, res_tick=[-3,0,3]):
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
    popt, perr, rchi, dof = expGaussFitMul(filePathtobeSaved, xx, yy, yerr, p0, x0, left, res_tick)
    popt[3] += x0-left
    popt[7] += x0-left
    print('Mean 1 (Not Scaled): %f \pm %f'%(popt[3], perr[3]))
    print('Mean 2 (Not Scaled): %f \pm %f'%(popt[7], perr[7])+'\n')
    
    return popt, perr, rchi, dof

def fitAlphaPeak(filePathtobeSaved, filePath, p0, left=100, right=100, res_tick=[-3,0,3], filenm=None):
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
    print(xx)
    print(x0-left)
    
    xx, xxErr = convertChannelToEnergy(xx)
    x0, x0Err = convertChannelToEnergy(x0)
    left, leftErr = convertChannelToEnergy(left)
    print(xx)
    print(x0-left)
#    p0[3] = convertChannelToEnergy(p0[3])
    
    yerr = np.sqrt(yy)
    popt, perr, rchi, dof = expGaussFit(filePathtobeSaved, xx, yy, yerr, p0, x0, left, res_tick, xerr=xxErr)
    popt[3] += x0-left
    print('File Name: %s'%filenm)
    print('Mean 1 (Not Scaled): %f \pm %f'%(popt[3], perr[3]))
    
    return popt, perr, rchi, dof

def fitAlphaPeaksGaussMul(filePathtobeSaved, filePath, p0, left=100, right=100, res_tick=[-3,0,3], sigmaFixed=True):

    """
    This is the function to fit Alpha Peak
    filepath: full path to the file; p0: list of initial guess; left: how much away
    to the left from peak channel; right: how much away to the right from peak channel
    res_tick: the residual plot y axis ticks
    """

    ch = Chn.Chn(filePath)
    y = ch.spectrum
    x = np.arange(len(y))
    ind = np.argmax(y)
    x0 = x[ind]
    yerr = np.sqrt(y)
    popt, perr, rchi, dof = gaussianFitMul(filePathtobeSaved, x, y, yerr, p0, left, right, res_tick, sigmaFixed=sigmaFixed)
    if not sigmaFixed:
        #a1, a1_e = popt[0], perr[0]
        m1, m1_e = popt[1], perr[1]
        #s1, s1_e = popt[2], perr[2]
        #a2, a2_e = popt[3], perr[3]
        m2, m2_e = popt[4], perr[4]
        #s2, s2_e = popt[5], perr[5]
        #a3, a3_e = popt[6], perr[6]
        m3, m3_e = popt[7], perr[7]
        #s3, s3_e = popt[8], perr[8]
        popt[1] += x0-left
        popt[4] += x0-left
        popt[7] += x0-left
    else:
        #a1, a1_e = popt[0], perr[0]
        m1, m1_e = popt[1], perr[1]
        #s1, s1_e = popt[-1], perr[-1]
        #a2, a2_e = popt[2], perr[2]
        m2, m2_e = popt[3], perr[3]
        #s2, s2_e = popt[-1], perr[-1]
        #a3, a3_e = popt[4], perr[4]
        m3, m3_e = popt[5], perr[5]
        #s3, s3_e = popt[-1], perr[-1]
        popt[1] += x0-left
        popt[3] += x0-left
        popt[5] += x0-left
        
    print('Mean 1 (Not Scaled): %f \pm %f'%(m1+x0-left, m1_e))
    print('Mean 2 (Not Scaled): %f \pm %f'%(m2+x0-left, m2_e))
    print('Mean 3 (Not Scaled): %f \pm %f'%(m3+x0-left, m3_e)+'\n')
    
    
    
    return popt, perr, rchi, dof

####################################################################################################################
################################# Transform from channel number data to energy #####################################
####################################################################################################################

def convertChannelToEnergy(channelData, err=0, noErr=False):
    m = slope
    b = intercept
    m_e = slopeErr
    b_e = interceptErr
    energyData = m*channelData + b
    errProp = np.sqrt((m*err)**2+(m_e*channelData)**2+b_e**2)
    if not noErr:
        return energyData, errProp
    else:
        return energyData




def calibratePulses(folderName):
    mean, sigma, mean_e, vol, vol_e, rchi = [], [], [], [], [], []
    data = os.listdir(folderName)

    for d in data:
        filePathtobeSaved = 'Figures/Calibration/'+d.split('.')[0]
        ch = Chn.Chn(folderName+'/'+d)
        y = ch.spectrum
        x = np.arange(len(y))
        yerr = np.sqrt(y)
        popt, perr, rchi_temp = gaussianFit(filePathtobeSaved, x, y, yerr, p0=[300, 20, 5], res_tick=[-3,0,3])
        mean.append(popt[1])
        print(type(popt[1]))
        sigma.append(popt[2])
        mean_e.append(perr[1])
        vol.append(int(d.split('_')[0]))
        vol_e.append(int(d.split('_')[2]))
        rchi.append(rchi_temp)

    y = vol
    y = np.asarray(y)/1000 # convert to volts
    x = mean

    xerr = mean_e
    yerr = np.asarray(vol_e)/1000 # convert to volts
    print(yerr)

    ###############For latex#################
    #for i in range(len(vol)):
    #    print('%d & $%.2f\pm%.2f$ & $%.2f\pm0.04$ & $%.2f$ \\\\'%(i+1, y[i], yerr[i], x[i], rchi[i]))
    #########################################
    
    filePathtobeSaved = 'Figures/Calibration/pulseLinear'
    fig = plt.figure(figsize=(8,6))
    plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='o', elinewidth=1.5 ,capsize=4.5, ecolor='b', \
                 label='Data', linestyle='None', markersize=3 ,color='k')

    popt, perr = LinearFit_xIntercept(x, y, yerr)

    m = popt[0]
    h = popt[1]
    m_e = perr[0]
    h_e = perr[1] # y-intercept
    npara = 2
    x = np.asarray(x)     # this line of code saved my life, and it may save your life with this error
                            # 'numpy.float64' object cannot be interpreted as an integer - Alvin
    y = np.asarray(y)
    yerr = np.asarray(yerr) 
    rchi, dof = reducedChiSquare(y, m*x-m*h*np.ones(len(x)), yerr, npara)
    xx = np.linspace(0, max(x))
    plt.plot(xx, m*xx-m*h*np.ones(len(xx)), color='r', label='Fit', linewidth=2)

    plt.xlabel('Mean MCA Channel Number')
    plt.ylabel('Pulser Voltage (V)')
    print('x-intercept: %f $\pm$ %f'%(h,h_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    plt.legend()
    
    d = y-(x*m-m*h*np.ones(len(x)))
    axes = plt.gca()
    plotStudRes(axes, d, x, yerr, res_tick=[-1,0,1])
    
    func = 'm=slope x-intercept=b'
    func = func.replace('b','('+str(int(round(popt[1],0)))+'$\pm$'+str(int(round(perr[1],0)))+')')
    func = func.replace('slope','('+str(round(popt[0],5))+'$\pm$'+str(round(perr[0],5))+')')
    
#    textstr = '$\chi^2$=%.2f\tDOF=%d\t%s'%(rchi, dof, func)
#    plt.text(0.1, 0.9, textstr, fontsize=10, transform=plt.gcf().transFigure)
    
    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)

    global E0 # americium energy needed for calibration
    global E0Err
    global calibIntercept #intercept on channel number versus voltage. beta in our notes.
    global calibInterceptErr
    global N0
    global N0Err
    global slope #slope on energy versus channel number
    global slopeErr
    global intercept #intercept on energy versus channel number
    global interceptErr
    
    E0 = 5.48556
    E0Err = 0.00012

    calibInterceptErr = h_e
    calibIntercept = h
    
    popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
                         [8, 20, 60, 30, 310, 40, 3], left=50, right=20, res_tick=[-2,0,2], sigmaFixed=True)
    N0, N0Err = popt_am[-2], perr_am[-2]
    
    print(N0)

    slope = E0/(N0-calibIntercept)
    slopeErr = slope*np.sqrt((E0Err/E0)**2+(1/(N0-calibIntercept))**2*(calibInterceptErr**2+N0Err**2))

    intercept = E0*calibIntercept/(calibIntercept-N0)
    interceptErr = intercept*np.sqrt((E0Err/E0)**2+(N0*calibInterceptErr/(calibIntercept*(calibIntercept-N0)))**2+(N0Err/(calibIntercept-N0))**2)

    print('\nAmericium Energy: '+str(E0)+' \pm '+str(E0Err)+' MeV')
    print('Beta intercept: ' + str(calibIntercept) + ' \pm ' + str(calibInterceptErr))
    print('Calibration Slope, m: ' + str(slope) + ' \pm ' + str(slopeErr))
    print('Calibration Intercept, b: ' + str(intercept) + ' \pm ' + str(interceptErr))

    return m, h, m_e, h_e





####################################################################################################################
############################################# Stopping power calculation ###########################################
####################################################################################################################

def pressureData(folderName):
    """
    This function reads all pressure varied data from the directory and fit linearly;
    folderName: string of the folder name under root directory
    """
    outPath = 'Figures/Pressure'
    
    peak_means, peak_means_e = [], []
    data = os.listdir(folderName)
    pressure = []
    
    for d in data:
        p = d.split('_')[0]
        pressure.append(int(p))
    pressure = np.asarray(pressure)
    for i in range(len(pressure)):
        if pressure[i]==50:
            pressure[i] = pressure[i]*0.0013332237
    print(pressure)
        
    for file in data:
        popt, perr, rchi, dof = fitAlphaPeak(outPath+'/'+file, folderName+'/'+file, p0=[60, 1, 0.7, 10],\
                                          right=80, filenm=file)
        peak_means.append(popt[3])
        peak_means_e.append(perr[3])
        
    fig = plt.figure(figsize=(8,6))
    x = pressure
    y = peak_means
    yerr = peak_means_e
    popt, perr = linearFit(x, y, yerr)
    m = popt[0]
    b = popt[1]
    m_e = perr[0]
    b_e = perr[1]
    xx = np.linspace(min(x), max(x))
    plt.plot(xx, m*xx+b*np.ones(len(xx)),label='Fit', color='r')
    plt.errorbar(x, y, yerr=yerr, fmt='+', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=4,color='b')
    npara = 2
    rchi, dof = reducedChiSquare(y, m*x+b*np.ones(len(x)), yerr, npara)
    plt.xlabel('Pressure (mBar)')
    plt.ylabel('Energy (MeV)')
    
    d = y-(m*x+b*np.ones(len(x)))
    axes = plt.gca()
    plotStudRes(axes, d, x, yerr, res_tick=[-5,0,5])
    
    print('\nIntercept: %f $\pm$ %f'%(b,b_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    print('r-chi-square: %.2f'%rchi)
    print('DOF: %d'%dof)
    
    plt.show()
    fig.savefig(outPath+'/StoppingPower.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    #Target thickness is (air density) * (distance b/w source and detector)
    #Using the names given in the notebook, the distance between the source and the
    #detector is Distance = L - length of detector - (length of source - bottom half of source)
    #I multiply by 100 to get the distance in centimeters
    Distance = (89.530 - 15.810 - (13.890-9.380))/10.0
    #The error, since we simply add 4 values with 0.005 mm error, is given by the following
    #(once again, in cm)
    Distance_e = ((4.0*(0.005**2.0))**0.5)/10.0
    #Air density rho_air is (M_air*pressure)/(R*T)
    #The value and error of M_air are referenced in the .bib file
    M_air =  28.964 #in [g] [mol-1]
    M_air_e =  0.002 #in [g] [mol-1]
    #The value and error of R are referenced in the .bib file
    R = 83.14449*1000.0  #in [cm3] [mbar] [K−1] [mol−1]
    R_e =  0.00056*1000.0 #in [cm3] [mbar] [K−1] [mol−1] 
    #T was measured as multiple values between 21.0 and 22.6, so for now take 21.8
    T = 21.8+273.16 #in [K]
    T_e = 0.1 #in [K]
    pressure_e = 10.0 #in [mbar]
    #Thickness is defined as M_air*pressure*Distance/(R*T)
    Thickness = []
    Thickness_e = []
    for i in range(len(pressure)):
        Thickness.append( M_air*pressure[i]*Distance/(R*T) ) #in [g] [cm-2]
        Thickness_e.append( ( (M_air_e * pressure[i]*Distance/(R*T))**2 + (pressure_e * M_air*Distance/(R*T))**2 + (Distance_e * M_air*pressure[i]/(R*T))**2 + (R_e * M_air*pressure[i]*Distance/(R*R*T))**2 + (T_e * M_air*pressure[i]*Distance/(R*T*T))**2 )**0.5 ) #in [g] [cm-2]
    

    ## Reshuffle data to be in order of increasing thickness

    Energy = peak_means
    Energy_e = peak_means_e

    data = [(Thickness[n], Energy[n], Thickness_e[n], Energy_e[n]) for n in range(len(Thickness))]

    data = sorted(data, key=lambda Thickness: Thickness[2])
    print(data)
    Thickness = [data[n][0] for n in range(len(data))]
    Energy = [data[n][1] for n in range(len(data))]
    Thickness_e = [data[n][2] for n in range(len(data))]
    Energy_e = [data[n][3] for n in range(len(data))]

    return m, m_e, b, b_e, Energy, Energy_e, Thickness, Thickness_e

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
    for n in range(1,len(x)):
        X.append((x[n]+x[n-1])/2)
        Xerr.append((xerr[n]+xerr[n-1])/2)
    print(len(X))
    Y=[]
    Yerr=[]
    for n in range(1,len(x)):
        Y.append((y[n]-y[n-1])/(x[n]-x[n-1]))
        Yerr.append(Y[n-1]*np.sqrt(((xerr[n]**2+xerr[n-1]**2)/(x[n]-x[n-1])**2)+((yerr[n]**2+yerr[n-1]**2)/(y[n]-y[n-1])**2)))
    print(len(Y)==len(y)-1)

    return X,Y,Xerr,Yerr

def calculateStoppingPower(folderName):
    # constants
    Distance = (89.530 - 15.810 - (13.890-9.380))/10.0
    Distance_e = ((4.0 * (0.005 ** 2.0)) ** 0.5) / 10.0
    pressure_e = 10.0  # in [mbar]
    prefactor = 1/(Distance) # such that the spotting power is just - prefactor * thickness * energy/thickness local derivative :)
    prefactor_e = Distance_e/(Distance)**2

    #summon the thickness versus energy data (properly ordered)
    m_press, m_press_e, b_press, b_press_e, Energy, Energy_e, Thickness, Thickness_e = pressureData('PressureWBias_1')

    # take the slopes of adjacent points to create a "local derivative" dataset
    x, y, xerr, yerr = locallyDifferentiate(Thickness, Energy, Thickness_e, Energy_e)

    #suggestively relabel
    t = np.asarray(x)
    dEdt = np.asarray(y)

    t_err = np.asarray(xerr)
    dEdt_err = np.asarray(yerr)

    #calculate the stopping power :DDD
    S = [-prefactor*t[n]*dEdt[n] for n in range(len(t))]
    #propagate the error
    Serr = [S[n]*np.sqrt((t_err[n]/t[n])**2+(prefactor_e/prefactor)**2+(dEdt_err[n]/dEdt[n])**2) for n in range(len(t))]
    
    # plot away
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(t, S, yerr=Serr, xerr=t_err, fmt='+', elinewidth=1, capsize=2, ecolor='b', label='Data', linestyle='None', markersize=4, color='b')

    plt.xlabel(r'Thickness (g$\cdot$cm$^{-2}$)')
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.ylabel('Stopping Power (MeV/cm)')


####################################################################################################################
################## Fit bismuth activity data in order to extract lead and bismuth half-lives #######################
####################################################################################################################

def activityFitFunc(x, lambda1, lambda2, N0, N1):
    return N0*lambda1*lambda2*(np.exp(-lambda1*x)-np.exp(-lambda2*x))/(lambda2-lambda1)+N1*np.exp(-lambda2*x)

def activityFit(filePathtobeSaved, x, y, yerr, guess):
    
    fig = plt.figure(figsize=[8,6])
    popt, pcov = curve_fit(activityFitFunc, x, y, p0=guess, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    npara = 4
    rchi, dof = reducedChiSquare(y, activityFitFunc(x, *popt), yerr, npara)
    plt.errorbar(x, y, yerr=yerr, fmt='+', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=4,color='b')
    xx = np.linspace(min(x), max(x))
    plt.plot(xx, activityFitFunc(xx, *popt), '-r', label='Fit')
    plt.legend()
    plt.ylabel('Activity ($s^{-1}$)')
    plt.xlabel('Time (s)')
    print('lambda1: %f \pm %f'%(popt[0], perr[0]))
    print('lambda2: %f \pm %f'%(popt[1], perr[1]))
    print('N0: %f \pm %f'%(popt[2], perr[2]))
    print('N1: %f \pm %f'%(popt[3], perr[3]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    
    # Plot residuals
    d = y-activityFitFunc(x,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, x, yerr, res_tick=[-2,0,2])
    

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)

    return popt, perr, rchi, dof

def halflifeMeasurement(outFileName, folderName):
    
    filePathtobeSaved = 'Figures/HalfLifeMeasurement/'+outFileName
    files = os.listdir(folderName)
    files.sort()
    activity, elapse = [], []
    for f in files:
        ch = Chn.Chn(folderName+'/'+f)
        spectrum = ch.spectrum
        activity.append(np.float64(sum(spectrum)))
        fileInd = float(f.split('_')[0])
        elapse.append(fileInd*10*60) #10 minutes, starts at 0
    
    x = np.asarray(elapse)
    # have not converted to energy
    y = np.asarray(activity)
    yerr = np.sqrt(activity)
    
    popt, perr, rchi, dof = activityFit(filePathtobeSaved, x, y, yerr, [1.81e-5,1.9e-4,1.0e5,1.0e2])
    
    l1, l2 = popt[0], popt[1]
    l1_e, l2_e = perr[0], perr[1]
    
    T1 = np.log(2)/l1/3600 # in hour
    T2 = np.log(2)/l2/3600 # in hour
    T1_e = np.log(2)/3600/l1**2*l1_e
    T2_e = np.log(2)/3600/l2**2*l2_e
    print('Tau1: %.1f \pm %.1f'%(T1, T1_e))
    print('Tau1: %.1f \pm %.1f'%(T2, T2_e))
    
    return T1, T1_e, T2, T2_e

def branchingRatio_FourPeaks(InFileName):
    outFileName = 'Figures/BranchingRatio/'+InFileName 
    spectrum = chnsum(InFileName)
    left = 1200
#    left = 1280
    right = 1330
#    right = 1280
    leftSpec = spectrum[left:right]
    y = leftSpec
    yerr = np.sqrt(y)
    x = np.arange(left,right)
    plt.plot(x,leftSpec)
    # Guess of fitting 5 peaks
    p0 = [600, 0.1, 2.5, left+20, 600, 0.1, 1.5, left+50, 15000, 0.3, 1.5, left+115, 6000, 0.1, 1.5, left+125, 5000, 0.3, 1.3, left+105] # A, l, s, m
    # Guess of fitting 4 peaks
#    p0 = [700, 0.1, 2.5, left+20, 7000, 0.1, 1.5, left+50, 15000, 0.3, 1.5, left+115, 6000, 0.3, 1.5, left+125]
#    p0 = [10000, 0.3, 1.3, left+35, 4000, 0.3, 1.3, left+45, 3000, 1, 1.3, left+15, 2000, 1, 1.3, left+25] # A, l, s, m
#    p0 = [500, 0.1, 2, left+20, 600, 0.1, 2, left+55] # A, l, s, m
    popt, perr, rchi, dof = expGaussFitMul(outFileName, x, y, yerr, p0=p0, x0=0, left=0, res_tick=[-2,0,2])

    # Plot each component convolution
    for i in range(int(len(popt)/4)):
        temp = popt[i*4:(i+1)*4]
        plt.plot(x, expGauss(x, *temp))

    #convert mean channels to energies
    for i in range(0, len(popt), 4):
        popt[i+3], perr[i+3] = convertChannelToEnergy(popt[i+3], err=perr[i+3])



    # make fitted parameter into a matrix, as well as the fitting error in another matrix
    l = int(len(popt)/4)
    valueToReturn, errToReturn = np.zeros((l,4)), np.zeros((l,4))
    for i in range(l):
        valueToReturn[i] = popt[4*i:4*i+4]
        errToReturn[i] = perr[4*i:4*i+4]
    print(valueToReturn)
    print(errToReturn)
    
    return valueToReturn, errToReturn
    
    
def branchingRatio_Largest(InFileName):
    outFileName = 'Figures/BranchingRatio/'+InFileName 
    spectrum = chnsum(InFileName)
    left = 1830
    right = 1852
    leftSpec = spectrum[left:right]
    y = leftSpec
    yerr = np.sqrt(y)
    x = np.arange(left,right)
    plt.plot(x,leftSpec)
    p0 = [35000, 1, 1, left+20, 12000, 1, 1, left+15]
    popt, perr, rchi, dof = expGaussFitMul(outFileName, x, y, yerr, p0=p0, x0=0, left=0, res_tick=[-2,0,2])
    
    # Plot each component convolution
    for i in range(int(len(popt)/4)):
        temp = popt[i*4:(i+1)*4]
        plt.plot(x, expGauss(x, *temp))

    # convert mean channels to energies
    for i in range(0, len(popt), 4):
        popt[i+3], perr[i+3] = convertChannelToEnergy(popt[i+3], err=perr[i+3])

    print(popt)
    # make fitted parameter into a matrix, as well as the fitting error in another matrix
    l = int(len(popt)/4)
    print(l)
    valueToReturn, errToReturn = np.zeros((l,4)), np.zeros((l,4))
    for i in range(l):
        valueToReturn[i] = popt[4*i:4*i+4]
        errToReturn[i] = perr[4*i:4*i+4]
        print(popt[4*i:4*i+4])
#    print(valueToReturn)
#    print(errToReturn)
#    valueToReturn[0], valueToReturn[1] = valueToReturn[1], valueToReturn[0]
#    errToReturn[0], errToReturn[1] = errToReturn[1], errToReturn[0]
#    print(valueToReturn)
#    print(errToReturn)
    
    return valueToReturn, errToReturn

####################################################################################################################
#################################### Branching ratios calculation ##################################################
####################################################################################################################

def integrateExpGauss(params):
    # params is the vector (A,lambda, sigma, mu) describing the expgaussian integrand
    return quad(expGauss, 1200, 1500, args=(params[0],params[1],params[2],params[3]))[0]

def diffExpGaussSigma(x,params):
    #returns the derivative of ExpGaussFunction with respect to sigma as function of x
    A = params[0]
    l = params[1]
    s = params[2]
    m = params[3]
    return A*l/(2*s**2)*np.exp(-(x-m)**2/(2*s**2))*(np.sqrt(2/np.pi)*(x-m-l*(s**2))+(l**2)*(s**3)*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s))))

def integrateDiffExpGaussSigma(params):
    return quad(diffExpGaussSigma, 1200, 1500, args=(params))[0]


def diffExpGaussLambda(x, params):
    # returns the derivative of ExpGaussFunction with respect to lambda as function of x
    A = params[0]
    l = params[1]
    s = params[2]
    m = params[3]

    return A/2*np.exp(l/2*(2*x-2*m+l*s*s))*((1+l*(x-m+l*s**2))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))-np.sqrt(2/np.pi)*l*s*(np.exp(-(x-m+l*s**2)**2/(2*s**2))))

def integrateDiffExpGaussLambda(params):
    return quad(diffExpGaussLambda, 1200, 1500, args=(params))[0]


def diffExpGaussA(x,params):
    # returns the derivative of ExpGaussFunction with respect to lambda as function of x
    l = params[1]
    s = params[2]
    m = params[3]
    return l/2*np.exp(l/2*(2*x-2*m+l*s*s))*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s)))

def integrateDiffExpGaussA(params):
    return quad(diffExpGaussA, 1200, 1500, args=(params))[0]


def diffExpGaussMu(x,params):
    # returns the derivative of ExpGaussFunction with respect to mu as function of x
    A = params[0]
    l = params[1]
    s = params[2]
    m = params[3]

    return A*l/(2*np.sqrt(np.pi)*s)*np.exp(l/2*(2*x-2*m+l*s*s))*(np.sqrt(2)*(np.exp(-(x-m+l*s**2)**2/(2*s**2)))-np.sqrt(np.pi)*l*s*(1-sp.special.erf((x+l*s*s-m)/(math.sqrt(2)*s))))

def integrateDiffExpGaussMu(params):
    return quad(diffExpGaussMu, 1200, 1500, args=(params))[0]

def calculateBranchRatio(Params,ParamErrs):
    # Params is a lists of lists. Each embedded list is a set of parameters for a single transition fit, [A,l,s,m]
    # ParamErrs is similarly a list of lists of the associated errors on the fits of each parameters
    # Assume the parameter vectors are orderred in the order of increasing energy
    b=[integrateExpGauss(Params[i-1]) for i in range(1,6)]
    totalArea = sum(b)
    totalDiffA= sum([integrateDiffExpGaussA(Params[i-1]) for i in range(1,6)])
    totalDiffLambda = sum([integrateDiffExpGaussLambda(Params[i-1]) for i in range(1,6)])
    totalDiffSigma = sum([integrateDiffExpGaussSigma(Params[i-1]) for i in range(1,6)])
    totalDiffMu = sum([integrateDiffExpGaussMu(Params[i-1]) for i in range(1,6)])
    b =np.array(b)

    print(b)
    bErr=[] #array of error on branching ratios
    for i in range(1,6):
        # propagate the error
        err = 0.0
        for n in range(1,6): #all fitted function parameter errors propogate so loop through all of them
            paramErrs = ParamErrs[n-1]
            if n==i:
                derivsA = [(integrateDiffExpGaussA(Params[i-1])*totalArea-totalDiffA*integrateExpGauss(Params[i-1]))/(totalArea**2),
                           (integrateDiffExpGaussLambda(Params[i-1])*totalArea-totalDiffLambda*integrateExpGauss(Params[i-1]))/(totalArea**2),
                          (integrateDiffExpGaussSigma(Params[i-1])*totalArea-totalDiffSigma*integrateExpGauss(Params[i-1]))/(totalArea**2),
                (integrateDiffExpGaussMu(Params[i-1])*totalArea-totalDiffMu*integrateExpGauss(Params[i-1]))/(totalArea**2)]
                #derivsA and derivsB represent the different derivtives that distinquish whether or not the index of the branching function corresponds to the index of the
                # propogated function
                err = err + sum([(derivsA[j-1]*paramErrs[j-1])**2 for j in range(1,5)])
            else:
                derivsB = [(integrateExpGauss(Params[i-1])*integrateDiffExpGaussA(Params[n-1]))/(totalArea**2),
                           (integrateExpGauss(Params[i-1])*integrateDiffExpGaussLambda(Params[n-1]))/(totalArea**2),
                           (integrateExpGauss(Params[i-1])*integrateDiffExpGaussSigma(Params[n-1]))/(totalArea**2),
                           (integrateExpGauss(Params[i-1]) * integrateDiffExpGaussMu(Params[n-1]))/(totalArea ** 2)]
                err = err + sum([(derivsB[j-1]*paramErrs[j-1])**2 for j in range(1,5)])
        bErr.append(np.sqrt(err))

    #b = np.array(b)
    #print(b)

    # add weighted contribution of the anomolous extra peak to the ratios of the two closely spaced peaks

    b[2]=b[2]+(b[2]/(b[2]+b[3]))*b[4]
    b[3]=b[3]+(b[3]/(b[2]+b[3]))*b[4]

    # propogate the errors one last time

    deriva = [1+(b[4]*b[3])/(b[2]+b[3])**2, b[2]*b[4]/(b[2]+b[3])**2, b[2]/(b[2]+b[3])]
    derivb = [1+(b[4]*b[2])/(b[2]+b[3])**2, b[3]*b[4]/(b[2]+b[3])**2, b[3]/(b[2]+b[3])]

    bErr[2]=np.sqrt(sum([(deriva[n]*bErr[n+2])**2 for n in range(0,3)]))
    bErr[3] = np.sqrt(sum([(derivb[n]*bErr[n+2])**2 for n in range(0,3)]))

    #elimate the last peak
    b = b[:-1]
    bErr = bErr[:-1]

    print(b)
    print(bErr)


    totalArea = sum(b)
    TOTAL = totalArea

    totalAreaErr = np.sqrt(sum([bErr[n]**2 for n in range(4)]))
    TOTALErr = totalAreaErr


    B = np.multiply((1 / totalArea), b)

    bErr = np.array(bErr)

    bErr = [B[n]*np.sqrt((totalAreaErr/totalArea)**2+(bErr[n]/b[n])) for n in range(4)]

    bErr = np.array(bErr)

    print(sum(B))
    print("Branching ratios: "+str(B))
    print("Errors on ratios: "+str(bErr))

####################################################################################################################
############################################## Function Calling Area ###############################################
####################################################################################################################
    
#m_calib, m_calib_e, b_calib, b_calib_e = calibratePulses('CalibrationWBias_2')
#m_press, m_press_e, b_press, b_press_e, Energy, Energy_e, Thickness, Thickness_e = pressureData('PressureWBias_1')

#calculateStoppingPower('PressureWBias_1')


#popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaks("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [100, 2.5, 2, 20, 785, 2.5, 1.8, 30, 1750, 2, 1.6, 40], left=40, right=20, res_tick=[-2,0,2])
#popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [8, 20, 60, 30, 310, 40, 3], left=50, right=20, res_tick=[-2,0,2], sigmaFixed=True)
#popt_am, perr_am, rchi_am, dof_am= fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [8, 50, 3, 60, 60, 3, 310, 70, 2], left=70, right=30, res_tick=[-2,0,2])

#halflifeMeasurement('OneDayCollectionTime', 'Decay_3')



#Values = branchingRatio_FourPeaks('Decay_3')[0]
#Errs = branchingRatio_FourPeaks('Decay_3')[1]
#calculateBranchRatio(Values,Errs)
branchingRatio_Largest('Decay_3')


############### For Vincent to have fun with ####################################
#popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [6, 10, 60, 20, 310, 30, 3, 0.01], left=40, right=20, res_tick=[-2,0,2], sigmaFixed=True)
#print(popt_am[-1])
###############################################################################3

#print(convertChannelToEnergy(1183))



