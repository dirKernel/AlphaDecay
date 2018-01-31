import numpy as np
import Chn
import pylab as plb
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5 #set the value globally
mpl.rcParams.update({'font.size': 13})
from scipy.optimize import curve_fit
import scipy as sp
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from chnsum import chnsum
import uncertainties as unc  
import uncertainties.unumpy as unp  

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

def plotStudRes(axes, d, xx , yerr, res_tick, x0=None, left=None):
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
    stu_d = d/yerr
    stu_d_err = np.ones(len(d))
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.1)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks(res_tick)
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(xx+x0-left, stu_d, yerr=stu_d_err, fmt='+', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=4,color='b')

def reducedChiSquare(y,fx,yerr, npara):
    """
    :param y: y vector
    :param fx: vector of f(x), where f is the fitting function
    :param m: the number of fitted parameters
    :return: Reduced chi^2 of the fit. Ideally should be 1, dof the degree of freedom.
    """
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

################ For Vincent to have fun with ################################# 
#def gaussMul(x, *params, sigmaFixed=True, overallConst=True):
#    y = np.zeros_like(x)
#    if not sigmaFixed:
#        for i in range(0, len(params), 3):
#            a = params[i]
#            mean = params[i+1]
#            sigma = params[i+2]
#            y = y + a*np.exp(-(x-mean)**2/(2*sigma**2))
#    else:
#        if not overallConst:
#            for i in range(0, len(params)-1, 2):
#                a = params[i]
#                mean = params[i+1]
#                sigma = params[-1]
#                y = y + a*np.exp(-(x-mean)**2/(2*sigma**2))
#        else:
#            for i in range(0, len(params)-2, 2):
#                a = params[i]
#                mean = params[i+1]
#                sigma = params[-2]
#                y = y + a*np.exp(-(x-mean)**2/(2*sigma**2))
#            y = y + params[-1]
#            
#    return y
###############################################################################

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
    popt, pcov = curve_fit(expGaussMul, x, y, p0=p0, maxfev=50000)
    npara = len(p0)/4
    rchi, dof = reducedChiSquare(y, expGaussMul(x, *popt), yerr, npara)
    plt.errorbar(x+x0-left, y, yerr=yerr,fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    plt.plot(x+x0-left, expGaussMul(x, *popt), '-r', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    perr = np.sqrt(np.diag(pcov))
    print('\nMean 1 (Scaled): %f $\pm$ %f'%(popt[3], perr[3]))
    print('Sigma 1: %f $\pm$ %f'%(popt[2], perr[2]))
    print('Lambda 1: %f $\pm$ %f'%(popt[1], perr[1]))
    print('A 1: %f $\pm$ %f'%(popt[0], perr[0]))
    print('Mean 2 (Scaled): %f $\pm$ %f'%(popt[7], perr[7]))
    print('Sigma 2: %f $\pm$ %f'%(popt[6], perr[6]))
    print('Lambda 2: %f $\pm$ %f'%(popt[5], perr[5]))
    print('A 2: %f $\pm$ %f'%(popt[4], perr[4]))
    print('Mean 3 (Scaled): %f $\pm$ %f'%(popt[11], perr[11]))
    print('Sigma 3: %f $\pm$ %f'%(popt[10], perr[10]))
    print('Lambda 3: %f $\pm$ %f'%(popt[9], perr[9]))
    print('A 3: %f $\pm$ %f'%(popt[8], perr[8]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
            
    # Plot residuals
    d = y-expGaussMul(x,*popt)
    stu_d = d/yerr# studentized residual
    stu_d_err = np.ones(len(d))
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.1)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks(res_tick)
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(x+x0-left, stu_d, yerr=stu_d_err, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    
    textstr = '$\chi^2$=%.2f\tDOF=%d'%(rchi, dof)
    plt.text(0.1, 0.9, textstr, fontsize=9, transform=plt.gcf().transFigure)

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    return popt, perr, rchi, dof # return the mean channel values  

def expGaussFit(filePathtobeSaved, x, y, yerr, p0, x0, left, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=(8, 6))
    popt, pcov = curve_fit(expGauss, x, y, p0=p0, maxfev=50000)
    npara = len(p0)
    rchi, dof = reducedChiSquare(y, expGauss(x, *popt), yerr, npara)
    plt.errorbar(x+x0-left, y, yerr=yerr,fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    plt.plot(x+x0-left, expGauss(x, *popt), '-r', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    perr = np.sqrt(np.diag(pcov))
    print('\nMean 1 (Scaled): %f $\pm$ %f'%(popt[3], perr[3]))
    print('Sigma 1: %f $\pm$ %f'%(popt[2], perr[2]))
    print('Lambda 1: %f $\pm$ %f'%(popt[1], perr[1]))
    print('A 1: %f $\pm$ %f'%(popt[0], perr[0]))
    print('RChi: %f'%(rchi))
    print('DOF: %d'%(dof))
    
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
            
    # Plot residuals
    d = y-expGauss(x,*popt)
    stu_d = d/yerr# studentized residual
    stu_d_err = np.ones(len(d))
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.1)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks(res_tick)
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(x+x0-left, stu_d, yerr=stu_d_err, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    
    textstr = '$\chi^2$=%.2f\tDOF=%d'%(rchi, dof)
    plt.text(0.1, 0.9, textstr, fontsize=9, transform=plt.gcf().transFigure)

    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    return popt, perr, rchi, dof # return the mean channel values  

def gaussianFit(filePathtobeSaved, x, y, yerr, p0=[300, 20, 2.5], left=15, right=15, res_tick=[-3,0,3]):
    fig = plt.figure(figsize=[6,3.5])
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = yerr[x0-left:x0+right]
    popt, pcov = curve_fit(gauss, xx, yy, p0=p0, maxfev=50000) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    npara = len(p0)
    rchi, dof = reducedChiSquare(yy, gauss(xx, *popt), yerr, npara)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(xx+x0-left, yy, yerr=yerr,fmt='x', elinewidth=1.5 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5, color='k')
    xxx = np.linspace(min(xx),max(xx),1000)
    plt.plot(xxx+x0-left, gauss(xxx, *popt), 'r-', label='Fit')
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
            
    # Plot residuals
    d = yy-gauss(xx,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, xx, yerr, res_tick=res_tick, x0=x0, left=left)
    
    func = 'A*exp(-(x-mean)^2/(2*sigma^2))'
    func = func.replace('A','('+str(int(popt[0]))+'$\pm$'+str(int(perr[0]))+')')
    func = func.replace('mean','('+str(round(popt[1]+x0-left,3))+'$\pm$'+str(round(perr[1],3))+')')
    func = func.replace('sigma','('+str(round(popt[2],1))+'$\pm$'+str(round(perr[2],1))+')',3)
    
    textstr = '$\chi^2$=%.2f\tDOF=%d\t$\mu$=%.2f$\pm$%.2f'%(rchi, dof, popt[1]+x0-left, perr[1])
    plt.text(0.1, 0.9, textstr, fontsize=10, transform=plt.gcf().transFigure)
    
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
    fig = plt.figure(figsize=[8, 6])
    ind = np.argmax(y) #to get the peak value x-coord
    x0 = x[ind] #x0 is the peak value x-coord (channel number)
    yy = y[x0-left:x0+right]
    xx = np.arange(len(yy))
    yerr = yerr[x0-left:x0+right]
    popt, pcov = curve_fit(gaussMul, xx, yy, p0=p0, maxfev=500000) #initial guess of the amplitude is 100, mean is x0 and variance (sigma) 5
    npara = len(p0)/3
    rchi, dof = reducedChiSquare(yy, gaussMul(xx, *popt), yerr, npara)
    perr = np.sqrt(np.diag(pcov))
    plt.errorbar(xx+x0-left, yy, yerr=yerr,fmt='+', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=4,color='b')
    xxx = np.linspace(min(xx),max(xx),1000)
    plt.plot(xxx+x0-left, gaussMul(xxx, *popt), 'r-', label='Fit', linewidth=1.5)
    plt.legend()
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    
    for i in range(len(yerr)):
        if yerr[i]==0:
            yerr[i] = 1
    
    # Plot residuals
    d = yy-gaussMul(xx,*popt)
    axes = plt.gca()
    plotStudRes(axes, d, xx, yerr, res_tick=res_tick, x0=x0, left=left)
    
    textstr = '$\chi^2$=%.2f\tDOF=%d'%(rchi, dof)
    plt.text(0.1, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)
    
    print(sigmaFixed)
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
    yerr = np.sqrt(yy)
    popt, perr, rchi, dof = expGaussFit(filePathtobeSaved, xx, yy, yerr, p0, x0, left, res_tick)
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
        
    print('Mean 1 (Not Scaled): %f \pm %f'%(m1+x0-left, m1_e))
    print('Mean 2 (Not Scaled): %f \pm %f'%(m2+x0-left, m2_e))
    print('Mean 3 (Not Scaled): %f \pm %f'%(m3+x0-left, m3_e)+'\n')
    
    return popt, perr, rchi, dof

################################# Transform from channel number data to energy ###########################################




def convertChannelToEnergy(channelData):




    energyData = m*channelData + b*np.ones(len(channelData))

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
    for i in range(len(vol)):
        print('%d & $%.2f\pm%.2f$ & $%.2f\pm0.04$ & $%.2f$ \\\\'%(i+1, y[i], yerr[i], x[i], rchi[i]))
    #########################################
    
    filePathtobeSaved = 'Figures/Calibration/pulseLinear'
    fig = plt.figure(figsize=[8,6])
    plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
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
    plt.plot(xx, m*xx-m*h*np.ones(len(xx)), color='r', label='Fit')

    plt.xlabel('Mean Channel')
    plt.ylabel('Voltage (V)')
    print('x-intercept: %f $\pm$ %f'%(h,h_e))
    print('Slope: %f $\pm$ %f'%(m,m_e))
    plt.legend()
    
    d = y-(x*m-m*h*np.ones(len(x)))
    stu_d = d/yerr
    stu_d_err = np.ones(len(x))
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.1)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks([-1,0,1])
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResidual', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes2.errorbar(x, stu_d, yerr=stu_d_err, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    
    func = 'm=slope x-intercept=b'
    func = func.replace('b','('+str(int(round(popt[1],0)))+'$\pm$'+str(int(round(perr[1],0)))+')')
    func = func.replace('slope','('+str(round(popt[0],5))+'$\pm$'+str(round(perr[0],5))+')')
    
    textstr = '$\chi^2$=%.2f\tDOF=%d\t%s'%(rchi, dof, func)
    plt.text(0.1, 0.9, textstr, fontsize=10, transform=plt.gcf().transFigure)
    
    plt.show()
    fig.savefig(filePathtobeSaved+'.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0.0)


    calibInterceptErr = h_e
    calibIntercept = h
    
    popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
                         [8, 20, 60, 30, 310, 40, 3], left=50, right=20, res_tick=[-2,0,2], sigmaFixed=True)
    N0, N0Err = popt_am[-2], perr_am[-2]

    slope = E0/(N0-calibIntercept)
    slopeErr = slope*np.sqrt((E0Err/E0)**2+(1/(N0-calibIntercept))**2*(calibInterceptErr**2+N0Err**2))

    intercept = E0*calibIntercept/(calibIntercept-N0)
    interceptErr = intercept*np.sqrt((E0Err/E0)**2+(N0*calibInterceptErr/(calibIntercept*(calibIntercept-N0)))**2+(N0Err/(calibIntercept-N0))**2)

    print('\nAmericium Energy: '+str(E0)+' \pm '+str(E0Err)+' MeV')
    print('Beta intercept: ' + str(calibIntercept) + ' \pm ' + str(calibInterceptErr))
    print('Calibration Slope, m: ' + str(slope) + ' \pm ' + str(slopeErr))
    print('Calibration Intercept, b: ' + str(intercept) + ' \pm ' + str(interceptErr))

    return m, h, m_e, h_e

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
        popt, perr, rchi, dof = fitAlphaPeak(outPath+'/'+file, folderName+'/'+file, p0=[600, 0.02, 7, 100],\
                                          right=80, filenm=file)
        peak_means.append(popt[3])
        peak_means_e.append(perr[3])
        
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
    
########################### Fit bismuth activity data in order to extract lead and bismuth half-lives ##########################

def activityFitFunc(x, lambda1, lambda2, N0, N1):
    return N0*lambda1*lambda2*(np.exp(-lambda1*x)-np.exp(-lambda2*x))/(lambda2-lambda1)+N1*np.exp(-lambda2*x)

def activityFit(filePathtobeSaved, x, y, yerr, guess):
    
    fig = plt.figure(figsize=[8,6])
    popt, pcov = curve_fit(activityFitFunc, x, y, p0=guess, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    npara = 4
    rchi, dof = reducedChiSquare(y, activityFitFunc(x, *popt), yerr, npara)
    plt.errorbar(x, y, yerr=yerr, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
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
    stu_d = d/np.std(d, ddof=1) # studentized residual
    stu_d_err = yerr/np.std(d, ddof=1)
    axes = plt.gca()
    divider = make_axes_locatable(axes)
    axes2 = divider.append_axes("top", size="20%", pad=0.1)
    axes.figure.add_axes(axes2)
    axes2.set_xlim(axes.get_xlim())
    axes2.set_yticks([-2,0,2])
    axes.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True)
    axes2.tick_params(width=1.3, axis='both', direction='in', bottom=True, top=True, left=True, right=True, labelbottom=False)
    axes2.set_ylabel('Studentized\nResiduals', color='k')
    axes2.axhline(y=0, color='r', linestyle='-')
    axes.tick_params(axis='both', direction='in')
    axes2.tick_params(axis='both', direction='in')
    axes2.errorbar(x, stu_d, yerr=stu_d_err, fmt='x', elinewidth=1 ,capsize=2, ecolor='b', \
                 label='Data', linestyle='None', markersize=5,color='k')
    
    textstr = 'l1=%.5f$\pm$%.5f l2=%.7f$\pm$%.7f\t$\chi^2$=%.2f\tDOF=%d'%\
                (popt[0], perr[0],popt[1], perr[1],rchi, dof)
    plt.text(0.1, 0.9, textstr, fontsize=12, transform=plt.gcf().transFigure)

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

    

######################## Function Calling Area ##################################
    
m_calib, m_calib_e, b_calib, b_calib_e = calibratePulses('CalibrationWBias_2')
#m_press, m_press_e, b_press, b_press_e = pressureData('PressureWBias_1')

#popt_am, perr_am, rchi_am, dof_am= fitAlphaPeaks("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [100, 2.5, 2, 20, 785, 2.5, 1.8, 30, 1750, 2, 1.6, 40], left=40, right=20, res_tick=[-2,0,2])
#popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [8, 20, 60, 30, 310, 40, 3], left=50, right=20, res_tick=[-2,0,2], sigmaFixed=True)
#popt_am, perr_am, rchi_am, dof_am= fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [8, 50, 3, 60, 60, 3, 310, 70, 2], left=70, right=30, res_tick=[-2,0,2])

#m_am, m_am_e = popt_am[3], perr_am[3]
#print('Amerisium Calibration: Mean channel = %f $\pm$ %f\nFit function = %s'%\
#      (m_am, m_am_e, func_am))

#halflifeMeasurement('OneDayCollectionTime', 'Decay_3')
#spectrum = chnsum('Decay_3')


############### For Vincent to have fun with ####################################
#popt_am, perr_am, rchi_am, dof_am = fitAlphaPeaksGaussMul("Figures/Calibration/Americium_300_sec.Chn", "Americium/Americium_300_sec.Chn", \
#                         [6, 10, 60, 20, 310, 30, 3, 0.01], left=40, right=20, res_tick=[-2,0,2], sigmaFixed=True)
#print(popt_am[-1])
###############################################################################3























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



