import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def GausFit(X,Y):
	#x = ar(range(10))
	#y = ar([0,1,2,3,4,5,4,3,2,1])

	n = len(x)                          #the number of data
	mean = sum(x*y)/n                   #note this correction
	sigma = sum(y*(x-mean)**2)/n        #note this correction

	popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

	plt.plot(x,y,'b+:',label='data')
	plt.plot(x,gaus(x,*popt),'ro:',label='fit')
	plt.legend()
	plt.title('Fig. 3 - Fit for Time Constant')
	plt.xlabel('Time (s)')
	plt.ylabel('Voltage (V)')
	plt.show()

	return popt[0],popt[1]
