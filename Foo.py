import spinmob as sm
import numpy as np


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



########################## Fit energy histogram to extract peak energy of the alpha particle ################################

def extractAlphaEnergy(Edata, Ndata):
# x variable is the energy bin and the y variable is the number of events in that bin
    if len(tdata)!= len(Adata):
        print("ERROR: Time and Activity data are not of the same length.")
        return

    f = sm.data.fitter()
    f.set_functions(f='(1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-x0)^2/(2*sig**2))',
                    p='sig,x0')
    params = f.results[0]

#def removeAsymmetry(Edata, Ndata):


########################### Determine stopping power as a function of distance ###############################################

