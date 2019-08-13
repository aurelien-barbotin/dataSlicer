# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:28:28 2017

@author: Aurelien
"""
import numpy as np
from scipy.optimize import curve_fit

def fitter_exp(x, x0, a, b, c):
    return a*np.exp(-b*((x - x0)**2)) + c


def fitter_lorentzian(x,x0,a,b,c):
    return a/(1+(x-x0)**2/b)+c
    
maxcorr=1.0

fitter_lorentzian_bounds = (
    (-maxcorr, 0.0, 0.0, 0.0),
    (maxcorr, np.inf, np.inf, np.inf))

fitter_exp_bounds = (
    (-maxcorr, 0.0, 0.0, 0.0),
    (maxcorr, np.inf, np.inf, np.inf))


# available fitters
fitters_map = {
    'exp': (fitter_exp, fitter_exp_bounds),
    'lorenzian':(fitter_lorentzian, fitter_lorentzian_bounds),
    }

fitters_list = fitters_map.keys()

class Fitter(object):
    def __init__(self,fitter_name,upsampling_factor = 3):
        if fitter_name not in fitters_map:
            raise ValueError(
            'Unknown fitter `{}`; Available fitters: {}'.format(
                fitter_name, ' '.join(fitters_map.keys())))
            
        self.name=fitter_name

        self.fitter = NormalFitter(self.name,upsampling_factor = upsampling_factor)
            
    def fit(self,xdata,ydata,normalise=True):
        return self.fitter.fit(xdata,ydata,normalise=normalise)
    
    def __getattr__(self,attr):
        return getattr(self.fitter,attr)
        
class NormalFitter(object):
    def __init__(self,fitter_name,upsampling_factor = 3):
        if fitter_name not in fitters_map:
            raise ValueError(
            'Unknown fitter `{}`; Available fitters: {}'.format(
                fitter_name, ' '.join(fitters_map.keys())))
            
        self.name=fitter_name
        self.function = fitters_map[fitter_name][0]
        self.bounds = fitters_map[fitter_name][1]
        
        self.upsampling = upsampling_factor
        
    def fit(self,xdata,ydata,normalise=True):
        ydata = ydata.squeeze()
        maxcorr = np.max(xdata)
        mincorr = np.min(xdata)
        
        #Make it a list to allow reassignment
        nbounds=[]
        bound1=[mincorr,maxcorr]
        for i in range(2):
            bd = (bound1[i],*self.bounds[i][1:])
            nbounds.append(bd)
        self.bounds=tuple(nbounds)

        self.xhat = np.linspace(mincorr,maxcorr,self.upsampling * xdata.shape[0])
        if ydata.ndim==2:
            new_yhat = np.zeros((ydata.shape[0]*self.upsampling,ydata.shape[1]))
            new_popt = []
            for j in range(ydata.shape[1]):
                # !!! TODO
                popt,xhat,yhat = self.fit(xdata,ydata[:,j])
                new_yhat[:,j] = yhat
                new_popt.append(popt)
            self.yhat = new_yhat
            self.popt = np.array(new_popt)
            
            return np.median(self.popt,axis=0),self.xhat,self.yhat
        
        try:
            if normalise:
                self.popt, _ = curve_fit(
                    self.function, xdata, ydata/np.max(ydata), p0=None, bounds=self.bounds)
            else:
                self.popt, _ = curve_fit(
                    self.function, xdata, ydata, p0=None, bounds=self.bounds)
            self.yhat = self.function(self.xhat, *self.popt)
            self.fiterr = False
        except:
            self.popt = np.zeros(len(self.bounds[0]))
            self.yhat = np.ones_like(self.xhat)*np.max(ydata)
            self.fiterr = True
            
        return self.popt,self.xhat,self.yhat
   
        
if __name__=="__main__":
    #Test 
    import matplotlib.pyplot as plt
    plt.close("all")
    x  = np.arange(7)
    y = fitter_exp(x,3,1,1,0.2)
    y+=np.random.rand(7)*0.1
    ft = Fitter("exp")
    popt,xhat,yhat = ft.fit(x,y)
    plt.figure()
    plt.plot(x,y,xhat,yhat)
    plt.legend(["Experimental","Fit"])
    
    #test wlt
    P=7
    ndims = 4
    xdata = np.linspace(-0.5,0.5,P)
    x0=-0.2
    cs=[1,-0.2,-0.4,-1]
    ydata = np.zeros((P,ndims))
    for j in range(ndims):
        ydata[:,j]=fitter_quad(xdata,x0,float(j==0),cs[j])
    fitter = Fitter('Wavelet fitter')
    popt,xh,yh=fitter.fit(xdata,ydata,normalise=False)
    plt.figure()
    for j in range(ndims):
        plt.plot(xdata,ydata[:,j],'o')
        plt.plot(xh,yh[:,j],'--')