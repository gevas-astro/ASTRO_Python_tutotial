import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import requests


def getgrb(GRB):
    # file = 'https://www.swift.ac.uk/xrt_curves/00106415/flux.qdp'
    # GRB = '00112453'

    file = 'https://www.swift.ac.uk/xrt_curves/'+GRB+'/flux.qdp'
    file_local = GRB+'_flux.txt'
    print(file_local)
    print(file)

    r = requests.get(file, allow_redirects=True)
    open(file_local, 'wb').write(r.content)

    data0 = np.loadtxt(file,usecols=range(0,6),skiprows=13,comments=['!', 'NO'])
    data = pd.DataFrame(data0,columns=['t','tu','tl','f','fu','fl'])

    mask = (data['fu'] > 0.0)
    mask0 = (data['fu'] == 0.0)
    data_fit = data[mask]
    data_ul = data[mask0]
    
    return data, data_fit, data_ul


def plotgrb(data):
    
    c1 = 'cornflowerblue'
    c2 = 'navy'
    c3 = 'rebeccapurple'
    c4 = '#CF6275'
    c5 = 'maroon'

    
    mask = (data['fu'] > 0.0)
    mask0 = (data['fu'] == 0.0)
    data_fit = data[mask]
    data_ul = data[mask0]
    
    x = data_fit['t']
    xu = data_fit['tu']
    xl = data_fit['tl']

    y= data_fit['f']
    yu = data_fit['fu']
    yl = data_fit['fl']

    x_ul= data_ul['t']
    xu_ul = data_ul['tu']
    xl_ul = data_ul['tl']

    y_ul= data_ul['f']


    plt.figure(figsize=(8,6))
    plt.errorbar(x,y,xerr=[-xl,xu],yerr=[-yl,yu],marker='o'
                 ,markersize=4,color=c1,ls=' ')

    if len(data_ul) > 0:
        plt.errorbar(x_ul, y_ul, xerr=xu_ul,yerr=y_ul/2, uplims=y_ul,ls=' ',color=c4)

    #flipped y axis since showing magnitudes

    plt.yscale('log')
    plt.xscale('log')
    
    