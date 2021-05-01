# -*- coding: utf-8 -*-
"""|
Created on Fri Apr 30 19:21:24 2021

@author: felip
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

#DATOS EXPERIMENTALES TABLA GUIÓN#
y = np.linspace(0.1,1.0,10)
x = [0.2,0.3,0.47,0.55,0.6,0.7,0.75,0.8,0.85,0.9]

## -------------------------- funcion Weibull ------------------- ##

def Weibull(y,x):
    x_1 = 1-np.exp(-(x*y))
    return x_1

plt.plot(Weibull(y,x),label='Weibull')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()

print("weibull: \n ",Weibull(y,x))
