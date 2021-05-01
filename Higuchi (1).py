# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 19:22:14 2021

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

## -------------------- funcion Higuchi -------------------- ##

def Higuchi(y,x):
    x_2 = x*y**(0.5)
    return x_2

plt.plot(Higuchi(y,x),label='Higuchi')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()

print("higuchi: \n ",Higuchi(y,x))