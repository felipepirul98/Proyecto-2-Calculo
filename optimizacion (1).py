# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:04:02 2021


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

## --------------------------------- Funcion Rosenbrock ------------------------ ##


b=10;

f = lambda x,y: (x-1)**2 + b*(y-x**2)**2;


## -------------------------------- Evaluar la funcion -------------------------- ##
X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

## -------------------------------- Graficar ------------------------------------- ##

fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
ros = ax.plot_surface(X, Y, Z, cmap=cm.gist_heat_r, linewidth=0, antialiased=False)

ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

ax.set_zlim(0, 300)
fig.colorbar(ros, shrink=0.5, aspect=10)
plt.show()


## -----------------------------  Gradiente -------------------------------- ##

df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x, \
                         2*b*(y-x**2)])
    
F = lambda X: f(X[0],X[1])
dF = lambda X: df(X[0],X[1])

x_0 = np.array([-1.4,1.1])
print(F(x_0))
print(dF(x_0))

plt.figure(figsize=(12, 7))
plt.contour(X,Y,Z,200)
plt.plot([x_0[0]],[x_0[1]],marker='o',markersize=15, color ='r')


### --------------------- Encontrar la direccion de descenso --------- ##

fx = F(x_0);
gx = dF(x_0);
s = -gx;
print(s)

plt.figure(figsize=(12, 7))
plt.contour(X,Y,Z,200)
ns = np.sqrt(s[0]**2+s[1]**2);
plt.plot([x_0[0]],[x_0[1]],marker='o',markersize=15, color ='r')
plt.arrow(x_0[0],x_0[1],s[0]/ns,s[1]/ns, head_width=0.2, head_length=0.1, fc='r', ec='r')

## ---------------------- ¿Hasta donde llega? ------------------ ##

al = np.linspace(0,0.1,101)
z = [F(x_0+a*s) for a in al]
figLS = plt.figure(figsize=(12, 7))
plt.plot(al,z)
plt.ylabel('$f(x_0+ \\alpha s)$')
plt.xlabel('$\\alpha$')
plt.show()

figLS = plt.figure(figsize=(12, 7))
plt.plot(al,z)
plt.yscale('log')
plt.ylabel('$f(x_0+ \\alpha s)$')
plt.xlabel('$\\alpha$')
plt.show()


theta = 0.1
alpha = 1
tol = 1e-10
d = theta*np.dot(gx,s)

print([fx,fx+0.01*d])

figLS1 = plt.figure(figsize=(12, 7))
plt.plot(al,z)
plt.plot(al,[fx+a*d for a in al])

for i in range(10):
    
    if (alpha<=0.1):
        plt.plot(alpha,F(x_0+alpha*s),marker='x');
        plt.plot(alpha,fx + alpha*d,marker='o')
        
    if F(x_0+alpha*s) < (fx + alpha*d):
        break;
    alpha = alpha/2;
    
plt.yscale('log')
plt.ylabel('$f(x_0+ \\alpha s)$')
plt.xlabel('$\\alpha$')
plt.show()

print ("Alpha \n", alpha)
