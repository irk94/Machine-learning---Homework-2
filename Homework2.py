import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import cm

# size of data
N = 50

# define the function F used in matrix M
def F(x,y):
    return np.exp(y*1j) + 2*np.exp(-y*1j/2)*np.cos(np.sqrt(3)/2*x)
F = np.vectorize(F)

# generate a (x,y) grid and turn it into 1D vectors
values = np.linspace(-np.pi,np.pi,N)

x,y = np.meshgrid(values,values)
X = x.flatten()
Y = y.flatten()

# define the diagonaliazing function
# returns the eigen values in two matrixes corresponding to the (x,y) grid
def diagonalize(x,y):
    M = np.array([[0,F(x,y)],[np.conjugate(F(x,y)),0]],dtype=complex)
    eigenvalues = tuple(np.linalg.eigvals(M))
    return eigenvalues
diagonalize = np.vectorize(diagonalize)
EV1,EV2 = diagonalize(X,Y)

EV1 = np.reshape(np.real(EV1),np.shape(x))
EV2 = np.reshape(np.real(EV2),np.shape(x))

# 3D plot of the eigen values
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
 
surf1 = ax.plot_surface(x, y, EV1, cmap='viridis', alpha=0.8)
surf2 = ax.plot_surface(x, y, EV2, cmap='viridis_r', alpha=0.8)
 
ax.set_title('Eigen values of the matrix M', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('Eigen values', fontsize=12)

cbar1 = fig.colorbar(surf1, shrink=0.5, aspect=5)
cbar2 = fig.colorbar(surf2, shrink=0.5, aspect=5)
cbar1.ax.get_yaxis().labelpad = 15
cbar1.ax.set_ylabel('Positive eigen values', rotation=270)
cbar2.ax.get_yaxis().labelpad = 15
cbar2.ax.set_ylabel('Negative eigen values', rotation=270)

plt.show()
