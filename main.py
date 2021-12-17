import random

import numpy as np
from scipy.interpolate import griddata
from mlpaceptron import MLP
import loader as dl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot2d():
    ldr = dl.loader()

    mlp = MLP(ldr)
    e_tr, e_ts = mlp.learn(epoches = 10000, eta = 0.025, epsilon=25e-4)
    # Номер ошибки на графике
    e_tr_x = [i for i in range(1, len(e_tr) + 1)]
    e_ts_x = [i for i in range(1, len(e_ts) + 1)]
    tri = ldr.getTrainInp()
    tro = ldr.getTrainOut()
    tsi = ldr.getTestInp()
    tso = ldr.getTestOut()
    f1 = plt.figure(1)
    fa1 = f1.add_subplot(1,1,1)
    out = mlp.calc(tri)
    # На тренировочном множестве
    fa1.plot(tri, out, "bo")
    fa1.plot(tri, tro , "r+")
    out = mlp.calc(tsi)
    # На тестовом множестве
    fa1.plot(tsi, out, "gv")
    fa1.plot(tsi, tso, "y+")
    f2 = plt.figure(2)
    fa2 = f2.add_subplot(2,1,1)
    fa2.plot(e_tr_x, e_tr, "r-")
    fa3 = f2.add_subplot(2,1,2)
    fa3.plot(e_ts_x, e_ts, "b-")
    plt.show()

#sss
def plot3D():
    ldr = dl.loader(3)
    mlp = MLP(ldr)
    e_tr, e_ts = mlp.learn(epoches=100, eta=0.005, epsilon=0.00001)
    tri = ldr.getTrainInp()
    tro = ldr.getTrainOut()
    tsi = ldr.getTestInp()
    tso = ldr.getTestOut()
    # На тренировочном множестве
    out = mlp.calc(tri)
    x = np.array([tr_inp[0] for tr_inp in tri])
    y = np.array([tr_inp[1] for tr_inp in tri])
    xData = sorted(set(x))
    yData = sorted(set(y))[::-1]
    X, Y = np.meshgrid(xData, yData)
    zTr = np.array([tr_out for tr_out in tro])
    Z = fillArray2D(zTr, xData, yData, tri)
    Out = fillArray2D(np.reshape(out, (len(out),1)), xData, yData, tri)
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1, projection='3d')
    ax2 = fig.add_subplot(2,1,2, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax.plot_surface(X, Y, Z, cmap='inferno')
    ax2.plot_surface(X, Y, Out, cmap='Greys')
    plt.show()

def fillArray2D(arrTr, xData, yData, tri):
    #
    Z = np.zeros((len(xData), len(yData)))
    k = 0
    for i in range(len(yData)):
        for j in range(len(xData)):
            if ([xData[i], yData[j]] not in tri):
                Z[i][j] = Z[i][j - 1]
            else:
                Z[i][j] = arrTr[k][0]
                k += 1
    return Z

if __name__ == '__main__':
    #plot3D()
    plot2d()