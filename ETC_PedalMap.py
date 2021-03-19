import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm, lines, collections, colors
import matplotlib.patches as patches
from scipy import interpolate, optimize
from scipy.interpolate import interp1d
import csv
import pandas as pd

TP_20 = np.array([10.06666667,11.06666667,9.833333333,12.16666667,11.73333333,11.76666667,8.6,5.633333333,4.6,4.3,4.166666667,2.833333333,1.666666667,0.4,0,0,0,0,0,0,0,0])
TP_30 = np.array([13.03333333,15.76666667,14.33333333,18.96666667,21.43333333,20.76666667,18.06666667,16.46666667,16.73333333,17.53333333,16.96666667,15.83333333,14.5,13.43333333,12.56666667,11.33333333,10.1,8.233333333,7.033333333,6.166666667,4.666666667,3.633333333])
TP_40 = np.array([14.16666667,16.1,15,20.9,25.06666667,25.1,22.56666667,21.16666667,23.26666667,25.46666667,25.56666667,23.83333333,23,23.23333333,23.43333333,22.9,21.56666667,20.06666667,18.43333333,17.23333333,15.93333333,13.35333333])
TP_50 = np.array([13.76666667,17.1,15.76666667,22.16666667,26.63333333,27,25.43333333,23.86666667,27.1,29.46666667,30.16666667,28.26666667,27.73333333,29.23333333,31.43333333,32.16666667,30.9,29.36666667,27.76666667,26.76666667,24.93333333,21.78])
TP_60 = np.array([14.36666667,18.53333333,16.83333333,23.3,27.93333333,28.5,27.06666667,25.5,29.23333333,31.7,32.86666667,31.16666667,30.8,33.3,36.06666667,38.36666667,37.06666667,35.66666667,34.13333333,32.9,30.86666667,27.91666667])
TP_70 = np.array([14.1,17.63333333,16.13333333,22.7,27.46666667,28.1,26.76666667,25.36666667,29.16666667,31.73333333,33.03333333,31.23333333,31,33.7,36.86666667,39.83333333,38.8,37.56666667,36.06666667,34.7,32.5,29.44333333])
TP_80 = np.array([14.2,17.4,16.4,22.73333333,27.33333333,27.93333333,26.63333333,25.23333333,29.03333333,31.5,32.8,31.1,30.86666667,33.6,36.73333333,39.93333333,38.96666667,37.86666667,36.26666667,34.96666667,32.6,29.60666667])
TP_90 = np.array([14.8,17.86666667,16.86666667,23.36666667,28.06666667,28.66666667,27.2,25.86666667,29.8,32.36666667,33.6,31.93333333,31.93333333,34.73333333,38.13333333,41.33333333,40.03333333,38.96666667,37.5,35.6,33.26666667,30.35666667])
TP_100 = np.array([14.13333333,19.36666667,15.93333333,22.7,28.13333333,28.83333333,27.4,25.83333333,29.86666667,32.66666667,33.83333333,32.2,32.13333333,34.9,38.1,41.13333333,40,39.03333333,37.5,35.63333333,33.43333333,30.02])


delta = 0.01 # spacing between samples in each dimension
RPM = np.array([2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500])
TP = np.array([20,30,40,50,60,70,80,90,100])
X, Y = np.meshgrid(RPM, TP)
Z = np.array([TP_20, TP_30, TP_40, TP_50, TP_60, TP_70, TP_80, TP_90, TP_100])


# Plot the functions
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Surface plot
surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(surf)

contours2 = ax2.contour(X, Y, Z, 30)

fig2 = plt.figure(figsize=(6,6))

TPa = TP[:-1]
TPa = np.array([20, 30, 40, 50, 60, 70, 80, 90])
APPSy = np.array([0, 4, 5, 6, 8, 11, 15, 25, 45, 60, 70, 80, 90])
APPS_Map = np.zeros((len(APPSy), len(RPM)))
for i in range(len(RPM)):
    data = Z[:-1,i]
    # Condition data
    for k in range(len(data)-1):
        if data[k+1] < data[k]:
            data[k+1] = data[k]*1.00001
    maxtrq = np.max(data)
    normtrq = data/maxtrq
    interp_fi = interp1d(TPa, normtrq, fill_value="extrapolate") # interpolating function
    plt.plot(TPa,normtrq)
    for j in range(len(APPSy)):
        interp_cost = lambda x: interp_fi(x) - APPSy[j]/100 # cost function
        initguess = 50
        minimizer = optimize.newton(interp_cost, initguess)
        if minimizer > 100:
            minimizer = 101
        elif minimizer < 0:
            minimizer = 0
        APPS_Map[j,i] = minimizer

# Surface plot
plt.close("all")
fig3 = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
Xa, Ya = np.meshgrid(RPM, APPSy)
surf = ax.plot_surface(Xa, Ya, APPS_Map, cmap='viridis')
ax.set_title('Linear Torque Pedal Map')
ax.set_xlabel('RPM')
ax.set_ylabel('APPS')
ax.set_zlabel('ATH')
ax.set(zlim = [0,70])

APPS_Map_Full = np.concatenate((APPSy[:,None],APPS_Map),axis=1)
DF = pd.DataFrame(APPS_Map_Full)
DF.columns = np.concatenate((np.array(['APPS\RPM']),RPM))
DF.to_csv("PedalMap.csv")

plt.show()
