

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import neighbors
from sklearn import gaussian_process

# Import dataset
data  = pd.read_csv('datasetA.csv', header=None)
data  = data.as_matrix()
L1,L2 = data.shape
L2    = L2-1
ref   = data[:,L2]

# Cut one period out of dataset
d1    = (ref==ref.max()).argmax()          # index 360
d2    = L1-(ref[::-1]==ref.min()).argmax() # index 0
ref   = data[d1:d2,L2]
D     = data[d1:d2,0:L2]
L1,L2 = D.shape

# Plot data
fig  = plt.figure()
ax   = fig.gca(projection='3d')
X,Y  = np.arange(0, L2, 1), np.arange(0, 360, 1)
X,Y  = np.meshgrid(X, Y)
Z    = D[0:L1:1000,:]
surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.hot,linewidth=0)
    
# K-NN regression model
X = D[0:L1:360,:]
y = ref[0:L1:360]
nn = 5 # number nearest neighbors
knn = neighbors.KNeighborsRegressor(nn, weights='uniform')

# Fit K-NN model
knn.fit(X, y);

# Two GP regression models: one in range 0-180, another in range 180-360
M = (ref==180).argmax()
target_1 = ref[0:M:180]
target_2 = ref[M::180]
trainD_1 = D[0:M:180,0:L2]
trainD_2 = D[M::180,0:L2]
gp1 = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp2 = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)

# Fit GP models
gp1.fit(trainD_1, target_1);
gp2.fit(trainD_2, target_2);

# Prediction on 3600 images
x = D[0:L1:100,0:L2]
y_true = ref[0:L1:100]
x_class = (knn.predict(x)<180)+0
x1 = x[x_class==0,:]
x2 = x[x_class==1,:]
y_true1 = y_true[x_class==0]
y_true2 = y_true[x_class==1]
y_pred1, s2_pred1 = gp1.predict(x1, eval_MSE=True)
y_pred2, s2_pred2 = gp2.predict(x2, eval_MSE=True)
y_true = np.concatenate([y_true1,y_true2])
y_pred = np.concatenate([y_pred1,y_pred2])

# Accuracy
acc = np.mean(np.sqrt((y_pred-y_true)**2))
print('Average accuracy (3600 test images):',acc)

# Plot predictions vs references
plt.figure()
plt.plot(y_true[0::],y_pred[0::],'r.')
plt.plot([0,360],[0,360],'k')
plt.ylabel('reference')
plt.xlabel('prediction')
axes = plt.gca()
axes.set_xlim([-20,360+20]);
axes.set_ylim([-20,360+20]);
