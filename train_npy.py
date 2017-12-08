import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
X = np.load('suimuX_train.npy')
y = np.load('suimuy_train.npy')





Rows = X.shape[0]
Columns = X.shape[1]
mask =np.equal(X,-np.inf)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if mask[i,j]:
            X[i,j] = 0
#X = X.reshape((Rows,Columns))
from sklearn import svm
C =1.0
models = (svm.SVC(kernel='rbf', C=C))
#models = (clf.fit(X, y) for clf in models)
clf = svm.SVC(kernel='rbf', C=16)
#clf = RandomForestClassifier()
clf.fit(X, y)
y_predict = clf.predict(X)

joblib.dump(clf, "16SVC_rbf_model.m")
#joblib.dump(clf, "RF_model.m")
ROI_predict = np.asanyarray(y_predict).reshape((232,256))


plt.figure('show')
plt.subplot(2, 1, 1)
#plt.imshow(ROI, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(ROI_predict, cmap='gray')
plt.show()