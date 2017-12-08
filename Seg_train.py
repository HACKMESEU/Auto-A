import imgTextureAnalysis as iTA
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp
from sklearn.externals import joblib

nii_T1 = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820t1tirmtradarkfluids003a1001.nii.gz')
nii_ADC = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820ep2ddiff3scantracep2s008a1001.nii.gz')
nii_T2 = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820t2tirmtradarkfluids005a1001.nii.gz')

ROI = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\髓母卞张意20150907083820t1tirmtradarkfluids010a1001.nii')
dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\seg\\星形'
OriginData = pp.loadData(dirpath)
#pp.showROI(OriginData,0,5)
PatientNum = OriginData.shape[0]
ModelNum = OriginData.shape[1]
SliceNum = OriginData.shape[4]
Rows = OriginData.shape[2]
Columns = OriginData.shape[3]
X_data = np.zeros((Rows * Columns * PatientNum * 4, 3))
y_data = np.zeros((Rows * Columns* PatientNum * 4))
for index in range(PatientNum):
    count = 0
    for s in range(SliceNum):
        ROI = OriginData[index, 3, :, :, s]
        if(ROI.max()):
            if(count<4):
                data = OriginData[index, 0, :, :, s]
                para = np.asanyarray(iTA.imgTA(data))
                y = np.zeros((Rows * Columns))
                X = np.zeros((Rows * Columns, len(para)))
                for i in range(Rows):
                    for j in range(Columns):
                        X[i * Columns + j] = para[:, i, j]
                        if (ROI[i, j]):
                            y[i * Columns + j] = 1

                X_data[((index*4+count)*Rows*Columns):((index*4+count+1)*Rows*Columns),:] = X
                y_data[((index * 4 + count) * Rows * Columns):((index * 4 + count + 1) * Rows * Columns)] = y
                count = count + 1


#data = OriginData[2,0,:,:,4]
#ROI = OriginData[2,3,:,:,4]
#Rows = data.shape[0]
#Columns = data.shape[1]
#
#
#
#para = np.asanyarray(iTA.imgTA(data))
##pp.showImage(para[0])
##pp.showImage(para[1])
##pp.showImage(para[2])
##pp.showImage(para[3])
##pp.showImage(para[4])
#
#
#y = np.zeros((Rows*Columns))
#X = np.zeros((Rows*Columns,len(para)))
#for i in range(Rows):
#    for j in range(Columns):
#        X[i*Columns+j] = para[:,i,j]
#        if(ROI[i,j]):
#            y[i*Columns+j] = 1

#np.save('suimuX_train.npy',X_data)
#np.save('suimuy_train.npy',y_data)

from sklearn import svm
C =1.0
models = (svm.SVC(kernel='rbf', C=C))
#models = (clf.fit(X, y) for clf in models)
clf = svm.SVC(kernel='rbf', C=16)
#clf = RandomForestClassifier()
clf.fit(X, y)
y_predict = clf.predict(X)

joblib.dump(clf, "SVC_rbf_model.m")
#joblib.dump(clf, "RF_model.m")
ROI_predict = np.asanyarray(y_predict).reshape((Rows,Columns))
plt.figure('show')
plt.subplot(2, 1, 1)
plt.imshow(ROI, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(ROI_predict, cmap='gray')
plt.show()