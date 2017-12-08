from skimage.feature import greycomatrix, greycoprops
import numpy as np
from skimage.feature import greycomatrix, greycoprops


def mapTo16Level(img):
    if(img.max()):
        img = np.uint(img / (img.max() / 15))
    return img




def imgTA(data):
    #pp.showImage(data)
    kernelsize = 9
    Rows = data.shape[0]
    Columns = data.shape[1]
    # 2 order
    margin = int(kernelsize/2)
    contrast = np.zeros(data.shape)
    energy = np.zeros(data.shape)
    correlation = np.zeros(data.shape)
    dissimilarity = np.zeros(data.shape)

    # histogram
    Skewness = np.zeros(data.shape)
    Kurtosis = np.zeros(data.shape)
    Entropy = np.zeros(data.shape)
    Mode = np.zeros(data.shape)
    Mode_Value = np.zeros(data.shape)
    Variance = np.zeros(data.shape)

    data16 = mapTo16Level(data)
    for i in range(Rows):
        print(i,Rows)
        for j in range(Columns):
            if((i>=margin)&(i<=Rows-1-margin)&(j>margin)&(j<=Columns-1-margin)):
                temp = data16[i-1:i+2,j-1:j+2]
                if(not np.all(np.equal(temp,0))):
                    g = greycomatrix(temp, [1], [ np.pi/4],levels=16)
                    contrast[i,j] = greycoprops(g, 'contrast')
                    energy[i, j] = greycoprops(g, 'energy')
                    correlation[i, j] = greycoprops(g, 'correlation')
                    dissimilarity[i, j] = greycoprops(g, 'dissimilarity')
                    #print(contrast[i,j])
                # reshape the image into an one-dim array
                temp = data[i - 1:i + 2, j - 1:j + 2]
                temp = temp.reshape((temp.shape[0] * temp.shape[1], 1))
                Skewness[i,j] = skew(temp)
                Kurtosis[i,j] = kurtosis(temp)
                Entropy[i,j] = entropy(temp)
                Mode = mode(temp)
                #Mode_count = Mode.count
                Mode_Value[i,j] = np.float(Mode.mode)
                Variance[i,j] = np.var(temp)
    paralist = Mode_Value,Entropy,Skewness
    return paralist

#nii_T1 = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820t1tirmtradarkfluids003a1001.nii.gz')
#nii_ADC = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820ep2ddiff3scantracep2s008a1001.nii.gz')
#nii_T2 = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820t2tirmtradarkfluids005a1001.nii.gz')
#
#ROI = pp.loadNii('C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母\\卞张意\\TJHC0006S41063\\髓母卞张意20150907083820t1tirmtradarkfluids010a1001.nii')
#dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\髓母'
#OriginData = pp.loadData(dirpath)
##pp.showROI(OriginData,0,5)
#PatientNum = OriginData.shape[0]
#ModelNum = OriginData.shape[1]
#SliceNum = OriginData.shape[4]
#Rows = OriginData.shape[2]
#Columns = OriginData.shape[3]
#
#
#
#data = OriginData[0,0,:,:,6]
#ROI = OriginData[0,3,:,:,6]
#Rows = data.shape[0]
#Columns = data.shape[1]
#
#
#
#para = np.asanyarray(imgTA(data))
##pp.showImage(para[0])
#y = np.zeros((Rows*Columns))
#X = np.zeros((Rows*Columns,len(para)))
#for i in range(Rows):
#    for j in range(Columns):
#        X[i*Columns+j] = para[:,i,j]
#        if(ROI[i,j]):
#            y[i*Columns+j] = 1
#from sklearn import svm
#C =1.0
#models = (svm.SVC(kernel='rbf', C=C))
##models = (clf.fit(X, y) for clf in models)
#clf = svm.SVC(kernel='rbf', C=C)
##clf = RandomForestClassifier()
#clf.fit(X, y)
#y_predict = clf.predict(X)
#
#joblib.dump(clf, "SVC_rbf_model.m")
#
#ROI_predict = np.asanyarray(y_predict).reshape((Rows,Columns))
#plt.figure('show')
#plt.subplot(2, 1, 1)
#plt.imshow(ROI, cmap='gray')
#plt.subplot(2, 1, 2)
#plt.imshow(ROI_predict, cmap='gray')
#plt.show()

#pp.showImage(Entropy)
#pp.showImage(Kurtosis)
#pp.showImage(Mode_Value)
#pp.showImage(Variance)
