import imgTextureAnalysis as iTA
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp
from sklearn.externals import joblib

dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\seg\\星形'
OriginData = pp.loadData(dirpath)
#pp.showROI(OriginData,0,5)
PatientNum = OriginData.shape[0]
ModelNum = OriginData.shape[1]
SliceNum = OriginData.shape[4]
Rows = OriginData.shape[2]
Columns = OriginData.shape[3]

data = OriginData[0,0,:,:,6]
ROI = OriginData[0,3,:,:,6]
Rows = data.shape[0]
Columns = data.shape[1]

para = np.asanyarray(iTA.imgTA(data))
X = np.zeros((Rows*Columns,len(para)))
for i in range(Rows):
    for j in range(Columns):
        X[i*Columns+j] = para[:,i,j]
mask =np.equal(X,-np.inf)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if mask[i,j]:
            X[i,j] = 0
clf = joblib.load('SVC_rbf_model.m')
y_predict = clf.predict(X)

ROI_predict = np.asanyarray(y_predict).reshape((Rows,Columns))
plt.figure('show')
plt.subplot(2, 1, 1)
plt.imshow(ROI, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(ROI_predict, cmap='gray')
plt.show()
#np.save('ROI_predict.npy',ROI_predict)
#ROI = ROI_predict
from scipy.ndimage.morphology import binary_erosion,binary_dilation,binary_fill_holes
import cv2
ROI_predict = binary_erosion(ROI_predict)
ROI_predict = binary_dilation(ROI_predict)
ROI_predict = binary_fill_holes(ROI_predict)
new_img = np.zeros_like(ROI_predict)                                        # step 1
for val in np.unique(ROI_predict)[1:]:                                      # step 2
    mask = np.uint8(ROI_predict == val)                                     # step 3
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      # step 5
    new_img[labels == largest_label] = val                          # step 6
ROI_predict = new_img


plt.figure('show')
plt.subplot(2, 1, 1)
plt.imshow(ROI, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(ROI_predict, cmap='gray')
plt.show()
count = 0
count1 = 0
count2 = 0
for i in range(ROI.shape[0]):
    for j in range(ROI.shape[1]):
        if((bool(ROI[i,j])) & (ROI_predict[i,j])):
            count = count +1
        if(ROI[i,j]):
            count1 = count1 +1
        if (ROI_predict[i, j]):
            count2 = count2 + 1
if((count1+count2)):
    Dice_Score = 2*count/(count1+count2)
    print(Dice_Score)