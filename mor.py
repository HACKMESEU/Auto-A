import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.morphology import binary_erosion,binary_dilation,binary_fill_holes
ROI_predict = np.load('ROI_predict.npy')
ROI = ROI_predict
ROI_predict = binary_erosion(ROI_predict)
ROI_predict = binary_dilation(ROI_predict)
ROI_predict = binary_fill_holes(ROI_predict)
new_img = np.zeros_like(ROI_predict)                                        # step 1
for val in np.unique(ROI_predict)[1:]:                                      # step 2
    mask = np.uint8(ROI_predict == val)                                     # step 3
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      # step 5
    new_img[labels == largest_label] = val                          # step 6

plt.figure('show')
plt.subplot(2, 1, 1)
plt.imshow(ROI, cmap='gray')
plt.subplot(2, 1, 2)
plt.imshow(new_img, cmap='gray')
plt.show()
