import os
import time

import FeatureAnalysis as FA
import numpy as np
import preprocess as pp

dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\室管膜瘤'
OriginData = pp.loadData(dirpath)
#pp.showROI(OriginData,0,5)
PatientNum = OriginData.shape[0]
ModelNum = OriginData.shape[1]
SliceNum = OriginData.shape[4]
Rows = OriginData.shape[2]
Columns = OriginData.shape[3]
paraTotal = []
for index in range(PatientNum):
    print('processing......', index, '/', PatientNum)
    paraADC = []
    paraT1 = []
    paraT2 = []
    for s in range(SliceNum):
        T1ROI = np.zeros((Rows,Columns))
        T2ROI = np.zeros((Rows,Columns))
        ADCROI = np.zeros((Rows,Columns))
        ADC = OriginData[index,0,:,:,s]
        T1 = OriginData[index, 1, :, :, s]
        T2 = OriginData[index, 2, :, :, s]
        ROI = OriginData[index, 3, :, :, s]
        for i in range(Rows):
            for j in range(Columns):
                if(ROI[i,j]):
                    T1ROI[i,j] = T1[i,j]
                    T2ROI[i, j] = T2[i, j]
                    ADCROI[i, j] = ADC[i, j]
        if(not(np.all(np.equal(T1ROI,0)))):
            try:
                paralist = FA.HistogramAnalysis(T1ROI) + FA.TextureAnalysis(np.uint16(T1ROI))
                paralist = np.float64(np.asanyarray(paralist))
                paraT1.append(paralist)
                paralist = FA.HistogramAnalysis(T2ROI) + FA.TextureAnalysis(np.uint16(T2ROI))
                paralist = np.float64(np.asanyarray(paralist))
                paraT2.append(paralist)
                paralist = FA.HistogramAnalysis(ADCROI) + FA.TextureAnalysis(np.uint16(ADCROI))
                paralist = np.float64(np.asanyarray(paralist))
                paraADC.append(paralist)
            except:
                print('Analysis failed!')
    paraT1 = np.asanyarray(paraT1).mean(axis=0)
    paraT2 = np.asanyarray(paraT2).mean(axis=0)
    paraADC = np.asanyarray(paraADC).mean(axis=0)
    para = np.asanyarray(list(paraADC) + list(paraT1) + list(paraT2))
    paraTotal.append(para)


import pandas as pd
data_df = pd.DataFrame(paraTotal)
writer = pd.ExcelWriter(os.path.join(dirpath,time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))+'para' + '.xlsx'))
data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format
writer.save()