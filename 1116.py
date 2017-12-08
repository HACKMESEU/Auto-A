
import os
import time

import FeatureAnalysis as FA
import numpy as np
import pandas as pd


def ListFileName(dirName):
    filelist = travelTree(dirName,1,[])
    return filelist

def travelTree(currentPath, count,filelist):

    if not os.path.exists(currentPath):
        return

    if os.path.isfile(currentPath):
        fileName = os.path.basename(currentPath)

        if((currentPath.find('xlsx') == -1)&(currentPath.find('enhanced') == -1)):
            filelist.append(currentPath)
        #else:
        #    if(currentPath.find('npy') == -1):
        #        filelist.append(currentPath)


    elif os.path.isdir(currentPath):

        pathList = os.listdir(currentPath)
        for eachPath in pathList:
            travelTree(currentPath + '/' + eachPath, count + 1,filelist)
    return filelist

def loadNii(fileName):
    import nibabel
    nii = nibabel.load(fileName)
    data = np.asanyarray(nii._data)
    return data
while(1):
    raw_input_A = input("dirpath: ")
    files = ListFileName(raw_input_A)
    for f in files:
        if((f.find('roi') == -1)&(f.find('nii.gz') != -1)):
            niipath = f
        #if((f.find('roi') != -1)&(f.find('nii.gz') != -1)):
        if ((f.find('roi') != -1) ):
            roipath = f
    print('Nii path is:',niipath)
    print('ROI path is:',roipath)
    #niipath = 'C:\\Users\\Sun Weihang\\Desktop\\Histogram\\20160111_133614ep2ddiff3scantracep2s007a1001.nii.gz'
    #roipath = 'C:\\Users\\Sun Weihang\\Desktop\\Histogram\\20160111_133614ep2ddiff3scantracep2s007a1001roi.nii.gz'

    nii = loadNii(niipath)
    roi = loadNii(roipath)

    sliceNum = nii.shape[2]

    dirpath = os.path.dirname(niipath)

    print('Analyzing......')
    paraADC = []
    for s in range(sliceNum):
        ROI = roi[:,:,s]
        data= np.zeros((roi.shape[0],roi.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (ROI[i,j]):
                    data[i,j] = nii[i, j, s]
        if(ROI.max()):
            para = np.float64(FA.HistogramAnalysis(data))
            paraADC.append(para)

    para = np.asanyarray(paraADC).mean(axis=0)
    data_df = pd.DataFrame(para)
    writer = pd.ExcelWriter(os.path.join(dirpath,time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))+'para' + '.xlsx'))
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format
    writer.save()
    print('Finished!')


