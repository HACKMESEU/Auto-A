import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

def loadExcel(str):
    import pandas as pd
    p = pd.read_excel(str).values
    return p

def showImage(img):
    import matplotlib.pyplot as plt
    plt.figure('test')
    plt.imshow(img,cmap='gray')
    #plt.plot((1,2,100),(100,100,100))
    plt.show()

def MappToInt(img):
    RangeSize = 4095
    if (img.max()):
        if (float(img.max() - img.min())):
            rslope = RangeSize / float(img.max() - img.min())
        else:
            rslope = 0.0
        rinter = rslope * img.min()
        img = img * rslope - rinter
    return np.uint16(img)

def cropImage(img):
    rowIndex = []
    columnIndex = []
    for i in range(img.shape[0]):
        if(img[i,:].max()):
            rowIndex.append(i)
    for i in range(img.shape[1]):
        if(img[:,i].max()):
            columnIndex.append(i)
    img2 = img
    try:
        img2 = img[min(rowIndex):(max(rowIndex)+1),min(columnIndex):(max(columnIndex)+1)]
    except:
        pass
    return img2

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

        #print('\t' * count + '├── ' + fileName)
    elif os.path.isdir(currentPath):
        #print('\t' * count + '├── ' + currentPath)
        pathList = os.listdir(currentPath)
        for eachPath in pathList:
            travelTree(currentPath + '/' + eachPath, count + 1,filelist)
    return filelist

def loadNii(fileName):
    import nibabel
    nii = nibabel.load(fileName)
    data = np.asanyarray(nii._data)
    return data

def ListFileName(dirName):
    filelist = travelTree(dirName,1,[])
    return filelist

def CropImage(img,reshape):
    try:
        shape = img.shape
    except:
        shape = reshape
        print('not a numpy class')
    offsetX = 0
    offsetY = 0
    if(shape[0]>reshape[0]):
        offsetX = int((shape[0]-reshape[0])/2)
    if (shape[1] > reshape[1]):
        offsetY = int((shape[1] - reshape[1]) / 2)
    img = img[offsetX:(shape[0]-offsetX),offsetY:(shape[1]-offsetY)]
    return img


def loadData(dirpath):
    #dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\星形'
    files = ListFileName(dirpath)

    PaitentName = []
    for file in files:
        if (os.path.dirname(file) not in PaitentName):
            PaitentName.append(os.path.dirname(file))

    ModalNum = 4
    PaitentNum = len(PaitentName)
    print('-------------------------')
    print('Patient counts:',len(PaitentName))
    for i in PaitentName:
        print(i)
    print('-------------------------')
    print('MR modal counts:', ModalNum)
    ArrangeName = files
    for file in files:
        if (os.path.dirname(file) in PaitentName):
            index = PaitentName.index(os.path.dirname(file))
            if ((file.find('diff') != -1) & (file.find('nii.gz') != -1)):
                Data_diffFile = file
                ArrangeName[index * ModalNum + 0] = Data_diffFile
            if ((file.find('t1') != -1) & (file.find('nii.gz') != -1)):
                Data_T1File = file
                ArrangeName[index * ModalNum + 1] = Data_T1File
            if ((file.find('t2') != -1) & (file.find('nii.gz') != -1)):
                Data_T2File = file
                ArrangeName[index * ModalNum + 2] = Data_T2File
            if ((file.find('nii') != -1) & (file.find('nii.gz') == -1)):
                Data_ROIFile = file
                ArrangeName[index * ModalNum + 3] = Data_ROIFile

    data = np.zeros((232,256,22))
    try:
        data = loadNii(ArrangeName[1])
    except:
        print('load data failed')
    OriginData = np.zeros((len(PaitentName),ModalNum,data.shape[0],data.shape[1],data.shape[2]))
    for index in range(PaitentNum):
        files = ArrangeName[index * ModalNum:(index+1)*ModalNum]
        for file in files:
            #file = os.path.join(file)
            if ((file.find('diff') != -1)&(file.find('nii.gz') != -1)):
                Data_diffFile = file
                try:
                    Data_diff = loadNii(Data_diffFile)
                except:
                    print('load nii failed!')
            if ((file.find('t1') != -1)&(file.find('nii.gz') != -1)):
                Data_T1File = file
                try:
                    Data_T1 = loadNii(Data_T1File)
                except:
                    print('load nii failed!')
            if ((file.find('t2') != -1)&(file.find('nii.gz') != -1)):
                Data_T2File = file
                try:
                    Data_T2 = loadNii(Data_T2File)
                except:
                    print('load nii failed!')
            if ((file.find('nii') != -1)&(file.find('nii.gz') == -1)):
                Data_ROIFile = file
                try:
                    Data_ROI = loadNii(Data_ROIFile)
                except:
                    print('load nii failed!')

        # crop the ROI image , resize the T2/ROI map according to T1 map
        from skimage import transform
        CropImg = np.zeros(Data_T1.shape)
        ResizeImgT2 = np.zeros(Data_T1.shape)
        ResizeImgROI = np.zeros(Data_T1.shape)
        for s in range(Data_T1.shape[2]):
            CropImg[:,:,s] = CropImage(Data_diff[:, :, s], (Data_T1.shape[0], Data_T1.shape[1]))
            ResizeImgT2[:, :, s] = transform.resize(Data_T2[:, :, s], (Data_T1.shape[0], Data_T1.shape[1]),preserve_range=True,mode='constant')
            if(Data_T1.shape != Data_ROI.shape):
                if (Data_diff.shape == Data_ROI.shape):
                    ResizeImgROI[:,:,s] = CropImage(Data_ROI[:, :, s], (Data_T1.shape[0], Data_T1.shape[1]))
                if (Data_T2.shape == Data_ROI.shape):
                    ResizeImgROI[:, :, s] = transform.resize(Data_ROI[:, :, s], (Data_T1.shape[0], Data_T1.shape[1]),preserve_range=True,mode='constant')
            else:
                ResizeImgROI[:, :, s] = Data_ROI[:, :, s]
                #plt.figure('1')
                #plt.imshow(ResizeImgROI[:, :, s], cmap='gray')
                #plt.show()

        Data_diff = CropImg
        Data_T2 = ResizeImgT2
        Data_ROI = ResizeImgROI
        OriginData[index, 0, :, :, :] = Data_diff[:,:,0:(data.shape[2])]
        OriginData[index, 1, :, :, :] = Data_T1[:,:,:data.shape[2]]
        OriginData[index, 2, :, :, :] = Data_T2[:,:,:data.shape[2]]
        OriginData[index, 3, :, :, :] = Data_ROI[:,:,:data.shape[2]]
    return OriginData

def showROI(OriginData,index,slice):
    s = slice
    Data_diff = OriginData[index,0,:,:,:]
    Data_T1 = OriginData[index, 1, :, :, :]
    Data_T2 = OriginData[index, 2, :, :, :]
    Data_ROI = OriginData[index, 3, :, :, :]
    # show & Check the ROI
    plt.figure('show')

    plt.subplot(2,2,1)
    plt.imshow(Data_diff[:,:,s],cmap='gray')
    contours  = measure.find_contours(Data_ROI[:,:,s],0)
    for n, contour in enumerate(contours):
       plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    plt.title("ADC")
    plt.subplot(2,2,2)
    plt.imshow(Data_T1[:,:,s],cmap='gray')
    contours  = measure.find_contours(Data_ROI[:,:,s],0)
    for n, contour in enumerate(contours):
       plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    plt.title("T1 map")
    plt.subplot(2,2,3)
    plt.imshow(Data_T2[:,:,s],cmap='gray')
    contours  = measure.find_contours(Data_ROI[:,:,s],0)
    for n, contour in enumerate(contours):
       plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    plt.title("T2 map")
    plt.show()


#example
#dirpath = 'C:\\Users\\Sun Weihang\\Desktop\\data1121\\室管膜瘤'
#OriginData = loadData(dirpath)
#showROI(OriginData,0,6)

    ## show & Check the ROI
    #plt.figure('show')
    #s = 5
    #plt.subplot(2,2,1)
    #plt.imshow(Data_diff[:,:,s],cmap='gray')
    #contours  = measure.find_contours(Data_ROI[:,:,s],0)
    #for n, contour in enumerate(contours):
    #    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    #plt.title("ADC")
    #
    #plt.subplot(2,2,2)
    #plt.imshow(Data_T1[:,:,s],cmap='gray')
    #contours  = measure.find_contours(Data_ROI[:,:,s],0)
    #for n, contour in enumerate(contours):
    #    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    #plt.title("T1 map")
    #
    #plt.subplot(2,2,3)
    #plt.imshow(Data_T2[:,:,s],cmap='gray')
    #contours  = measure.find_contours(Data_ROI[:,:,s],0)
    #for n, contour in enumerate(contours):
    #    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='cyan')
    #plt.title("T2 map")
    #plt.show()

#OriginData = np.zeros((len(PaitentName),ModalNum,data.shape[0],data.shape[1]))










#plt.subplot(2,2,4)
#plt.imshow(Data_ROI[:,:,s],cmap='gray')

#showImage(Data_ROI[:,:,5])
#showImage(Data_T1[:,:,5])
#showImage(Data_T2[:,:,5])
#showImage(Data_diff[:,:,5])



#dataROI = loadNii('C:\\Users\\Sun Weihang\\Desktop\\roi.nii\\20151030_081240t1tirmtradarkfluids003a1001.nii')
#data = loadNii('C:\\Users\\Sun Weihang\\Desktop\\20151030_081240t1tirmtradarkfluids003a1001.nii\\20151030_081240t1tirmtradarkfluids003a1001.nii')

#showImage(dataROI[:,:,6])
#dataROI = loadNii('C:\\Users\\Sun Weihang\\Desktop\\新建文件夹 (2) - 副本\\髓母\\卞张意\\TJHC0006S41063\\ROI.nii.gz')
#data = loadNii('C:\\Users\\Sun Weihang\\Desktop\\新建文件夹 (2) - 副本\\髓母\\卞张意\\TJHC0006S41063\\20150907_083820t1tirmtradarkfluids010a1001.nii.gz')
#
#showImage(data[:,:,4])
#showImage(dataROI[:,:,4])


#filelist = travelTree('C:\\Users\\Sun Weihang\\Desktop\\1106\\1106\\1106', 1,[])
#
#import os
#
#for i in range(len(filelist)):
#    #i = 1445
#    dirName = os.path.dirname(filelist[i])
#    baseName = os.path.basename(filelist[i])
#    try:
#        img = loadExcel(filelist[i])
#    except:
#        print('load failed')
#    import FeatureAnalysis as FA
#    Skewness,Kurtosis,Entropy,Mode_count,Mode_Value,Variance,Percentile_10,Percentile_25,Percentile_50,Percentile_75,Percentile_90 = FA.HistogramAnalysis(img)
#    img = MappToInt(img)
#    #showImage(img)
#    img2 = cropImage(img)
#    #showImage(img2)
#    contrast,dissimilarity,homogeneity,ASM,energy,correlation = FA.TextureAnalysis(img2)
#    paralist = [Skewness,Kurtosis,Entropy,Mode_count,Mode_Value,Variance,Percentile_10,Percentile_25,Percentile_50,
#                Percentile_75,Percentile_90,contrast, dissimilarity, homogeneity, ASM, energy, correlation]
#    paralist = np.float64(np.asanyarray(paralist))
#    np.save(os.path.join(dirName,baseName+'.npy'),paralist)
#    print(str(i),'//',str(len(filelist)))
#    #para = np.load(os.path.join(dirName,baseName+'.npy'))
#    #1+1


