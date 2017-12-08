# Histogram Analysis
# para: Mean variance,mode,maximum probability
#skewness,kurtosis,energy,entropy;
#percentiles:10%,25%,50%,75%,and 90%.


#from pydicom import dicomio
#dcm = dicomio.read_file('C:\\Users\\Sun Weihang\\Desktop\\Histogram\\1.dcm')


#import numpy as np

#img= np.asanyarray(dcm.pixel_array)



def HistogramAnalysis(img):
    import numpy as np
    from scipy.stats import kurtosis, skew, entropy, mode
    # reshape the image into an one-dim array
    img = img.reshape((img.shape[0]*img.shape[1],1))
    # delete the zero value
    mask = np.all(np.equal(img,0),axis=1)
    img = img[~mask]

    # Calculate the histogram paras
    Skewness = skew(img)
    Kurtosis = kurtosis(img)
    Entropy = entropy(img)
    Mode = mode(img)
    Mode_count = Mode.count
    Mode_Value = Mode.mode
    Variance = np.var(img)
    Percentile_10 = np.percentile(img,10.0)
    Percentile_20 = np.percentile(img, 20.0)
    Percentile_30 = np.percentile(img, 30.0)
    Percentile_40 = np.percentile(img, 40.0)
    Percentile_50 = np.percentile(img, 50.0)
    Percentile_60 = np.percentile(img, 60.0)
    Percentile_70 = np.percentile(img, 70.0)
    Percentile_80 = np.percentile(img, 80.0)
    Percentile_90 = np.percentile(img, 90.0)

    Percentile_15 = np.percentile(img, 15.0)
    Percentile_25 = np.percentile(img, 25.0)
    Percentile_35 = np.percentile(img, 35.0)
    Percentile_45 = np.percentile(img, 45.0)
    Percentile_55 = np.percentile(img, 55.0)
    Percentile_65 = np.percentile(img, 65.0)
    Percentile_75 = np.percentile(img, 75.0)
    Percentile_85 = np.percentile(img, 85.0)
    Percentile_95 = np.percentile(img, 95.0)

    Percentile_10 = np.percentile(img, 10.0)
    Percentile_25 = np.percentile(img,25.0)
    Percentile_50 = np.percentile(img,50.0)
    Percentile_75 = np.percentile(img,75.0)
    Percentile_90 = np.percentile(img,90.0)
    paralist = Skewness,Kurtosis,Entropy,Mode_count,Mode_Value,Variance,Percentile_10,Percentile_25,Percentile_50,Percentile_75,Percentile_90
    #paralist = np.float64(np.asanyarray(paralist))

    #paralist = Skewness, Kurtosis, Entropy, Mode_count, Mode_Value, Variance, Percentile_10, Percentile_20, Percentile_30, Percentile_40, Percentile_50, Percentile_60, Percentile_70, Percentile_80, Percentile_90,Percentile_15, Percentile_25, Percentile_35, Percentile_45, Percentile_55, Percentile_65, Percentile_75, Percentile_85, Percentile_95
    return paralist

def TextureAnalysis(img):
    from skimage.feature import greycomatrix, greycoprops
    level = 4096
    if (img.dtype == 'uint16'):
        level = 4096
    if (img.dtype == 'uint8'):
        level = 256
    img = mapTo16Level(img)
    level = 16
    g = greycomatrix(img, [1], [0], levels=level)
    contrast = greycoprops(g, 'contrast')
    dissimilarity = greycoprops(g, 'dissimilarity')
    homogeneity = greycoprops(g, 'homogeneity')
    ASM = greycoprops(g, 'ASM')
    energy = greycoprops(g, 'energy')
    correlation = greycoprops(g, 'correlation')
    paralist = contrast,dissimilarity,homogeneity,ASM,energy,correlation
    #paralist = np.float64(np.asanyarray(paralist))
    return  paralist

#Skewness,Kurtosis,Entropy,Mode_count,Mode_Value,Variance,Percentile_10,Percentile_25,Percentile_50,Percentile_75,Percentile_90 = HistogramAnalysis(img)
#contrast,dissimilarity,homogeneity,ASM,energy,correlation = TextureAnalysis(img)
import numpy as np
def mapTo16Level(img):
    if(img.max()):
        img = np.uint(img / (img.max() / 15))
    return img

def test():
    1+1
    print('import FA OK!')

