################################################################
#In this file, ti is the number you could select to choose which image you could like to run: (feel free to change in line148)
# 0 for the 1st image, 054112895010_20.tif
# 1 for the 2nd image, 054112895030_20.tif
# 2 for the 3rd image, 054330675010_20.tif
# 3 for all the images.
################################################################

# Import a library for raster management and the estimators used
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from osgeo import ogr
import os

# Importing code from our modules
from getPixelValues import getGeneralSinglePixelValues
from visualization import plot_tif, plotconfusionmatrix, visualizeTree
import utility as ut
import myFunction as mf


# set your local path of this python file
a = os.path.abspath('muitiparaRF.py')
path_py = a[:-15]

# define locations of tif and shp data
path_plg = path_py + r"\STARS_Sentinel_Mali\polygon"
path_tif = path_py + r"\STARS_Sentinel_Mali\images"
path_mask = path_py + r"\STARS_Sentinel_Mali\remask.tif"
path_out = path_py + r"\STARS_Sentinel_Mali\{0}.tif"

# define variables
name_samples_img = "CropsSamples"
name_classified_sgl = "Crops_RF_Classified_single"
name_classified_mul = "Crops_RF_Classified_multit"
name_train_test_img = "Train_Test"
name_source_img1 = "054112895010_20.tif"
name_source_img2 = "054112895030_20.tif"
name_source_img3 = "054330675010_20.tif"
name_label_col = "Crop_label"

# set current work path
os.chdir(path_plg)

# reprojection shp files
datasource = ogr.Open("mergeSHP_file.shp")
shape = ut.reproject_vector(datasource, epsg_from = 4326, epsg_to = 32630)

#test of the function

def testAcc(data1,lnam):
    mAcc = np.empty(12, dtype=np.float).reshape(4,3)

    for p in range(1, 5, 1):
        for n in range(1, 4, 1):
            # split into training and test data
            shuffled = ut.shuffle_data(data1[:, 2:], (100-p*20), where1)
            trainingSample1 = shuffled[0]
            trainingLabels1 = shuffled[1]
            trainingLabels1 = np.array([int(trainingLabels1[x][0]) for x in range(0, len(trainingLabels1))])
            testSamples1 = shuffled[2]
            testLabels1 = shuffled[3]
            testLabels1 = np.array([int(testLabels1[x][0]) for x in range(0, len(testLabels1))])
            k1 = shuffled[4]
            idx_shuffled1 = shuffled[5]

            # Initialize estimator
            rf1 = RandomForestClassifier(n_estimators=n*50)

            # Fitting the model to our training samples
            dtresult1 = rf1.fit(trainingSample1, trainingLabels1)

            # Predicting over test samples
            test_pred1 = rf1.predict(testSamples1)

            # Calculating the model score
            score1 = rf1.score(testSamples1, testLabels1)
            print("Score of", lnam, ":", score1)
            mAcc[p-1,n-1] = score1



    print("accuracy matrix of",lnam,":\n",mAcc)
    mf.plot_row(mAcc, x_label, lnam)
    #mf.plot_col(mAcc, y_label, lnam)

    return mAcc

def testFullImage(p, n, tif,lnam):
    # split into training and test data
    shuffled = ut.shuffle_data(data1[:, 2:], (100-p), where1)
    trainingSample1 = shuffled[0]
    trainingLabels1 = shuffled[1]
    trainingLabels1 = np.array([int(trainingLabels1[x][0]) for x in range(0, len(trainingLabels1))])
    testSamples1 = shuffled[2]
    testLabels1 = shuffled[3]
    testLabels1 = np.array([int(testLabels1[x][0]) for x in range(0, len(testLabels1))])
    k1 = shuffled[4]
    idx_shuffled1 = shuffled[5]

    rf1 = RandomForestClassifier(n_estimators=n)

    # Fitting the model to our training samples
    dtresult1 = rf1.fit(trainingSample1, trainingLabels1)

    # Predicting over test samples
    test_pred1 = rf1.predict(testSamples1)
    # Calculating the model score
    score1 = rf1.score(testSamples1, testLabels1)
    print("Score of image: ", score1)

    # Visualize the confusion matrix
    result_names = ['Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
    plotconfusionmatrix(testLabels1, test_pred1, result_names)

    nt = len(tif)
    # read Full Image As Array
    if nt == 1:
        tifsrc = gdal.Open(path_tif + "/" + tif[0])
        stack1 = ut.readFullImageAsArray(tifsrc)
    elif nt == 3:
        tifsrc = gdal.Open(path_tif + "/" + tif[0])
        tifsrc2 = gdal.Open(path_tif + "/" + tif[1])
        tifsrc3 = gdal.Open(path_tif + "/" + tif[2])
        stack1 = ut.readThreeFullImageAsArray(tifsrc, tifsrc2, tifsrc3)
    else:
        print("Wrong input number of image(s).")

    # predict the values
    prediction1 = ut.test_image(rf1, stack1)

    # plot the prediction and save into tif
    layer = ut.prediction_to_image(prediction1)

    plot_tif(layer, result_names)
    if nt == 1:
        ut.write_tiff(layer, tifsrc, path_out, (name_classified_sgl +"_"+ lnam))
    elif nt == 3:
        ut.write_tiff(layer, tifsrc, path_out, name_classified_mul)
    else:
        print("wrong.")

if __name__ == "__main__":
    # list of tif images, choose one or three
    labelname = ("1st", "2nd", "3rd", "all")
    tif0 = [[name_source_img1], [name_source_img2], [name_source_img3], [name_source_img1, name_source_img2, name_source_img3]]

    # choose which one to test: (change from 0 to 3)
    ti = 3

    sre = tif0[ti]
    lnam = labelname[ti]
    # Get the pixels from vector and raster
    data1, uniqueLabels1, columnNames1, m1, where1, idx1 = getGeneralSinglePixelValues(shape, path_tif, name_label_col, sre, rastermask=path_mask, subset=None, returnsubset=False)

    # draw out the sample area when you want to
    #target_names = ['NoSample','Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
    #plot_tif(m1,target_names)

    x_label = [20, 40, 60, 80]
    y_label = [50, 100, 150]
    ma = testAcc(data1, lnam)
    re = np.where(ma == np.max(ma))
    mp = x_label[int(re[0][0])]
    mn = y_label[int(re[1][0])]
    print(mp, mn)
    testFullImage(mp, mn, sre,lnam)
