################################################################
#In this file, ti is the number you could select to choose which image you could like to run: (feel free to change in line169)
# 0 for the 1st image, 054112895010_20.tif
# 1 for the 2nd image, 054112895030_20.tif
# 2 for the 3rd image, 054330675010_20.tif
# 3 for all the images.
################################################################

# Import a library for raster management and the estimators used
from osgeo import gdal
import numpy as np
from osgeo import ogr
import os

# Importing code from our modules
from getPixelValues import getGeneralSinglePixelValues
from visualization import plot_tif, plotconfusionmatrix
import utility as ut
import Tools as tl
import myFunction as mf


# define variables
name_samples_img = "CropsSamples"
name_classified_sgl = "Crops_SVM_Classified_single"
name_classified_mul = "Crops_SVM_Classified_multit"
name_train_test_img = "Train_Test"
name_source_img1 = "054112895010_20.tif"
name_source_img2 = "054112895030_20.tif"
name_source_img3 = "054330675010_20.tif"
name_label_col = "Crop_label"

# set your local path of this python file
a = os.path.abspath('muitiparaSVM.py')
path_py = a[:-16]

# define locations of tif and shp data
path_plg = path_py + r"\STARS_Sentinel_Mali\polygon"
path_tif = path_py + r"\STARS_Sentinel_Mali\images"
path_mask = path_py + r"\STARS_Sentinel_Mali\remask.tif"
path_out = path_py + r"\STARS_Sentinel_Mali\{0}.tif"

# reprojection shp files
os.chdir(path_plg)
datasource = ogr.Open("mergeSHP_file.shp")
shape = ut.reproject_vector(datasource, epsg_from = 4326, epsg_to = 32630)

## get accuracy from a single image or three images using Random Forest
# p for percentage, C for cost, g for gamma, tifsrc for tif list,
def testAccSVM(data1, lnam):
    mAcc = np.empty(15, dtype=np.float).reshape(5, 3)

    for p in range(1, 6, 1):

        # Split part
        # split into training and test data
        shuffled = ut.shuffle_data(data1[:, 2:], (100-p*10), where1)
        trainingSample1 = shuffled[0]
        trainingLabels1 = shuffled[1]
        trainingLabels1 = np.array([int(trainingLabels1[x][0]) for x in range(0, len(trainingLabels1))])
        testSamples1 = shuffled[2]
        testLabels1 = shuffled[3]
        testLabels1 = np.array([int(testLabels1[x][0]) for x in range(0, len(testLabels1))])
        k1 = shuffled[4]
        idx_shuffled1 = shuffled[5]

        # Normalization part
        # Concatenated the train and test to scale the values of the pixels:
        X1 = np.append(trainingSample1, np.array(testSamples1.T, copy=False, subok=True, ndmin=2).T, axis=0)
        X1, val1 = tl.scale(X1)

        # Split again the train and test:
        trainingSample1 = X1[0:len(trainingSample1), :]
        testSamples1 = X1[len(trainingSample1):, :]

        # Define the names of the classes:
        target_names = ['Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
        # Training and test an SVM classifier:
        RES1 = tl.SVMclassification(trainingSample1, trainingLabels1, testSamples1, testLabels1, target_names, 6)

        # Calculating the model score
        print("Score of", name_source_img1, ":", RES1[1])
        print("Best C:", RES1[6][0], "Best G:", RES1[7][0])
        mAcc[p - 1, 0] = RES1[1]
        mAcc[p - 1, 1] = RES1[6][0]
        mAcc[p - 1, 2] = RES1[7][0]

    x_label = [90, 80, 70, 60, 50]
    mf.plot_onerow(mAcc, x_label, lnam)

    return mAcc

def testFullSVM(mp, mc, mg, tif, data1, lnam):
    nt = len(tif)
    if nt == 1:
        tifsrc = gdal.Open(path_tif + "/" + tif[0])
        stack1 = ut.readFullImageAsArray(tifsrc)      # in this function, we delete the top 10 rows to split all 0 values
    elif nt == 3:
        tifsrc = gdal.Open(path_tif + "/" + tif[0])
        tifsrc2 = gdal.Open(path_tif + "/" + tif[1])
        tifsrc3 = gdal.Open(path_tif + "/" + tif[2])
        stack1 = ut.readThreeFullImageAsArray(tifsrc, tifsrc2, tifsrc3)
    else:
        print("Wrong input number of image(s).")

    # Select 5% from all the pixels within the crops and they are split for training and testing:
    percentage = 100-mp

    # split into training and test data
    shuffled = ut.shuffle_data(data1[:, 2:], percentage, where1)
    trainingSample1 = shuffled[0]
    trainingLabels1 = shuffled[1]
    trainingLabels1 = np.array([int(trainingLabels1[x][0]) for x in range(0, len(trainingLabels1))])
    testSamples1 = shuffled[2]
    testLabels1 = shuffled[3]
    testLabels1 = np.array([int(testLabels1[x][0]) for x in range(0, len(testLabels1))])
    # Number of training data and Index of pixels selected:
    k1 = shuffled[4]
    idx_shuffled1 = shuffled[5]

    # Concatenated the train and test to scale the values of the pixels:
    X1 = np.append(trainingSample1, np.array(testSamples1.T, copy=False, subok=True, ndmin=2).T, axis=0)
    X1, val1 = tl.scale(X1)

    # Split again the train and test:
    trainingSample1 = X1[0:len(trainingSample1), :]
    testSamples1 = X1[len(trainingSample1):, :]

    # Save a tiff of the crops with values 1 if they are train and 2 if they are test
    train_test_matrix = ut.write_training_test_image(tifsrc, path_out, idx1, idx_shuffled1, percentage,
                                                     name_train_test_img)

    # Define the names of the classes:
    result_names = ['Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
    # Training and test an SVM classifier:
    RES1 = tl.SVMclassificationBest(trainingSample1, trainingLabels1, testSamples1, testLabels1, result_names, mc, mg)

    # Calculating the model score
    print("Score of", name_source_img1, ":", RES1[1])

    # Visualize the confusion matrix
    target_names = ['Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
    plotconfusionmatrix(testLabels1, RES1[5], target_names)

    # Normalize the array using the training maximum and minimum values:
    stack1 = tl.norm_stack(stack1, val1)

    # Predict the image classes:
    prediction1 = ut.test_image(RES1[0], stack1)

    # plot the prediction and save into tif
    layer = ut.prediction_to_image(prediction1)

    plot_tif(layer, result_names)
    if nt == 1:
        ut.write_tiff(layer, tifsrc, path_out, (name_classified_sgl +"_"+ lnam))
    elif nt == 3:
        ut.write_tiff(layer, tifsrc, path_out, name_classified_mul)
    else:
        print("wrong~~~~~~~~~~~")


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
    target_names = ['NonSample', 'Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']
    plot_tif(m1, target_names)

    x_label = [90, 80, 70, 60, 50]
    y_label = [50, 100, 150]
    ma = testAccSVM(data1, lnam)
    re = np.where(ma[:,0] == np.max(ma[:,0]))
    mp = x_label[int(re[0][0])]
    mc = ma[int(re[0][0]), 1]
    mg = ma[int(re[0][0]), 2]
    testFullSVM(mp, mc, mg, sre, data1, lnam)
