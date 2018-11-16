#change the path of dataSource to your local shp file
from osgeo import ogr
from PIL import Image
import matplotlib.pyplot as plt
import pylab as pl
import os, sys

def myAddField(path, name, fieldn, value):
    # Get the input Layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(path+"/"+name, 1) #1 is read/write

    #get layer:
    layer = dataSource.GetLayer()
    if layer[1].GetField(fieldn) == value:
        print("already created this field.")
    else:
        # Add an ID field
        idField = ogr.FieldDefn(fieldn, ogr.OFTInteger)
        layer.CreateField(idField)

    for i in layer:
         # feat = lyr.GetFeature(i)
        layer.SetFeature(i)
        i.SetField(fieldn, value)
        layer.SetFeature(i)

    fea = None
    idField = None
    layer = None
    dataSource = None

# this function can only be used in this case for crop tif image's 0 value lines
def myCrop(path, srename, outname):
    # size is width/height
    img = Image.open(path+r"/"+srename)

    box = (0, 11, 308, 246)
    area = img.crop(box)
    area.save(outname+'.tif')
    result_names = ['Maize', 'Millet', 'Peanut', 'Sorghum', 'Cotton', 'nonCrop']

    area.close()
    img.close()

def plot_row(mAcc, x_label, tif):
    # plot the influence of Influence of training numbers using RF
    pl.figure(figsize=(6, 4))
    p1 = pl.plot(x_label, mAcc[:, 0], label="n: 50")
    p2 = pl.plot(x_label, mAcc[:, 1], label="n: 100")
    p3 = pl.plot(x_label, mAcc[:, 2], label="n: 150")
    pl.xlabel("change of training numbers(%)")
    pl.ylabel("Accuracy")
    pl.title("RF: Influence of training numbers ("+tif+ ")")
    pl.legend()
    pl.savefig('_rowAcc.jpg')
    pl.show()

def plot_onerow(mAcc, x_label, tif):
    # plot the influence of Influence of training numbers using RF
    pl.figure(figsize=(6, 4))
    p1 = pl.plot(x_label, mAcc[:, 0])
    pl.xlabel("change of training numbers(%)")
    pl.ylabel("Accuracy")
    pl.title("SVM: Influence of training numbers ("+tif+ ")")
    pl.savefig(tif+'_onerowAcc.jpg')
    pl.show()

def plot_col(mAcc, y_label, tif):
    # plot the influence of Influence of training numbers using RF
    pl.figure(figsize=(6, 4))
    p1 = pl.plot(y_label, mAcc[0, :], label="p: 80")
    p2 = pl.plot(y_label, mAcc[1, :], label="p: 60")
    p3 = pl.plot(y_label, mAcc[2, :], label="p: 40")
    p4 = pl.plot(y_label, mAcc[3, :], label="p: 20")
    pl.xlabel("change of estimators: n")
    pl.xlim(40, 160)
    pl.ylabel("Accuracy")
    pl.title("RF: Influence of estimator numbers("+tif+ ")")
    pl.legend()
    pl.savefig(tif+'_colAcc.jpg')
    pl.show()


if __name__ == "__main__":
    path = r"C:\Users\TianMG\Documents\itc\M13\project\STARS_Sentinel_Mali\testpolygons"
    path1 = r"C:\Users\TianMG\Documents\itc\M13\project\STARS_Sentinel_Mali"
    name = "NonCropsSHP_file.shp"
    fieldn = "Croops"
    value = 6
    #myAddField(path, name, fieldn, value)


    path_py = r"C:\Users\TianMG\Documents\itc\M13\project"
    os.chdir(path_py)
    path_tif = path_py + r"\STARS_Sentinel_Mali\images"
    name_source_img1 = "054112895010_20.tif"
    tif = [name_source_img1]

    myCrop(path1,"Crops_RF_test.tif", "crop_image_my")
    # There's currently no support in pillow for multiband images at more than 8bits per channel. That one appears to be RGB
    # myCrop(path_tif, "054112895010_20.tif", tif[0][:-8])
