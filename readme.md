Read me first.

Title
Module 13: Spatio-Temporal Modelling and Analytics
Classification of Smallholder farming landscape in Africa

Written by -
Tian Mengge (S6035280)

Purpose:
In this practice, we want to compare two classifiers: Random Forest(RF) and Support Vector Machine(SVM). 

Datasets:
In this case, we use a stack of 3 WorldView-2 images resampled to 20m spatial resolution to classify 5 crops and 1 non-crop in mali, Africa. A shapefile of groundtruth information for training and testing the classifiers.

Display:
In this folder, there are several python files.
Only muitiparaRF.py and muitiparaSVM.py are used to display the comparison of two classification methods, RF and SVM.
Other files are libraries of functions to support this project.
In muitiparaRF.py, firstly, there are a kind of definitions of variables like paths, file names, label names. And then, set the current work path (some IDE do not need this, but some recognize this as necessary). Although the images we get have the same projection with the shapefile, they have different coordinate systems (one in meter unit, the other in decimal degree). So the next step is to reproject the shapefile. After all these preprocesses, there is a function named testAcc to try different values of parameters combination. In this case, we compare the influence of the percentage of training sample and the number of trees (in program, it is shown as n_estimator). The function testAcc could give we the accuracy of each combination. The following function use the given parameter to build a random forest model and use it for the whole image. In the main function, we identify the label for four situations and select one to run this program. The four situation is for the first, second, third and the stack of all images. After you select one situation, the program get samples, test different parameter combinations, choose the highest accuracy, and use that pair of parameters to build a RF model for the full image, show it in your screen and save it in your disk. It is nearly the same for muitiparaSVM.py, except under each test of different percentage, we find the best combination of cost and gamma and the samples are normalized to improve the accuracy. 
The only one thing we didn't achieve in python is to merge two shapfile. So we use ArcMap tool to finish this step and use the merged shapfile to do reproject.

There are few libraries are needed to install:
numpy
gdal
itertools
pillow
pylab
sklearn
matplotlib
csv
scipy

Or you can simply use your Command Prompt and run:
pip install -r requirement.txt

Warning:
If you meet error when excuting like this:
File "C:\Users\yourComputerName\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\metrics\pairwise.py", line 30, in <module>
    from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
ImportError: DLL load failed: The specified module could not be found.
This maybe cause by incomplete module files download through Command Prompt, there is a possible solution:
use your Command Prompt:
	pip uninstall numpy
	pip uninstall scipy
	pip uninstall matplotlib
	pip uninstall sklearn
and use your local wheel to install, like:
	pip install scikit_learn-0.19.2-cp36-cp36m-win_amd64.whl
There are some support wheels in this folder. 

Maybe the following library can not find by cmd and you can install by local:
itertools


Notice:
Please cautiously change the file path, because there are lots of relative paths used in those codes. Politely remind you that default files path is as follow:

Folders
project:
│   Fiona-1.7.12-cp36-cp36m-win_amd64.whl
│   GDAL-2.2.4-cp37-cp37m-win32.whl
│   GDAL-2.2.4-cp37-cp37m-win_amd64.whl
│   getPixelValues.py
│   matplotlib-2.2.2-cp37-cp37m-win32.whl
│   muitiparaRF.py
│   muitiparaSVM.py
│   myFunction.py
│   numpy-1.15.0rc2+mkl-cp37-cp37m-win32.whl
│   readme.md
│   requirement.txt
│   scikit_learn-0.19.2-cp36-cp36m-win32.whl
│   scikit_learn-0.19.2-cp36-cp36m-win_amd64.whl
│   scikit_learn-0.19.2-cp37-cp37m-win32 (1).whl
│   scikit_learn-0.19.2-cp37-cp37m-win32.whl
│   scikit_learn-0.19.2-cp37-cp37m-win_amd64.whl
│   scipy-1.1.0-cp37-cp37m-win32.whl
│   Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl
│   Tools.py
│   utility.py
│   visualization.py
│
└───STARS_Sentinel_Mali
    │   Crops_RF_Classified_multit.tif
    │   Crops_RF_Classified_single_1st.tif
    │   Crops_RF_Classified_single_2nd.tif
    │   Crops_RF_Classified_single_3rd.tif
    │   Crops_SVM_Classified_multit.tif
    │   Crops_SVM_Classified_single_1st.tif
    │   Crops_SVM_Classified_single_2nd.tif
    │   Crops_SVM_Classified_single_3rd.tif
    │   remask.aux
    │   remask.rrd
    │   remask.tif
    │   shpSample.tif
    │   Train_Test.tif
    │
    ├───images
    │       054112895010_20.tif
    │       054112895030_20.tif
    │       054330675010_20.tif
    │
    ├───polygon
    │       1st_onerowAcc.jpg
    │       CropsKML_file.kml
    │       CropsSHP_file.dbf
    │       CropsSHP_file.prj
    │       CropsSHP_file.qpj
    │       CropsSHP_file.shp
    │       CropsSHP_file.shx
    │       mergeSHP_file.cpg
    │       mergeSHP_file.dbf
    │       mergeSHP_file.prj
    │       mergeSHP_file.sbn
    │       mergeSHP_file.sbx
    │       mergeSHP_file.shp
    │       mergeSHP_file.shp.xml
    │       mergeSHP_file.shx
    │       NonCropsSHP_file.cpg
    │       NonCropsSHP_file.dbf
    │       NonCropsSHP_file.fix
    │       NonCropsSHP_file.prj
    │       NonCropsSHP_file.shp
    │       NonCropsSHP_file.shx
    │       _onerowAcc.jpg
    │       _rowAcc.jpg
    │
    └───testpolygons
            1st_colAcc.jpg
            1st_rowAcc.jpg
            _colAcc.jpg
            _onerowAcc.jpg
            _rowAcc.jpg