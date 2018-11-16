# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        getPixelValues.py
# Purpose:
#
# Author:      claudio piccinini
#
# Created:     11/09/2015
#-------------------------------------------------------------------------------

import math
import os

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdalconst
#import rasterstats
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import utility

ogr.UseExceptions()
gdal.UseExceptions()


def getSinglePixelValues(shapes, inraster, fieldname,rastermask=None, combinations='*', subset=None, returnsubset = False):
    """intersect polygons/multipolygons with multiband rasters
        IMPORTANT:
        polygons and raster must have the same coordinate system!!!
        feature falling partially or totally outside the raster will not be considered
        when passing the subset as a dictionary be sure to use the same rastermask options used for the subset source

    :param shapes: polygons/multipolygons shapefile
    :param inraster: multiband raster
    :param fieldname: vector fieldname that contains the labelvalue
    :param rastermask: raster where value 0 is the mask
    :param bandcombination: bandcombination : if true the
    :param combinations: possible values '*', [], None, [(),()]
        define if the result will contain columns with normalized difference indexes
                                    ndi = (bandj) - (bandi)/(bandj) + (bandi)
        '*' -> all combinations
        [] or None -> no combinations
        [(),()] -> a list of tuples , each tuple with 2 band numbers for which we want to
        calculate the NDI [(1,2), (3,4), .....]
    :param  subset: integer or dictionary
                    - integer percentage (> 0; <100) deciding how much of each polygon you want to consider
                    - a dictionary { polygonID: numpy.ndarray} where the numpy.ndarray is used to apply fancy index
                    to filter the polygon with ID == polygonID
    :param  returnsubset: bool, if true a subset datastructure { polygonID: numpy.ndarray} is returned
    :return: 1) a 2d numpy array
            --if combinations was [] or None: each row contains the polygonID column, the unique id column, the apixel
                   values for each raster band plus a column with the label:
                   the array shape is (numberpixels, nbands + 3)
            --if combination was '*' or [(),()]: each row contains the polygonID column, the unique id column, the pixel
                   values for each raster band, the NDI bands, plus a column with the label:
                   if conbination is '*' we get all the combinations
                   the array shape is (numberpixels, nbands + number of band combinations +3)

            -- if subset was 0< subset<=100 the numberpixels will be decreased
            2) a set with the unique labels
            3) a list with column names
            4) if returnsubset is True will return the subset datastructure { polygonID: numpy.ndarray}
    """

    #checking if combinations and subset parameters are correct
    if all([combinations != '*', type(combinations) != list , combinations is not None] ):
        raise TypeError("combinations should be '*' or [] or None or [(),()] ")

    elif type(combinations) == list and len(combinations)>0:
        if type(combinations[0]) != tuple:
            raise TypeError("combinations format should be [(),(),...] ")

    if all([type(subset) != int, type(subset) != dict, subset is not None]):
        raise TypeError('subset should be an integer, a dictionary, or None')
    elif type(subset) == int and not(0 < subset < 100):
        raise ValueError('subset should be more than 0 and less than 100 ')
    elif type(subset) == dict and not subset:
        raise ValueError('subset dictionary should not be empty')
    elif type(subset) == dict and not type(next(iter(subset.values()))) == np.ndarray:
        raise ValueError('subset should be a dictionary of ndarrays')

    if subset is None:
        returnsubset = False
    if returnsubset:
        subsetcollection = {}



    raster = None
    pixelmask = None
    shp = None
    lyr = None
    target_ds = None
    outDataSet = None
    outLayer = None
    outdata = []

    try:

        # Open data
        raster = gdal.Open(inraster,gdalconst.GA_ReadOnly)
        shp = ogr.Open(shapes)
        lyr = shp.GetLayer()

        sourceSR = lyr.GetSpatialRef()

        # get number of features; get number of bands
        featureCount = lyr.GetFeatureCount()
        nbands = raster.RasterCount

        # iterate features and extract unique labels
        classValues = []
        for feature in lyr:
            classValues.append(feature.GetField(fieldname))
        # get the classes unique values and reset the iterator
        uniqueLabels = set(classValues)
        lyr.ResetReading()

        # Get raster georeference info

        width = raster.RasterXSize
        height = raster.RasterYSize

        transform = raster.GetGeoTransform()
        xOrigin = minx = transform[0]
        yOrigin = maxy = transform[3]
        miny = transform[3] + width*transform[4] + height*transform[5]
        maxx = transform[0] + width*transform[1] + height*transform[2]
        pixelWidth = transform[1]
        pixelHeight = transform[5]


        numfeature = 0

        # keep trak of the number of ids, necessary to assign id to subsequent polygons
        idcounter=1

        # if we want the normalized indexes we need to add additional columns to the outputdata
        if combinations:
            # get the number of combinations and the column names
            numberCombinations, comb_column_names = utility.combination_count(nbands)

            #combination_count() will return all the combination, but pixel value of  ndi A/B is just the inverse of ndi 2/1;
            # therefore we get only the first half of the combinations
            numberCombinations = numberCombinations/2
            comb_column_names = comb_column_names[0: int(numberCombinations)]

        if rastermask:
            pixelmask = gdal.Open(rastermask,gdalconst.GA_ReadOnly)

        for feat in lyr:

            numfeature +=1
            print ("working on feature %d of %d"%(numfeature,featureCount))

            # get the label and the polygon ID
            label = feat.GetField(fieldname)
            polygonID = feat.GetFID() + 1  # I add one to avoid the first polygonID==0

            # Get extent of feature
            geom = feat.GetGeometryRef()
            if geom.GetGeometryName() == "MULTIPOLYGON":
                count = 0
                pointsX = []
                pointsY = []
                for polygon in geom:
                    geomInner = geom.GetGeometryRef(count)
                    ring = geomInner.GetGeometryRef(0)
                    numpoints = ring.GetPointCount()
                    for p in range(numpoints):
                            lon, lat, z = ring.GetPoint(p)
                            pointsX.append(lon)
                            pointsY.append(lat)
                    count += 1
            elif geom.GetGeometryName() == "POLYGON":
                ring = geom.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                pointsX = []; pointsY = []
                for p in range(numpoints):
                        lon, lat, z = ring.GetPoint(p)
                        pointsX.append(lon)
                        pointsY.append(lat)

            else:
                raise Exception("ERROR: Geometry needs to be either Polygon or Multipolygon")

            xmin = min(pointsX)
            xmax = max(pointsX)
            ymin = min(pointsY)
            ymax = max(pointsY)

            #check if this feature is completely inside the raster, if not skip it
            if any([xmin < minx, xmax > maxx, ymin < miny, ymax > maxy]):
                print('feature with id = %d is falling outside the raster and will not be considered'%feat.GetFID())
                continue

            # Specify offset and rows and columns to read
            xoff = int((xmin - xOrigin)/pixelWidth)
            yoff = int((yOrigin - ymax)/pixelWidth)
            xcount = int((xmax - xmin)/pixelWidth)+1
            ycount = int((ymax - ymin)/pixelWidth)+1

            # Create memory target multiband raster
            target_ds = gdal.GetDriverByName("MEM").Create('', xcount, ycount, nbands, gdalconst.GDT_UInt16)
            target_ds.SetGeoTransform((
                xmin, pixelWidth, 0,
                ymax, 0, pixelHeight,
            ))

            # Create for target raster the same projection as for the value raster
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(raster.GetProjectionRef())
            target_ds.SetProjection(raster_srs.ExportToWkt())


            # create in memory vector layer that contains the feature
            drv = ogr.GetDriverByName("ESRI Shapefile")
            outDataSet = drv.CreateDataSource("/vsimem/memory.shp")
            outLayer = outDataSet.CreateLayer("memoryshp", srs=sourceSR, geom_type=lyr.GetGeomType())

            # set the output layer's feature definition
            outLayerDefn = lyr.GetLayerDefn()
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)


            # Rasterize zone polygon to raster
            # outputraster, list of bands to update, input layer, list of values to burn
            gdal.RasterizeLayer(target_ds, list(range(1, nbands+1)), outLayer, burn_values=[label]*nbands)

            # Read rasters as arrays
            dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
            datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

            if rastermask: #if we have a mask (e.g trees)
                pixelmasker = pixelmask.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                datamask = datamask * pixelmasker


            # extract the data for each band
            data = []
            for i in range(nbands):
                data.append(dataraster[i][datamask[i]>0])

            # define label data for this polygon
            label = (np.zeros(data[0].shape[0]) + label).reshape(data[0].shape[0],1)
            polygonID = (np.zeros(data[0].shape[0]) + polygonID).reshape(data[0].shape[0],1)

            id = np.arange(idcounter,(data[0].shape[0])+idcounter ).reshape(data[0].shape[0],1) # +1 is there to avoid first polygon different from 0

            # update the starting id for the next polygon
            idcounter += data[0].shape[0]

            vstackdata = np.vstack(data).T

            #calculate once indexes for subsetting polygons; we will use this in the next "if combinations"
            if subset:
                if type(subset) == int: #if the subset was a percentage we need to define the fancy indexer
                    subsize = int((polygonID.shape[0]) * subset/100)
                    idxsubsize = np.array(range(0, polygonID.shape[0]))
                    numpy.random.shuffle(idxsubsize)
                    idxsubsize = idxsubsize[:subsize]
                    #print(idxsubsize.shape)

                    if returnsubset: #if we want to return the subset datastructure we need add a key:value
                        subsetcollection[int(polygonID[0,0])] = idxsubsize

                else: #if the subset was a dictionary we extract the correct fancy indexer by key
                    #print(int(polygonID[0,0]))
                    idxsubsize = subset[int(polygonID[0,0])]
                    #print(idxsubsize)


            if combinations:  #  '*'  or [(),(),...]  -> all combinations or specific combinations

                if combinations == '*':  #all band combinations in the output
                    if subset:
                        # multi band samples and labels : shape -> num. subsetted pixels * (8 bands+ all number of band combinations + 3)
                        # use numpy fancy indexing to subset polygons
                        outdata.append(np.hstack((polygonID[idxsubsize], id[idxsubsize],  np.hstack((vstackdata, np.zeros((vstackdata.shape[0],numberCombinations))))[idxsubsize],label[idxsubsize])))

                    else: #no suset
                        # multi band samples and labels : shape -> num.pixels * (8 bands+ all number of band combinations + 3)
                        outdata.append(np.hstack((polygonID, id,  np.hstack((vstackdata, np.zeros((vstackdata.shape[0],numberCombinations)))),label)))

                else:  # specific band combinations
                    if subset:
                        # multi band samples and labels : shape -> num. subsetted pixels * (8 bands+ custom band combinations + 3)
                        # use numpy fancy indexing to subset polygons
                        outdata.append(np.hstack((polygonID[idxsubsize], id[idxsubsize],  np.hstack((vstackdata, np.zeros((vstackdata.shape[0],len(combinations)))))[idxsubsize],label[idxsubsize])))
                    else: #no suset
                        # multi band samples and labels : shape -> num.pixels * (8 bands+ custom band combinations + 3)
                        outdata.append(np.hstack((polygonID, id,  np.hstack((vstackdata, np.zeros((vstackdata.shape[0],len(combinations))))),label)))

            elif not combinations or combinations is None:  # [] or None  -> no band combinations in the output

                if subset:
                    # multi band samples and labels : shape -> num. subsetted pixels * (8 bands+3)
                    # use numpy fancy indexing to subset polygons
                    outdata.append(np.hstack((polygonID[idxsubsize], id[idxsubsize],  vstackdata[idxsubsize], label[idxsubsize])))
                else: #no suset
                    # multi band samples and labels : shape -> num.pixels * (8 bands+3)
                    outdata.append(np.hstack((polygonID, id,  vstackdata, label)))



            # Mask zone of raster
            #zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
            # Calculate statistics of zonal raster
            #return np.average(zoneraster),np.mean(zoneraster),np.median(zoneraster),np.std(zoneraster),np.var(zoneraster)

            # give control back to c++ to free memory
            target_ds = None
            outLayer = None
            outDataSet = None

        # id = np.arange(1,(data[0].shape[0])+1).reshape(data[0].shape[0],1)

        # define the output data
        outdata = np.vstack(outdata)

        # store the field names
        columnNames = ["polyID\t", "id\t"]
        for i in range(nbands):
            columnNames.append("band" + str(i+1)+"\t")
        # if we want the normalized indexes we need additional columns to the outputdata
        if combinations == '*':  #this is when we want all the combinations

            columnNames += utility.column_names_to_string(comb_column_names)

            # calculate the NDI for all the band combinations
            print("calculating NDI for "+str(numberCombinations) + " columns, it will take some time")
            for i in comb_column_names:
                #get column index  ->  i[0]-i[1]/i[0]+i[1]

                #which column we want to update? # the 2 is there because the first 2 columns are the polygonid/id
                idx = comb_column_names.index(i)+ nbands + 2
                #calculate index  # the +1 is there because the first two columns are polyid and id
                outdata[:, idx] = (outdata[:, i[0]+1] - outdata[:, i[1]+1]) / (outdata[:, i[0]+1] + outdata[:, i[1]+1])
                print(".", end="")
            print()

        elif all([combinations ,combinations is not None]): # this is when we want specific combinations -> [(),(),...]

            columnNames += utility.column_names_to_string(combinations)

            # calculate the NDI for all the band combinations
            print("calculating NDI for "+str(len(combinations)) + " columns, it will take some time")
            for i in combinations:
                # get column index  ->  i[0]-i[1]/i[0]+i[1]

                # which column we want to update? # the 2 is there because the first 2 columns are the polygonid/id
                idx = combinations.index(i)+ nbands + 2
                # calculate index  # the +1 is there because the first two columns are polyid and id
                outdata[:,idx] = (outdata[:, i[0]+1] - outdata[:, i[1]+1]) / (outdata[:, i[0]+1] + outdata[:, i[1]+1])
                print(".", end="")
            print()

        columnNames.append("label")

        if returnsubset:
            if type(subset) == int:
                return (outdata, uniqueLabels, columnNames, subsetcollection)
            else: #if the subset was already a datasturecture we just return it
                return (outdata, uniqueLabels, columnNames, subset)
        return (outdata, uniqueLabels, columnNames)

    finally:

        #give control back to c++ to free memory
        if raster: raster = None
        if pixelmask:  pixelmask = None
        if shp: shp = None
        if lyr: lyr = None
        if target_ds: target_ds = None
        if outLayer: outLayer = None
        if outDataSet: outDataSet = None


def getGeneralSinglePixelValues_filter(shapes, folderpath, fieldname, inimgfrmt = ['.tif'], rastermask=None, subset=None, returnsubset = False):
    """ general function to intersect polygons/multipolygons with a group of multiband rasters
        IMPORTANT
        polygons and raster must have the same coordinate system!!!
        the bands of a raster must have the same data type
        feature falling partially or totally outside the raster will not be considered
        when passing the subset as a dictionary be sure to use the same rastermask options used for the subset source

    :param shapes: polygons/multipolygons shapefile
    :param folderpath: folder with multiband rasters
    :param fieldname: vector fieldname that contains the labelvalue
    :param inimgfrmt: a list of image formats necessary to filter the input folder content (default is tif format)
    :param rastermask: raster where value 0 is the mask
    :param  subset: integer or dictionary
                    - integer percentage (> 0; <100) deciding how much of each polygon you want to consider
                    - a dictionary { polygonID: numpy.ndarray} where the numpy.ndarray is used to apply fancy index
                    to filter the polygon with ID == polygonID
    :param  returnsubset: bool, if true a subset datastructure { polygonID: numpy.ndarray} is returned
    :return: 1) a 2d numpy array,
                each row contains the polygonID column, the unique id column, the pixel
                values for each raster band plus a column with the label:
                the array shape is (numberpixels, numberofrasters*nbands + 3)

                if mask the max numberpixels  per polygon may decrease
                if subset the numberpixels will decrease
             2) a set with the unique labels
             3) a list with column names
             4) if returnsubset is True will return the subset datastructure { polygonID: numpy.ndarray}
    """

    #checking if subset parameters is correct
    if all([type(subset) != int, type(subset) != dict, subset is not None]):
        raise TypeError('subset should be an integer, a dictionary, or None')
    elif type(subset) == int and not(0 < subset < 100):
        raise ValueError('subset should be more than 0 and less than 100 ')
    elif type(subset) == dict and not subset:
        raise ValueError('subset dictionary should not be empty')
    elif type(subset) == dict and not type(next(iter(subset.values()))) == np.ndarray:
        raise ValueError('subset should be a dictionary of ndarrays')


    subsetcollection = {}


    raster = None
    shp = None
    lyr = None
    target_ds = None
    outDataSet = None
    outLayer = None
    band = None
    pixelmask = None
    outdata = []

    try:

        shp = ogr.Open(shapes)
        lyr = shp.GetLayer()

        sourceSR = lyr.GetSpatialRef()

        # get number of features; get number of bands
        featureCount = lyr.GetFeatureCount()

        # iterate features and extract unique labels
        classValues = []
        for feature in lyr:
            classValues.append(feature.GetField(fieldname))
        # get the classes unique values
        uniqueLabels = set(classValues)
        # reset the iterator
        lyr.ResetReading()
        # get the content of the images directory
        imgs= os.listdir(folderpath)

        imgcounter = 0  #keep track of the image number
        label = None
        columnNames = []
        labels = []  #this will store all the labels

        # iterate all the files and keep only the ones with the correct extension
        for i in imgs:
            # filter content, we want files with the correct extension
            if os.path.isfile(folderpath+'/'+i) and (os.path.splitext(folderpath+'/'+i)[-1] in inimgfrmt) :
                # increase the image counter and open raster data
                imgcounter += 1
                raster = gdal.Open(folderpath+'/'+i, gdalconst.GA_ReadOnly)
                nbands = raster.RasterCount

                # we need to get the raster datatype for later use (assumption:every band has the same data type)
                band = raster.GetRasterBand(1)
                raster_data_type = band.DataType

                # Get raster georeference info

                width = raster.RasterXSize
                height = raster.RasterYSize
                #print(width, height)

                transform = raster.GetGeoTransform()
                xOrigin = minx = transform[0]
                yOrigin = maxy = transform[3]
                miny = transform[3] + width*transform[4] + height*transform[5]
                maxx = transform[0] + width*transform[1] + height*transform[2]
                pixelWidth = transform[1]
                pixelHeight = transform[5]

                numfeature = 0

                # keep trak of the number of ids, necessary to assign id to subsequent polygons
                idcounter = 1

                # reset the iterator
                lyr.ResetReading()

                intermediatedata = []


                if rastermask:
                    pixelmask = gdal.Open(rastermask,gdalconst.GA_ReadOnly)

                for feat in lyr:

                    numfeature += 1
                    print("working on feature %d of %d, raster %s" % (numfeature, featureCount, i))

                    #get the label and the polygon ID
                    label = feat.GetField(fieldname)
                    polygonID = feat.GetFID() + 1  #I add one to avoid the first polygonID==0

                    #  Get extent of feature
                    geom = feat.GetGeometryRef()
                    if geom.GetGeometryName() == "MULTIPOLYGON":
                        count = 0
                        pointsX = []; pointsY = []
                        for polygon in geom:
                            geomInner = geom.GetGeometryRef(count)
                            ring = geomInner.GetGeometryRef(0)
                            numpoints = ring.GetPointCount()
                            for p in range(numpoints):
                                lon, lat, z = ring.GetPoint(p)
                                pointsX.append(lon)
                                pointsY.append(lat)
                            count += 1
                    elif geom.GetGeometryName() == "POLYGON":
                        ring = geom.GetGeometryRef(0)
                        numpoints = ring.GetPointCount()
                        pointsX = []
                        pointsY = []
                        for p in range(numpoints):
                            lon, lat, z = ring.GetPoint(p)
                            pointsX.append(lon)
                            pointsY.append(lat)

                    else:
                        raise Exception("ERROR: Geometry needs to be either Polygon or Multipolygon")

                    xmin = min(pointsX)
                    xmax = max(pointsX)
                    ymin = min(pointsY)
                    ymax = max(pointsY)

                    #check if this feature is completely inside the raster, if not skip it
                    if any([xmin < minx, xmax > maxx, ymin < miny, ymax > maxy]):
                        print('feature with id = %d is falling outside the raster and will not be considered'%feat.GetFID())
                        continue

                    # Specify offset and rows and columns to read
                    xoff = int((xmin - xOrigin)/pixelWidth)
                    yoff = int((yOrigin - ymax)/pixelWidth)
                    xcount = int((xmax - xmin)/pixelWidth)+1
                    ycount = int((ymax - ymin)/pixelWidth)+1

                    # Create memory target multiband raster, with the same nbands and datatype as the input raster
                    target_ds = gdal.GetDriverByName("MEM").Create("", xcount, ycount, nbands, raster_data_type)
                    target_ds.SetGeoTransform((
                        xmin, pixelWidth, 0,
                        ymax, 0, pixelHeight,
                    ))

                    # Create for target raster the same projection as for the value raster
                    raster_srs = osr.SpatialReference()
                    raster_srs.ImportFromWkt(raster.GetProjectionRef())
                    target_ds.SetProjection(raster_srs.ExportToWkt())

                    #create in memory vector layer that contains the feature
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    outDataSet = drv.CreateDataSource("/vsimem/memory.shp")
                    outLayer = outDataSet.CreateLayer("memoryshp", srs=sourceSR, geom_type=lyr.GetGeomType())

                    # set the output layer's feature definition
                    outLayerDefn = lyr.GetLayerDefn()
                    # create a new feature
                    outFeature = ogr.Feature(outLayerDefn)
                    # set the geometry and attribute
                    outFeature.SetGeometry(geom)
                    # add the feature to the shapefile
                    outLayer.CreateFeature(outFeature)

                    # Rasterize zone polygon to raster
                    # outputraster, list of bands to update, input layer, list of values to burn
                    gdal.RasterizeLayer(target_ds, list(range(1,nbands+1)), outLayer, burn_values=[label]*nbands)

                    # Read rasters as arrays
                    dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                    datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

                    if rastermask: #if we have a mask (e.g trees)
                        pixelmasker = pixelmask.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                        datamask = datamask * pixelmasker

                    #extract the data for each band
                    data = []
                    for j in range(nbands):
                        data.append(dataraster[j][datamask[j]>0])

                    if imgcounter == 1:
                        #define label data for this polygon
                        label = (np.zeros(data[0].shape[0]) + label).reshape(data[0].shape[0],1)
                        polygonIDarray = (np.zeros(data[0].shape[0]) + polygonID).reshape(data[0].shape[0],1)
                        # fill in the list with all the labels, this will be the last column in the final output


                    id = np.arange(idcounter,(data[0].shape[0]) + idcounter).reshape(data[0].shape[0], 1) #+1 is there to avoid first polygon different from 0

                    # update the starting id for the next polygon
                    idcounter += data[0].shape[0]
                    vstackdata = np.vstack(data).T


                    #if subset we need to define the correct fancy indexing
                    if subset:

                        #if the subset was a percentage we need to define the fancy indexer
                        #we can get it only for the first image
                        if type(subset) == int and imgcounter == 1:
                            subsize = int((polygonIDarray.shape[0]) * subset/100)
                            idxsubsize = np.array(range(0, polygonIDarray.shape[0]))
                            numpy.random.shuffle(idxsubsize)
                            idxsubsize = idxsubsize[:subsize]
                            #print(idxsubsize.shape)

                            #we store the fancy index for this polygon
                            subsetcollection[int(polygonID)] = idxsubsize

                        #if this is not the first image we get the correct fancy index from the collection
                        elif type(subset) == int:
                            idxsubsize = subsetcollection[int(polygonID)]

                        else: #if the subset was a dictionary we extract the correct fancy indexer by key
                            print(int(polygonID))
                            idxsubsize = subset[int(polygonID)]
                            #print(idxsubsize)

                        # and we apply the fancy indexing
                        if imgcounter == 1:
                            intermediatedata.append(np.hstack((polygonIDarray, id,  vstackdata))[idxsubsize])
                            labels.append(label[idxsubsize])
                        else:
                            intermediatedata.append(vstackdata[idxsubsize])

                    else: #there is no subset to apply, take all
                        if imgcounter == 1:
                            intermediatedata.append(np.hstack((polygonIDarray, id,  vstackdata)))
                            labels.append(label)
                        else:
                            intermediatedata.append(vstackdata)

                    # Mask zone of raster
                    #zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
                    # Calculate statistics of zonal raster
                    #return np.average(zoneraster),np.mean(zoneraster),np.median(zoneraster),np.std(zoneraster),np.var(zoneraster)

                    #give control back to c++ to free memory
                    target_ds = None
                    outLayer = None
                    outDataSet = None

                #########END for feat in lyr


                #store the field names
                if imgcounter == 1:
                    columnNames += ["polyID\t","id\t"]
                for k in range(nbands):
                    columnNames.append(i + "_b" + str(k+1)+"\t")

                # stack vertically the output of each feature class
                outdata.append (np.vstack(intermediatedata))

        ########## END for i in imgs

        # stack horizontally
        outdata = np.hstack(outdata)
        # finally append the lables at the end
        outdata = np.hstack((outdata, np.vstack(labels)))

        columnNames.append("label")

        if returnsubset:
            if type(subset) == int:
                return (outdata, uniqueLabels, columnNames, subsetcollection)
            else: #if the subset was already a datastucture we just return it
                return (outdata, uniqueLabels, columnNames, subset)
        return (outdata, uniqueLabels, columnNames)

    finally:

        #give control back to c++ to free memory
        if raster:
            raster = None
        if pixelmask:
            pixelmask = None
        if shp:
            shp = None
        if lyr:
            lyr = None
        if target_ds:
            target_ds = None
        if outLayer:
            outLayer = None
        if outDataSet:
            outDataSet = None
        if band:
            band = None


def getGeneralSinglePixelValues(shapes, folderpath, fieldname, images, rastermask=None, subset=None, returnsubset = False):
    """ general function to intersect polygons/multipolygons with a group of multiband rasters
        IMPORTANT
        polygons and raster must have the same coordinate system!!!
        the bands of a raster must have the same data type
        feature falling partially or totally outside the raster will not be considered
        when passing the subset as a dictionary be sure to use the same rastermask options used for the subset source

    :param shapes: polygons/multipolygons shapefile
    :param folderpath: folder with multiband rasters
    :param fieldname: vector fieldname that contains the labelvalue
    :param images: a list of images to process (not the absolute path)
    :param rastermask: raster where value 0 is the mask
    :param  subset: integer or dictionary
                    - integer percentage (> 0; <100) deciding how much of each polygon you want to consider
                    - a dictionary { polygonID: numpy.ndarray} where the numpy.ndarray is used to apply fancy index
                    to filter the polygon with ID == polygonID
    :param  returnsubset: bool, if true a subset datastructure { polygonID: numpy.ndarray} is returned
    :return: 1) a 2d numpy array,
                each row contains the polygonID column, the unique id column, the pixel
                values for each raster band plus a column with the label:
                the array shape is (numberpixels, numberofrasters*nbands + 3)

                if mask the max numberpixels  per polygon may decrease
                if subset the numberpixels will decrease
             2) a set with the unique labels
             3) a list with column names
             4) if returnsubset is True will return the subset datastructure { polygonID: numpy.ndarray}
    """

    #checking if subset parameters is correct
    if all([type(subset) != int, type(subset) != dict, subset is not None]):
        raise TypeError('subset should be an integer, a dictionary, or None')
    elif type(subset) == int and not(0 < subset < 100):
        raise ValueError('subset should be more than 0 and less than 100 ')
    elif type(subset) == dict and not subset:
        raise ValueError('subset dictionary should not be empty')
    elif type(subset) == dict and not type(next(iter(subset.values()))) == np.ndarray:
        raise ValueError('subset should be a dictionary of ndarrays')


    subsetcollection = {}


    raster = None
    shp = None
    lyr = None
    target_ds = None
    outDataSet = None
    outLayer = None
    band = None
    pixelmask = None
    outdata = []

    m = np.zeros((246, 308))
    where = np.zeros((246, 308))
    idx = []
    pixelnumber = 0

    try:

        shp = shapes #ogr.Open(shapes)
        lyr = shp.GetLayer()

        sourceSR = lyr.GetSpatialRef()

        # get number of features; get number of bands
        featureCount = lyr.GetFeatureCount()

        # iterate features and extract unique labels
        classValues = []
        for feature in lyr:
            classValues.append(feature.GetField(fieldname))
        # get the classes unique values
        uniqueLabels = set(classValues)
        # reset the iterator
        lyr.ResetReading()
        # get the content of the images directory
        ########imgs= os.listdir(folderpath)

        imgcounter = 0  #keep track of the image number
        label = None
        columnNames = []
        labels = []  #this will store all the labels

        # iterate all the files and keep only the ones with the correct extension
        for i in images:

            ##### filter content, we want files with the correct extension
            #####if os.path.isfile(folderpath+'/'+i) and (os.path.splitext(folderpath+'/'+i)[-1] in inimgfrmt) :

            # increase the image counter and open raster data
            imgcounter += 1
            raster = gdal.Open(folderpath+'/'+i,gdalconst.GA_ReadOnly)
            nbands = raster.RasterCount

            # we need to get the raster datatype for later use (assumption:every band has the same data type)
            band = raster.GetRasterBand(1)
            raster_data_type = band.DataType

            # Get raster georeference info

            width = raster.RasterXSize
            height = raster.RasterYSize

            transform = raster.GetGeoTransform()
            xOrigin = minx = transform[0] # Top left coordinates of the starting raster
            yOrigin = maxy =  transform[3]
            miny = transform[3] + width*transform[4] + height*transform[5]
            maxx = transform[0] + width*transform[1] + height*transform[2]
            pixelWidth = transform[1]
            pixelHeight = transform[5]

            numfeature = 0

            # keep trak of the number of ids, necessary to assign id to subsequent polygons
            idcounter = 1

            # reset the iterator
            lyr.ResetReading()

            intermediatedata = []


            if rastermask:
                pixelmask = gdal.Open(rastermask,gdalconst.GA_ReadOnly)

            for feat in lyr:

                numfeature += 1
                #print("working on feature %d of %d, raster %s" % (numfeature, featureCount, i))

                #get the label and the polygon ID
                label = int(float(feat.GetField(fieldname)))
                polygonID = feat.GetFID() + 1  #I add one to avoid the first polygonID==0

                #  Get extent of feature
                geom = feat.GetGeometryRef()
                if geom.GetGeometryName() == "MULTIPOLYGON":
                    count = 0
                    pointsX = []; pointsY = []
                    for polygon in geom:
                        geomInner = geom.GetGeometryRef(count)
                        ring = geomInner.GetGeometryRef(0)
                        numpoints = ring.GetPointCount()
                        for p in range(numpoints):
                            lon, lat, z = ring.GetPoint(p)
                            pointsX.append(lon)
                            pointsY.append(lat)
                        count += 1
                elif geom.GetGeometryName() == "POLYGON":
                    ring = geom.GetGeometryRef(0)
                    numpoints = ring.GetPointCount()
                    pointsX = []
                    pointsY = []
                    for p in range(numpoints):
                        lon, lat, z = ring.GetPoint(p)
                        pointsX.append(lon)
                        pointsY.append(lat)

                else:
                    raise Exception("ERROR: Geometry needs to be either Polygon or Multipolygon")

                xmin = min(pointsX)
                xmax = max(pointsX)
                ymin = min(pointsY)
                ymax = max(pointsY)

                #check if this feature is completely inside the raster, if not skip it
                if any([xmin < minx, xmax > maxx, ymin < miny, ymax > maxy]):
                    print('feature with id = %d is falling outside the raster and will not be considered'%feat.GetFID())
                    continue

                # Specify offset and rows and columns to read
                # Offset of the little raster, created to wrap each polygon feature
                xoff = int((xmin - xOrigin)/pixelWidth)
                yoff = int((yOrigin - ymax)/pixelWidth)
                # print("Offset of little raster: ", xoff, yoff)

                # Number of rows and columns of the little raster
                xcount = int((xmax - xmin)/pixelWidth)+1
                ycount = int((ymax - ymin)/pixelWidth)+1
                # print("Rows and cols of little raster: ", xcount, ycount)


                # Create memory target multiband raster, with the same nbands and datatype as the input raster
                target_ds = gdal.GetDriverByName("MEM").Create("", xcount, ycount, nbands, raster_data_type)
                target_ds.SetGeoTransform((
                    xmin, pixelWidth, 0,
                    ymax, 0, pixelHeight,
                ))

                # Create for target raster the same projection as for the value raster
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(raster.GetProjectionRef())
                target_ds.SetProjection(raster_srs.ExportToWkt())

                #create in memory vector layer that contains the feature
                drv = ogr.GetDriverByName("ESRI Shapefile")
                outDataSet = drv.CreateDataSource("/vsimem/memory.shp")
                outLayer = outDataSet.CreateLayer("memoryshp", srs=sourceSR, geom_type=lyr.GetGeomType())

                # set the output layer's feature definition
                outLayerDefn = lyr.GetLayerDefn()
                # create a new feature
                outFeature = ogr.Feature(outLayerDefn)
                # set the geometry and attribute
                outFeature.SetGeometry(geom)
                # add the feature to the shapefile
                outLayer.CreateFeature(outFeature)

                # Rasterize zone polygon to raster
                # outputraster, list of bands to update, input layer, list of values to burn
                gdal.RasterizeLayer(target_ds, list(range(1,nbands+1)), outLayer, burn_values=[label]*nbands)

                # Read rasters as arrays

                dataraster = raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

                if rastermask: #if we have a mask (e.g trees)
                    weizhi = pixelmask.ReadAsArray(xoff, yoff, xcount, ycount)
                    if weizhi is None:
                        pixelmasker = np.zeros((1,1))
                        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    else:
                        pixelmasker = pixelmask.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
                    datamask = datamask * pixelmasker

                    m, where, pixelnumber, idx = utility.fill_matrix(m, where, idx, xoff, yoff, xcount, ycount, datamask, pixelnumber)

                #extract the data for each band
                data = []
                for j in range(nbands):
                    data.append(dataraster[j][datamask[j]>0])

                if imgcounter == 1:
                    #define label data for this polygon
                    label = (np.zeros(data[0].shape[0]) + label).reshape(data[0].shape[0],1)
                    polygonIDarray = (np.zeros(data[0].shape[0]) + polygonID).reshape(data[0].shape[0],1)
                    # fill in the list with all the labels, this will be the last column in the final output

                id = np.arange(idcounter,(data[0].shape[0]) + idcounter).reshape(data[0].shape[0], 1) #+1 is there to avoid first polygon different from 0

                # update the starting id for the next polygon
                idcounter += data[0].shape[0]
                vstackdata = np.vstack(data).T


                #if subset we need to define the correct fancy indexing
                if subset:
                    #if the subset was a percentage we need to define the fancy indexer
                    #we can get it only for the first image
                    if type(subset) == int and imgcounter == 1:
                        subsize = int((polygonIDarray.shape[0]) * subset/100)
                        idxsubsize = np.array(range(0, polygonIDarray.shape[0]))
                        numpy.random.shuffle(idxsubsize)
                        idxsubsize = idxsubsize[:subsize]
                        print(idxsubsize.shape)

                        #we store the fancy index for this polygon
                        subsetcollection[int(polygonID)] = idxsubsize

                    #if this is not the first image we get the correct fancy index from the collection
                    elif type(subset) == int:
                        idxsubsize = subsetcollection[int(polygonID)]

                    else: #if the subset was a dictionary we extract the correct fancy indexer by key
                        print(int(polygonID))
                        idxsubsize = subset[int(polygonID)]
                        #print(idxsubsize)

                    # and we apply the fancy indexing
                    if imgcounter == 1:
                        intermediatedata.append(np.hstack((polygonIDarray, id,  vstackdata))[idxsubsize])
                        labels.append(label[idxsubsize])
                    else:
                        intermediatedata.append(vstackdata[idxsubsize])

                else: #there is no subset to apply, take all
                    if imgcounter == 1:
                        intermediatedata.append(np.hstack((polygonIDarray, id,  vstackdata)))
                        labels.append(label)
                    else:
                        intermediatedata.append(vstackdata)

                # Mask zone of raster
                #zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
                # Calculate statistics of zonal raster
                #return np.average(zoneraster),np.mean(zoneraster),np.median(zoneraster),np.std(zoneraster),np.var(zoneraster)

                #give control back to c++ to free memory
                target_ds = None
                outLayer = None
                outDataSet = None

            #########END for feat in lyr


            #store the field names
            if imgcounter == 1:
                columnNames += ["polyID\t","id\t"]
            for k in range(nbands):
                columnNames.append(i + "_b" + str(k+1)+"\t")

            # stack vertically the output of each feature class
            outdata.append (np.vstack(intermediatedata))

        ########## END for i in imgs

        # stack horizontally
        outdata = np.hstack(outdata)
        # finally append the lables at the end
        outdata = np.hstack((outdata, np.vstack(labels)))

        columnNames.append("label")

        if returnsubset:
            if type(subset) == int:
                return (outdata, uniqueLabels, columnNames, subsetcollection)
            else: #if the subset was already a datastucture we just return it
                return (outdata, uniqueLabels, columnNames, subset)
        return (outdata, uniqueLabels, columnNames, m, where, idx)

    finally:

        #give control back to c++ to free memory
        if raster:
            raster = None
        if pixelmask:
            pixelmask = None
        if shp:
            shp = None
        if lyr:
            lyr = None
        if target_ds:
            target_ds = None
        if outLayer:
            outLayer = None
        if outDataSet:
            outDataSet = None
        if band:
            band = None


def getMeanPixelValues(shapes, inraster, fieldname, combinations ='*', nodatavalue=None):
    """intersect shapefile with multiband rasters

        shapefile and raster must have the same coordinate system!!!

        feature falling partially or totally outside the raster will output nodatavalue for all bands and combinations

        a nodatavalue == 0 is not allowed because it will crash

    :param shapes: shapefile
    :param inraster: multiband raster
    :param fieldname: vector fieldname that contains the labelvalue

    :param combinations: possible values '*', [], None, [(),()]
        define if the result will contain columns with normalized difference indexes
                                    ndi = (bandj) - (bandi)/(bandj) + (bandi)
        '*' -> all combinations
        [] or None -> no combinations
        [(),()] -> a list of tuples , each tuple with 2 band numbers for which we want to
        calculate the NDI [(1,2), (3,4), .....]

    :param nodatavalue: the nodatavalue to assign when polygons falls outside the raster;
                        if None it will get the nodatavalue from the raster

    :return: - a 2d numpy array,
                -if bandcombination was False: each row contains the polygonID column, the unique id column, the average pixel
                values for each raster band plus a column with the label:
                   the array shape is (numberfeatures, nbands + 3)
                -if bandcombination was True: each row contains the polygonID column, the unique id column, the average pixel
                values for each raster band, the NDI bands, plus a column with the label:
                   the array shape is (numberfeatures, nbands + number of band combinations +3)
            - a set with the labels
            - a list with column names
    """

    #checking if combinations and subset parameters are correct
    if all([combinations != '*', type(combinations) != list , combinations is not None] ):
        raise TypeError("combinations should be '*' or [] or None or [(),()] ")

    elif type(combinations) == list and len(combinations)>0:
        if type(combinations[0]) != tuple:
            raise TypeError("combinations format should be [(),(),...] ")

    if nodatavalue == 0:
        raise  ValueError("a nodatavalue == 0.0 is not allowed")
    dataset = None
    layer = None
    try:
        # open image
        # import the NDVI raster and get number of bands
        dataset = gdal.Open(inraster,gdalconst.GA_ReadOnly)

        nbands = dataset.RasterCount

        if nodatavalue is None:
            band = dataset.GetRasterBand(1)
            nodatavalue = band.GetNoDataValue()
            band = None
            if nodatavalue == 0:
                raise  ValueError("the raster has a nodatavalue == 0.0, but this is not allowed,"
                                  " please rerun the code with a custom nodatavalue argument")
        #print(nodatavalue)

        print(nbands)

        dataset = None # destroy dataset

        # use rasterstat to get the average pixel values
        # also collect the polygon IDs
        meanValues = []
        polygonIDs = []

        # store the field names
        columnNames = ["polyID\t", "id\t"]


        for i in range(nbands):
            print("getting mean values for band %d" % (i+1))
            crossing = rasterstats.zonal_stats( shapes, inraster, stats=["mean"], band_num =i+1, geojson_out=True)
            #print (crossing)
            meanValues.append([j["properties"]["mean"] for j in crossing ])
            # we get the polygons ID only once because they are the same for the 8 bands
            if i == 0:
                polygonIDs.append([int(j["id"])+1 for j in crossing])
            columnNames.append("band" + str(i+1) + "\t")

        #print(meanValues)

        #create samples as a numpy array

        #define samples
        samples = np.array(meanValues).T

        #get rid of the rows with None values(this is where the polygon is outside the raster) and assign nodatavalue
        #samplesShape = samples.shape
        #print(samplesShape)
        #print(len(polygonIDs))
        mask = (samples == np.array(None))
        samples[mask] = nodatavalue
        #samples = samples[mask].reshape(-1,samples.shape[1])


        #if we want the band combinations, add columns to the samples
        if combinations: #'*'  or [(),(),...]  -> all combinations or specific combinations

            if combinations == '*':
                #get the number of combinations and the column names
                numberCombinations, combColumnNames = utility.combination_count(nbands)

                #combination_count() will return all the combination, but pixel value of  ndi A/B is just the inverse of ndi 2/1;
                # therefore we get only the first half of the combinations
                numberCombinations = numberCombinations/2
                combColumnNames = combColumnNames[0: int(numberCombinations)]


            elif all([combinations ,combinations is not None]): # this is when we want specific combinations -> [(),(),...]
                combColumnNames = combinations
                numberCombinations = len(combinations)


            #add columns to store th normalized indexes
            samples = np.hstack((samples, np.zeros((samples.shape[0], numberCombinations))))
            #add column names
            columnNames += utility.column_names_to_string(combColumnNames)
            #calculate the NDI for all the band combinations
            print("calculating NDI for " + str(numberCombinations) + " columns")

            for i in combColumnNames:
                #get column index  ->  i[0]-i[1]/i[0]+i[1]
                #which column we want to update?
                idx = combColumnNames.index(i)+ nbands
                #calculate index  # the -1 is there because the numpy array index starts from 0
                samples[:, idx] = np.where( (samples[:, i[0]-1] - samples[:, i[1]-1]) == 0.0, nodatavalue,(samples[:, i[0]-1] - samples[:, i[1]-1]) / (samples[:, i[0]-1] + samples[:, i[1]-1]))
                #samples[:, idx] = (samples[:, i[0]-1] - samples[:, i[1]-1]) / (samples[:, i[0]-1] + samples[:, i[1]-1])
                print(".", end="")
            print()

        columnNames.append("label")

        #define a column with the polygons id
        polygonIDs = np.array(polygonIDs).T
        id = polygonIDs

        #now we define the unique classes
        dataset = ogr.Open(shapes)
        layer = dataset.GetLayer()

        count = layer.GetFeatureCount()
        print("there are %d shapes" % count)

        #iterate features and extract labels
        print("getting feature labels...")
        classValues = []
        for feature in layer:
            classValues.append(feature.GetField(fieldname))
            print(".", end="")
            print()

        #give control back to c++ to free memory
        layer = None
        dataset = None

        #get the labels unique values
        print("filering feature labels...")
        uniqueLabels = set(classValues)

        #create classes as a numpy array
        labels = np.array(classValues).reshape(count, 1)

        return (np.hstack((polygonIDs, id, samples, labels.reshape(count,1))), uniqueLabels, columnNames)

    finally:
        if layer:
            layer = None
        if dataset:
            dataset = None