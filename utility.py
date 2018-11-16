# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        utility.py
# Purpose:      some utility functions:
#                    - get a gdal/ogr dataset spatial reference name
#                    - get the spatial reference in different formats from a GDAL/OGR dataset
#                    - to reproject raster or vectors
#                    - get multiple indices for a simple varsalue in a list
#                    - to get envelope coordinates
#                    - to compare attribute tables
#                    - to save a gdal/ogr data source
#                    - to get the first geometry from a shapefile.
#                    - Determine if the current GDAL is built with SpatiaLite support
#                    - Print attribute values in a layer
#                    - Print capabilities for a driver, datasource, or layer
#                    - Print a list of available drivers
#                    - Print a list of layers in a data source
#                    - Get a geometry string for printing attributes
#                    - Get attribute values from a feature
#                    - Get the datasource and layer from a filename
#                    - Shuffle a 2d numpy array and split in training and validation
#                    - Get a list of band combinations as a string
#                    - Convert a list of band combinations to a string
#                    - Rescale image (need orfeo installed)
#                    - execute an external executable file
#                    - execute an external python script
#                    - Get min max values from an image band
#                    - filter files by extension name
#
# Author:      Claudio Piccinini; Chris Garrard
#
# Created:     13/03/2015
# -------------------------------------------------------------------------------

import sys
import math
import codecs
import os

import numpy as np
import osgeo #this is necessary for the type comparison in some methods
from osgeo import osr
from osgeo import gdal
from osgeo import ogr
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

def fill_matrix(m, where, idx, xoff, yoff, xcount, ycount, pixelmask, pixelnumber):
    # print("xoff: {0}, yoff: {1}, xcount: {2}, ycount: {3}".format(xoff, yoff, xcount, ycount))
    v = pixelmask.max()
    for i in range(0, ycount):
        for j in range(0, xcount):
            if pixelmask[0, i, j] == v:
                m[i+yoff][j+xoff] = v
                where[i+yoff][j+xoff] = pixelnumber
                idx.append((pixelnumber, i+yoff, j+xoff))
                pixelnumber += 1
    return [m, where, pixelnumber, idx]

def write_tiff(m, tifsrc, path_out, name):
    rows = tifsrc.RasterXSize
    cols = tifsrc.RasterYSize
    prj_wkt = tifsrc.GetProjectionRef()
    geotransform = tifsrc.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path_out.format(name), rows, cols, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(prj_wkt)
    outband=ds.GetRasterBand(1)
    outband.WriteArray(m)
    ds = None
    outband = None

def pixels_placement(idx, idx_shuffled, percentage):
    where = np.zeros((246, 308))
    index_matrix = np.zeros((246, 308))
    npixels = int((len(idx)) * percentage/100)
    dic = defaultdict(list)

    for index in idx:
        key = index[0]
        row = index[1]
        col = index[2]
        dic[key] = [row, col]
        index_matrix[row, col] = 1

    counter = 0
    for index in idx_shuffled:
        pixel = index[1]
        row, col = dic[pixel]
        if counter <= npixels:
            where[row, col] = 1
        else:
            where[row, col] = 2
        counter += 1
    return where

def write_training_test_image(tif, path_out, idx, idx_shuffled, percentage, name):
    placed = pixels_placement(idx, idx_shuffled, percentage)
    write_tiff(placed, tif, path_out, name)
    return placed

def test_image(rf, stack):
    prev = 0
    prediction = np.array([])
    for i in range(10000, 80000, 10000): # Only reaches 1790000
        chunk = stack[prev:i, :]
        #print("\t\t Testing: ", prev, " to ", i)
        prev = i
        y_pred_chunk = rf.predict(chunk)
        prediction = np.append(prediction,y_pred_chunk, axis=0)
    chunk = stack[60000:]
    y_pred_chunk = rf.predict(chunk)
    prediction = np.append(prediction,y_pred_chunk, axis=0).reshape(-1, 1)
    return prediction

def prediction_to_image(prediction):
    rows = range(0, 246)
    cols = range(0, 308)
    layer = np.zeros((246, 308))
    for i, j in itertools.product(rows, cols):
        pos = (308 * i) + j
        layer[i, j] = prediction[pos]
    return layer

def readFullImageAsArray(tif):
    rows = tif.RasterXSize
    cols = tif.RasterYSize
    i = 0
    bands = []
    for band_number in range(1, 9):
        band = tif.GetRasterBand(band_number)
        data = band.ReadAsArray(0, 0, rows, cols)
        # delete the top 10 lines because of zero data after check
        data = np.delete(data, [0,1,2,3,4,5,6,7,8,9], 0)
        bands.append(data)
    pixels = rows * (cols-10)
    stack = np.dstack(bands).reshape(pixels, 8)
    return stack

def readThreeFullImageAsArray(tif1, tif2, tif3):
    rows = tif1.RasterXSize
    cols = tif1.RasterYSize
    bands = []
    for band_number in range(1, 9):
        band = tif1.GetRasterBand(band_number)
        data = band.ReadAsArray(0, 0, rows, cols)
        # delete the top 10 lines because of zero data after check
        data = np.delete(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0)
        bands.append(data)
    for band_number in range(1, 9):
        band = tif2.GetRasterBand(band_number)
        data = band.ReadAsArray(0, 0, rows, cols)
        # delete the top 10 lines because of zero data after check
        data = np.delete(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0)
        bands.append(data)
    for band_number in range(1, 9):
        band = tif3.GetRasterBand(band_number)
        data = band.ReadAsArray(0, 0, rows, cols)
        # delete the top 10 lines because of zero data after check
        data = np.delete(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0)
        bands.append(data)
    pixels = rows * (cols-10)
    stack = np.dstack(bands).reshape(pixels, 24)
    return stack


def get_coordinate_name(dataset):
    """ get a dataset spatial reference name
    :param gdal/ogr dataset:
    :return: the name of the spatial reference
    """
    if isinstance(dataset, osgeo.gdal.Dataset):

        prj=dataset.GetProjection()
        srs=osr.SpatialReference(wkt=prj)
    else:
        layer = dataset.GetLayer()
        srs = layer.GetSpatialRef()

    if srs.IsProjected():
        return srs.GetAttrValue("projcs")
    else: return srs.GetAttrValue("geogcs")


def export_spatialref(dataset):
    """ get the spatial reference in different formats from a GDAL/OGR dataset
    :param dataset: a gdal/ogr raster dataset
    :return: a dictionary
    """
    if isinstance(dataset, osgeo.gdal.Dataset):
        spatialRef = osr.SpatialReference()
        wkt = dataset.GetProjection()
        spatialRef.ImportFromWkt(wkt)
    else:
        layer = dataset.GetLayer()
        spatialRef = layer.GetSpatialRef()

    print(spatialRef)
    out={}
    out["Wkt"] =spatialRef.ExportToWkt()
    out["PrettyWkt"] =spatialRef.ExportToPrettyWkt()
    out["PCI"] =spatialRef.ExportToPCI()
    out["USGS"] =spatialRef.ExportToUSGS()
    out["XML"] =spatialRef.ExportToXML()

    return out


def reproject_vector(inDataSet, epsg_from=None, epsg_to=None):

    """ reproject a vector file (only the first layer!) (it does not save the dataset to disk)
    :param inDataSet: the input ogr dataset
    :param epsg_from: the input spatial reference; in None take the source reference
    :param epsg_to: the output spatial reference
    :return: the reprojected dataset
    """

    if not epsg_to: raise Exception("please, specify the output EPSG codes")

    outDataSet = None
    inFeature = None
    outFeature = None
    outLayer = None

    try:
        #driver = inDataSet.GetDriver()

        # define input SpatialReference
        if not epsg_from:
            layer = inDataSet.GetLayer()
            inSpatialRef = layer.GetSpatialRef()
        else:
            inSpatialRef = osr.SpatialReference()
            inSpatialRef.ImportFromEPSG(epsg_from)

        # define output SpatialReference
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(epsg_to)

        # create the CoordinateTransformation
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

        # get the first input layer and the geometry type
        inLayer = inDataSet.GetLayer()
        geotype = inLayer.GetGeomType()
        lname = inLayer.GetName()

        drv = ogr.GetDriverByName("ESRI Shapefile")
        outDataSet = drv.CreateDataSource( "/vsimem/memory.shp" )

        outLayer = outDataSet.CreateLayer(lname, srs = outSpatialRef, geom_type = geotype)

        # add fields
        inLayerDefn = inLayer.GetLayerDefn()

        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            outLayer.CreateField(fieldDefn)

        # get the output layer"s feature definition
        outLayerDefn = outLayer.GetLayerDefn()


        counter=1

        # loop through the input features
        inFeature = inLayer.GetNextFeature()
        while inFeature:
            # get the input geometry
            geom = inFeature.GetGeometryRef()
            # reproject the geometry
            geom.Transform(coordTrans)
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)

            # destroy the features and get the next input feature
            if outFeature: outFeature = None
            inFeature = inLayer.GetNextFeature()

            counter += 1
            #print(counter)

        return outDataSet


    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        if outDataSet: outDataSet == None #give back control to C++
        if outLayer: outLayer == None
        if inFeature: inFeature == None
        if outFeature: outFeature = None


def reproject_raster(dataset, epsg_from = None, epsg_to=None, fltr=gdal.GRA_NearestNeighbour):
    """reproject a gdal raster dataset
    :param dataset: a gdal dataset
    :param epsg_from: the input epsg; if None get from the sorce
    :param epsg_to: the output epsg; if None throw exception
    :param fltr: the filter to apply when reprojecting
        GRA_NearestNeighbour
        Nearest neighbour (select on one input pixel)
        GRA_Bilinear
        Bilinear (2x2 kernel)
        GRA_Cubic
        Cubic Convolution Approximation (4x4 kernel)
        GRA_CubicSpline
        Cubic B-Spline Approximation (4x4 kernel)
        GRA_Lanczos
        Lanczos windowed sinc interpolation (6x6 kernel)
        GRA_Average
        Average (computes the average of all non-NODATA contributing pixels)
        GRA_Mode
        Mode (selects the value which appears most often of all the sampled points)

    #############NearestNeighbour filter is good for categorical data###########

    :return: the reprojected dataset
    """

    try:

        if epsg_to is None:
            raise Exception("select the destination projected spatial reference!!!")

        if epsg_from == epsg_to:
            print("the input and output projections are the same!")
            return dataset

        # Define input/output spatial references
        if epsg_from:
            source = osr.SpatialReference()
            source.ImportFromEPSG(epsg_from)
            inwkt = source.ExportToWkt()
        else:
            source = osr.SpatialReference()
            source.ImportFromWkt(dataset.GetProjection())
            source.MorphFromESRI()  #this is to avoid reprojection errors
            inwkt = source.ExportToWkt()

        destination = osr.SpatialReference()
        destination.ImportFromEPSG(epsg_to)
        outwkt = destination.ExportToWkt()

        vrt_ds = gdal.AutoCreateWarpedVRT(dataset, inwkt, outwkt, fltr)

        return vrt_ds

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

################################################################################
# this is an old function to reproject rasters, it allows to define the output pixel size
# NOTE: it does not work with geographic spatial references


def reproject_raster_1(dataset, pixel_spacing = None, epsg_from=None, epsg_to=None, fltr=gdal.GRA_NearestNeighbour):
    """ reproject a raster. It does not work with geographic spatial references
    :param dataset: a gdal dataset
    :param pixel_spacing: the output pixel width and pixel height (we assume they are the same), if None use source
    :param epsg_from: the input epsg; if None get from the sorce
    :param epsg_to: the output epsg; if None throw exception
    :param fltr: the filter to apply when reprojecting
                GRA_NearestNeighbour
                Nearest neighbour (select on one input pixel)
                GRA_Bilinear
                Bilinear (2x2 kernel)
                GRA_Cubic
                Cubic Convolution Approximation (4x4 kernel)
                GRA_CubicSpline
                Cubic B-Spline Approximation (4x4 kernel)
                GRA_Lanczos
                Lanczos windowed sinc interpolation (6x6 kernel)
                GRA_Average
                Average (computes the average of all non-NODATA contributing pixels)
                GRA_Mode
                Mode (selects the value which appears most often of all the sampled points)

    :return: the reprojected dataset

    #############NearestNeighbour filter is good for categorical data###########

    """

    mem_drv = None

    try:

        if epsg_to is None:
            raise Exception("select the destination projected spatial reference!!!")

        # Define spatial references and transformation
        source = osr.SpatialReference()
        if epsg_from:
            source.ImportFromEPSG(epsg_from)
        else:
            wkt = dataset.GetProjection()
            source.ImportFromWkt(wkt)
            source.MorphFromESRI()  # this is to avoid reprojection errors

        destination = osr.SpatialReference()
        destination.ImportFromEPSG(epsg_to)

        # print(source.GetAttrValue("projcs"))
        # print(source.GetAttrValue("geogcs"))
        # check we have projected spatial references
        if destination.IsGeographic() or source.IsGeographic():
            raise Exception("geographic spatial reference are not allowed (still...)")

        tx = osr.CoordinateTransformation(source, destination)

        # get the number of bands
        nbands = dataset.RasterCount

        # get the data type from the first band
        go = True
        while go:
            band = dataset.GetRasterBand(1)
            if band is None: continue
            go = False
            btype = band.DataType
            if band is not None: band = None #give back control to C++

        # Get the Geotransform vector
        geo_t = dataset.GetGeoTransform ()
        x_size = dataset.RasterXSize  # Raster xsize
        y_size = dataset.RasterYSize  # Raster ysize

        # Work out the boundaries of the new dataset in the target projection
        # TODO what if the input raster is rotated?
        (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
        (llx, lly, llz) = tx.TransformPoint(geo_t[0], geo_t[3] + geo_t[5]*y_size)

        (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3] + geo_t[5]*y_size )
        (urx, ury, urz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3])

        ulx = min(ulx, llx)
        uly = max(uly, ury)  #TODO check for the southern emisphere

        lrx = max(lrx,urx)
        lry = min(lry,lly)  #TODO check for the southern emisphere


        # set the input spacing if user did not set it
        if pixel_spacing is None:
            pixel_spacing = geo_t[1]

        # Now, we create an in-memory raster
        mem_drv = gdal.GetDriverByName("MEM")
        dest = mem_drv.Create("", int((lrx - ulx)/pixel_spacing), int((uly - lry)/pixel_spacing), nbands, btype)

        # Calculate the new geotransform
        new_geo = ( ulx, pixel_spacing, geo_t[2], uly, geo_t[4], -pixel_spacing) #TODO the - might not work in the southern emisphere

        # Set the geotransform
        dest.SetGeoTransform(new_geo)
        dest.SetProjection(destination.ExportToWkt())

        # Perform the projection/resampling
        res = gdal.ReprojectImage(dataset, dest, source.ExportToWkt(), destination.ExportToWkt(), fltr)

        return dest

    except RuntimeError as err:
        raise err
    except Exception as e:
         raise e

    finally:
        if mem_drv is not None:
            mem_drv == None  # give back control to C++


def save_vector(dataset, outpath):
    """ save an ogr dataset to disk, (it will delete preexisting output)
    :param dataset: ogr dataset
    :param outpath: output path
    :return: None
    """

    try:
        driver = dataset.GetDriver()
        if os.path.exists(outpath):
            driver.DeleteDataSource(outpath)
        dst_ds = driver.CopyDataSource(dataset, outpath)

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        dst_ds = None  # Flush the dataset to disk


def save_raster(dataset, outpath):

    """save an gdal dataset to disk, (it will delete preexisting output)
    :param dataset: gdal dataset
    :param outpath: output path
    :return: None
    """

    try:
        driver = dataset.GetDriver()
        if os.path.exists(outpath):
            driver.Delete(outpath)
        dst_ds = driver.CreateCopy(outpath,dataset)

    except RuntimeError as err:
        raise err
    except Exception as e:
         raise e

    finally:
        dst_ds = None # Flush the dataset to disk


################################################################################

def get_envelope_coordinates(fc, start=0, stop=None, prnt=False):
    """get the envelopes coordinates of an ogr dataset; it's possible to select a range of features
    :param fc:the ogr dataset
    :param start: positive start index; first index is 0 ; last index is fc.featureCount()-1
    :param stop: positive stop index; stop index is not comprised in the result
    :param prnt: if True print the coordinate on screen
    :return: a list of coordinates (by default get all the features)
    """

    raise Exception("not implemented")


def compare_tables(at1,at2):
    """ compare 2 attribute tables
    :param at1: attribute table 1
    :param at2: attribute table 2
    :return: a list of cell differences
    """

    raise Exception("not implemented")


def get_indexes(l,value):
    """ function that returns a list of indices for a given simple value in a list l
    :param l: a list
    :param value: a simple value (character or number)
    :return: the list of indices
    """

    m = l.copy()  # create a copy of the list
    idx = []
    try:
        while True:
            ind = m.index(value)
            idx.append(ind)
            m[ind] = value/2  # create a fake value to continue the iteration
    except:  # this happens when the value is not found
            return idx
######################################################################


def get_shp_geom(fn):
    """ OGR layer object or filename to datasource (will use 1st layer)
    :param fn:
    :return:
    """

    lyr, ds = _get_layer(fn)
    feat = lyr.GetNextFeature()
    return feat.geometry().Clone()


def has_spatialite():
    """Determine if the current GDAL is built with SpatiaLite support."""
    use_exceptions = ogr.GetUseExceptions()
    ogr.UseExceptions()
    try:
        ds = ogr.GetDriverByName("Memory").CreateDataSource("memory")
        sql = '''SELECT sqlite_version(), spatialite_version()'''
        lyr = ds.ExecuteSQL(sql, dialect="SQLite")
        return True
    except Exception as e:
        return False
    finally:
        if not use_exceptions:
            ogr.DontUseExceptions()


def print_attributes(lyr_or_fn, n=None, fields=None, geom=True, reset=True):
    """ Print attribute values in a layer.
    :param lyr_or_fn: OGR layer object or filename to datasource (will use 1st layer)
    :param n: optional number of features to print; default is all
    :param fields: optional list of case-sensitive field names to print; default is all
    :param geom: optional boolean flag denoting whether geometry type is printed; default is True
    :param reset: optional boolean flag denoting whether the layer should be reset to the first record before printing; default is True
    :return:
    """

    lyr, ds = _get_layer(lyr_or_fn)
    if reset:
        lyr.ResetReading()

    n = n or lyr.GetFeatureCount()
    geom = geom and lyr.GetGeomType() != ogr.wkbNone
    fields = fields or [field.name for field in lyr.schema]
    data = [["FID"] + fields]
    if geom:
        data[0].insert(1, "Geometry")
    feat = lyr.GetNextFeature()
    while feat and len(data) <= n:
        data.append(_get_atts(feat, fields, geom))
        feat = lyr.GetNextFeature()
    lens = map(lambda i: max(map(lambda j: len(str(j)), i)), zip(*data))
    format_str = "".join(map(lambda x: "{{:<{}}}".format(x + 4), lens))
    for row in data:
        try:
            print(format_str.format(*row))
        except UnicodeEncodeError:
            e = sys.stdout.encoding
            print(codecs.decode(format_str.format(*row).encode(e, "replace"), e))
    print("{0} of {1} features".format(min(n, lyr.GetFeatureCount()), lyr.GetFeatureCount()))
    if reset:
        lyr.ResetReading()


def print_capabilities(item):
    """Print capabilities for a driver, datasource, or layer."""
    if isinstance(item, ogr.Driver):
        _print_capabilites(item, "Driver", "ODrC")
    elif isinstance(item, ogr.DataSource):
        _print_capabilites(item, "DataSource", "ODsC")
    elif isinstance(item, ogr.Layer):
        _print_capabilites(item, "Layer", "OLC")
    else:
        print("Unsupported item")


def print_drivers():
    """Print a list of available drivers."""
    for i in range(ogr.GetDriverCount()):
        driver = ogr.GetDriver(i)
        writeable = driver.TestCapability(ogr.ODrCCreateDataSource)
        print("{0} ({1})".format(driver.GetName(),
                                 "read/write" if writeable else "readonly"))


def print_layers(fn):
    """ Print a list of layers in a data source.
    :param fn: path to data source
    :return:
    """

    ds = ogr.Open(fn, 0)
    if ds is None:
        raise OSError("Could not open {}".format(fn))
    for i in range(ds.GetLayerCount()):
        lyr = ds.GetLayer(i)
        print("{0}: {1} ({2})".format(i, lyr.GetName(), _geom_constants[lyr.GetGeomType()]))


def _geom_str(geom):
    """ Get a geometry string for printing attributes.
    :param geom:  gdal geometry
    :return: geometry name
    """
    if geom.GetGeometryType() == ogr.wkbPoint:
        return "POINT ({:.3f}, {:.3f})".format(geom.GetX(), geom.GetY())
    else:
        return geom.GetGeometryName()


def _get_atts(feature, fields, geom):
    """Get attribute values from a feature.
    :param feature: input feature
    :param fields: which fields you want to get?
    :param geom: do you want the geometry?
    :return: a list with attributes
    """
    data = [feature.GetFID()]
    geometry = feature.geometry()
    if geom and geometry:
        data.append(_geom_str(geometry))
    values = feature.items()
    data += [values[field] for field in fields]
    return data


def _get_layer(lyr_or_fn):
    """ Get the datasource and layer from a filename.
    :param lyr_or_fn: filename
    :return: layer and dataset
    """

    if type(lyr_or_fn) is str:
        ds = ogr.Open(lyr_or_fn)
        if ds is None:
            raise OSError("Could not open {0}.".format(lyr_or_fn))
        return ds.GetLayer(), ds
    else:
        return lyr_or_fn, None


def _print_capabilites(item, name, prefix):
    """ Print capabilities for a driver, datasource, or layer.
    :param item: item to test
    :param name: name of the type of item
    :param prefix: prefix of the ogr constants to use for testing
    :return: None
    """

    print("*** {0} Capabilities ***".format(name))
    for c in filter(lambda x: x.startswith(prefix), dir(ogr)):
        print("{0}: {1}".format(c, item.TestCapability(ogr.__dict__[c])))


def shuffle_data(data, percentage, where):
    """ shuffle a 2d numpy array and split in training and validation

    :param data: 2d snumpy array with all the data (samples, labels)
    :param percentage: percentage of validation data(integer)
    :return: trainingSamples,trainingLabels,validationSamples,validationLabels, number of training data
    """

    count = data.shape[0]  #nuber of rows
    r = np.array(list(range(0, count))).reshape(-1, 1)
    plusone = np.hstack((r, data))

    # shuffle all the data randomly
    np.random.shuffle(plusone)

    # get number of training data
    k = int(math.ceil(count*((100-percentage)/100.0)))

    # set training data

    trainingSamples = plusone[:k, 1:-1]
    trainingLabels = plusone[:k, -1:]

    # set validation data
    validationSamples= plusone[k:, 1:-1]
    validationLabels = plusone[k:, -1:]

    shuf = plusone[:,0]
    zipped = list(zip(r.tolist(), shuf.tolist()))

    return trainingSamples,trainingLabels,validationSamples,validationLabels, k, zipped


def combination_count(nbands = 8):
    """ get possible band combinations ( where band1-band2 is different from band2- band1)
    :param nbands: number of bands
    :return: a tuple: the total number of band combinations, a list of tuples where each tuple is the band combination
    """

    n = nbands  # number of bands (get this from the numpy array)
    i = 1
    # sum = 0
    k = n + 1
    names1 = []
    names2 = []
    while i < n:

        # sum += (n - i)

        # left-right band combination names
        j = i + 1
        while j <= n:
            # names1.append(str(i)+"-"+str(j)+"\t")
            names1.append((i, j))
            j += 1

        # right - left band combination names
        j = n - i
        k -= 1
        while j >= 1:
            # names2.append(str(k) +"-"+ str(j) + "\t")
            names2.append((k, j))
            j -= 1
        i += 1

    # sum *=2
    names = names1+names2

    # return sum, names1+names2
    return len(names), names


def column_names_to_string(t,sep1="-", sep2="\t"):
    """
    :param t: a list of tuples, each tuple with a combination e.g [(1,2),(1,3),...]
    :param sep1: the separator within a couple of names
    :param sep2: the separator between names
    :return: a list of strings
    """
    text = []
    for i in t:
        text.append(str(i[0]) + sep1 + str(i[1]) + sep2)
    return text


def convert_raster(imgpath=None, outpath=None, outdatatype="uint8", rescaletype="linear", exepath="otbcli_Convert.bat"):
    """ convert a raster to a new pixel format
    :param imgpath: input raster
    :param outpath: output raster
    :param outdatatype: output data type
    :param rescaletype: conversion type
    :param exepath: the path to conversion utility (change if the system path variable is not set)
    :return: messages
    """
    import subprocess
    params = [exepath, "-in", imgpath, "-out", outpath, outdatatype, "-type", rescaletype]
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # A pipe is a section of shared memory that processes use for communication.
    out, err = p.communicate()
    return bytes.decode(out),bytes.decode(err)


def run_tool(params):
    """ run an executable tool (exe, bat,..)
    :param params: list of string parameters  ["tool path", "parameter1", "parameter2",.... ]
    :return: messages
    """
    import subprocess
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # A pipe is a section of shared memory that processes use for communication.
    out, err = p.communicate()
    return bytes.decode(out), bytes.decode(err)



def run_script(params, callpy= ["py","-3.5"]):
    """ execute a python script
    :param params: a list of strings [ 'python version' , 'parameters']
    :param callpy: how to start the python interpreter
    :return: script output
    """
    import subprocess
    #params.insert(0,callpy)
    params = callpy + params
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print(params)
    out, err= p.communicate()
    #return bytes.decode(out)+'\n'+ bytes.decode(err)
    return bytes.decode(out), bytes.decode(err)



def get_minmax(imagepath, band = 1 ):
    """ get the minimum and maximum of a raster band
    :param   input raster path
    :param   band number to analyse (starts at 1)
    """
    d = None
    try:
        d = gdal.Open(imagepath)
        band = d.GetRasterBand(band)
        band.ComputeStatistics(False)
        mn = band.GetMinimum()
        mx = band.GetMaximum()
        return mn, mx
    except Exception as e:
        print(e)
    finally:
        if d:
            d = None


def filter_files(basepath, filter):
    """ filter files by extension name and return of list of names
    :param path: tha path to the folder
    :param filter: a list with the file extensions   e.g. ['.tif']
    :return: a list of file names
    """
    # get the content of the images directory
    f = os.listdir(basepath)
    a = []
    # iterate all the files and keep only the ones with the correct extension
    for i in f:
        # filter content, we want files with the correct extension
        if os.path.isfile(basepath+'/'+i) and (os.path.splitext(basepath+'/'+i)[-1] in filter):
            a.append(i)
    return a


_geom_constants = {}
_ignore = ["wkb25DBit", "wkb25Bit", "wkbXDR", "wkbNDR"]
for c in filter(lambda x: x.startswith("wkb"), dir(ogr)):
    if c not in _ignore:
        _geom_constants[ogr.__dict__[c]] = c[3:]


if __name__ == "__main__":


    ### tests

    import os
    os.chdir(r"C:\Users\TianMG\Documents\itc\M13\project\STARS_Sentinel_Mali")
    #test projections
    etrs1989 = 3035
    rdnew = 28992
    wgs84 = 4326

    gdal.UseExceptions()  # allow gdal exceptions

    def test_reproject_raster():

        ndviname = "054112895010_20.tif"

        inp = None
        out = None
        try:
            # create dataset
            inp = gdal.Open(ndviname)

            # reproject
            print("reprojecting to etrs1989...")
            out = reproject_raster(inp,epsg_from = None, epsg_to = etrs1989, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_3035.tif", out)

            # reproject
            print("reprojecting to etrs1989...")
            out = reproject_raster_1(inp, pixel_spacing = None, epsg_from=None, epsg_to = etrs1989, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_3035_1.tif", out)

            # reproject
            print("reprojecting to 4326...")
            out = reproject_raster(inp,epsg_from = None, epsg_to = 4326, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_4326.tif", out)

            # reproject back
            print("reprojecting from 4326 to rdnew...")

            inp = gdal.Open(ndviname+"_4326.tif")
            out = reproject_raster(inp,epsg_from = None, epsg_to=28992, fltr = gdal.GRA_NearestNeighbour)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_28992.tif", out)


        except RuntimeError as err:
            raise err
        except Exception as e:
             raise e

        finally:
            # close datasets
            if inp:
                inp = None
            if out:
                out = None

    test_reproject_raster()