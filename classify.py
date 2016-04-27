"""
Classify a multi-band, satellite image.

Usage:
    classify.py <input_fname> <train_data_path> <output_fname> [--method=<classification_method>]
                                                               [--validation=<validation_data_path>]
                                                               [--verbose]
    classify.py -h | --help

The <input_fname> argument must be the path to a GeoTIFF image.

The <train_data_path> argument must be a path to a directory with vector data files
(in shapefile format). These vectors must specify the target class of the training pixels. One file
per class. The base filename (without extension) is taken as class name.

If a <validation_data_path> is given, then the validation vector files must correspond by name with
the training data. That is, if there is a training file train_data_path/A.shp then the corresponding
validation_data_path/A.shp is expected.

The <output_fname> argument must be a filename where the classification will be saved (GeoTIFF format).

No geographic transformation is performed on the data. The raster and vector data geographic
parameters must match.

Options:
  -h --help  Show this screen.
  --method=<classification_method>      Classification method to use: random-forest (for random
                                        forest), svm (for support vector machines) or
                                        xgb (for xgboost (eXtreme Gradient Boosting))
                                        [default: random-forest]
  --validation=<validation_data_path>   If given, it must be a path to a directory with vector data
                                        files (in shapefile format). These vectors must specify the
                                        target class of the validation pixels. A classification
                                        accuracy report is writen to stdout.
  --verbose                             If given, debug output is writen to stdout.

"""
import logging
import numpy as np
import os
import pickle
import sys

import xgboost as xgb

from docopt import docopt
from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


logger = logging.getLogger(__name__)

# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection,
                            target_value=1, output_fname='', dataset_format='MEM'):
    """
    Rasterize the given vector (wrapper for gdal.RasterizeLayer). Return a gdal.Dataset.

    :param vector_data_path: Path to a shapefile
    :param cols: Number of columns of the result
    :param rows: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    :param target_value: Pixel value for the pixels. Must be a valid gdal.GDT_UInt16 value.
    :param output_fname: If the dataset_format is GeoTIFF, this is the output file name
    :param dataset_format: The gdal.Dataset driver name. [default: MEM]

    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """
    Rasterize, in a single image, all the vectors in the given directory.

    The data of each file will be assigned the same pixel value. This value is defined by the order
    of the file in file_paths, starting with 1: so the points/poligons/etc in the same file will be
    marked as 1, those in the second file will be 2, and so on.

    :param file_paths: Path to a directory with shapefiles
    :param rows: Number of rows of the result
    :param cols: Number of columns of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)

    """
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        logger.debug("Processing file %s: label (pixel value) %i", path, label)
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        aux = band.ReadAsArray()
        logger.debug("Labeled pixels: %i", len(aux.nonzero()[0]))
        labeled_pixels += aux
        ds = None
    return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.

    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)

    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    ct = gdal.ColorTable()

    for pixel_value in range(len(classes) + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    band.WriteArray(data)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):
    logger.error(txt, *args, **kwargs)
    sys.exit(1)


def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()


# ###############################################
# Starts testing code
# ###############################################
TRANSLATE_DICT = {
    'SJ': 1,
    'MZ': 2,
    'SRG': 3,
    'PN': 4,
    'MN': 5,
}


def test_method(vector_data_path, geo_transform, projection, target_value):
    """
    Rasterize our modified vector.
    """
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=%s" % target_value])
    return target_ds


def predict_by_chunks(data, classifier, rows, cols):
    """
    Classify data by chunks.

    :param data:        An array of pixels.
    :param classifier:  A trained classifier.
    :param rows:        Number of rows.
    :param cols:        Number of cols.
    :return:            An array with classified data.
    """

    result = np.array([])
    chunk_size = 100  # Number of rows
    chunk_nr = 0
    while chunk_nr * chunk_size < rows:
        start = chunk_nr * cols * chunk_size
        end = start + cols * chunk_size
        chunk = flat_pixels[start: end]
        predicted_chunk = classifier.predict(chunk)
        result = np.concatenate([result, predicted_chunk])
        chunk_nr += 1
    return result


def divide_test_and_training(dataset, for_training):
    """
    Split in a randon way a dataset in two, one for test and other for training.

    :param dataset: A numpy array.
    :for_training:  An integer representating the percentage of data training required.
    :return:        test array and trainign array.
    """
    rows, cols = dataset.shape
    sample = rows * cols

    zeros = int((sample * for_training) / 100)
    ones = sample - zeros

    mask = np.array([0] * zeros + [1] * ones)

    # Randomize
    np.random.shuffle(mask)
    mask = mask.reshape(dataset.shape)

    training = dataset * mask
    # if mask = [0, 1, 0] ==> ((mask - 1) * (-1)) = [1, 0, 1]
    test = dataset * ((mask - 1) * (-1))

    return test, training


def delete_extra_fields_from_vector(vector_data_path, field):
    """
    Delete useless "fields" of a vector.

    :param vector_data_path:    Path to vector file.
    :fiedls:                    List of fields to delete.

    Beware here, this method modifies the file.
    """

    # Open for update
    ds = gdal.OpenEx(vector_data_path, gdal.OF_UPDATE)
    if ds is None:
        print("Open failed.")
        sys.exit(1)

    lyr = ds.GetLayer()
    lyr_defn = lyr.GetLayerDefn()

    # Get field by name
    for field in fiedls:
        fld = lyr_defn.GetFieldIndex(field)
        lyr.DeleteField(fld)

    # Close vector
    ds = None


def translate_strings_to_int(vector_data_path, field):
    ds = gdal.OpenEx(vector_data_path, gdal.OF_UPDATE)
    if ds is None:
        print("Open failed.")
        sys.exit(1)
    lyr = ds.GetLayer()

    lyr.ResetReading()
    lyr_defn = lyr.GetLayerDefn()
    fld = lyr_defn.GetFieldIndex(field)
    for feat in lyr:
        field_key = feat.GetField(fld)
        feat.SetField(fld, TRANSLATE_DICT[field_key])
        lyr.SetFeature(feat)
    ds = None
# ###############################################
# End testing code
# ###############################################


if __name__ == "__main__":
    opts = docopt(__doc__)

    raster_data_path = opts["<input_fname>"]
    train_data_path = opts["<train_data_path>"]
    output_fname = opts["<output_fname>"]
    validation_data_path = opts['--validation'] if opts['--validation'] else None
    log_level = logging.DEBUG if opts["--verbose"] else logging.INFO
    method = opts["--method"]

    logging.basicConfig(level=log_level, format='%(asctime)-15s\t %(message)s')
    gdal.UseExceptions()

    logger.debug("Reading the input: %s", raster_data_path)
    try:
        raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    except RuntimeError as e:
        report_and_exit(str(e))

    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    # A sample is a vector with all the bands data. Each pixel (independent of
    # its position) is a sample.
    n_samples = rows * cols

    logger.debug("Process the training data")
    try:
        files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]
    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))

    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

    # If you want divide, uncomment next line and comment [1] and [2]
    # verification_pixels, training_data = divide_test_and_training(labeled_pixels, 25)
    # is_train = np.nonzero(training_data)

    # [1]
    is_train = np.nonzero(labeled_pixels)

    training_labels = labeled_pixels[is_train]
    training_samples = bands_data[is_train]

    flat_pixels = bands_data.reshape((n_samples, n_bands))

    #
    # Perform classification
    #

    # for xgb
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['silent'] = 1
    param['nthread'] = 3
    param['num_class'] = 9
    num_round = 5
    CLASSIFIERS = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random-forest': RandomForestClassifier(n_jobs=4, n_estimators=10, class_weight='balanced'),
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        'svm': SVC(class_weight='balanced'),
        # https://github.com/dmlc/xgboost
        'xgb': xgb.Booster(params=param)
    }

    classifier = CLASSIFIERS[method]
    logger.debug("Train the classifier: %s", str(classifier))

    logger.debug("Train the classifier...")
    if method == 'xgb':
        dtrain = xgb.DMatrix(training_samples, label=training_labels)
        booster = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round)
    else:
        classifier.fit(training_samples, training_labels)

    logger.debug("Classifing...")
    if method == 'xgb':
        dpredict = xgb.DMatrix(flat_pixels)
        result = booster.predict(dpredict)
    else:
        result = predict_by_chunks(flat_pixels, classifier, rows, cols)

    # Reshape the result: split the labeled pixels into rows to create an image
    classification = result.reshape((rows, cols))
    write_geotiff(output_fname, classification, geo_transform, proj)
    logger.info("Classification created: %s", output_fname)

    #
    # Validate the results
    #
    if validation_data_path:
        logger.debug("Process the verification (testing) data")
        try:
            shapefiles = [os.path.join(validation_data_path, "%s.shp" % c) for c in classes]
        except OSError.FileNotFoundError as e:
            report_and_exit(str(e))

        # [2]
        verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

        for_verification = np.nonzero(verification_pixels)
        verification_labels = verification_pixels[for_verification]
        predicted_labels = classification[for_verification]

        logger.info("Confussion matrix:\n")
        short_names = [(x.split('_')[0])[:3] for x in clasess]
        print_cm(metrics.confusion_matrix(verification_labels, predicted_labels), short_names)
        target_names = ['Class %s' % s for s in classes]
        logger.info("Classification report:\n%s",
                    metrics.classification_report(verification_labels, predicted_labels,
                                                  target_names=target_names))
        logger.info("Classification accuracy: %f",
                    metrics.accuracy_score(verification_labels, predicted_labels))
