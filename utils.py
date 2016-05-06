
# coding: utf-8

# In[2]:

import os
from shutil import copyfile

from osgeo import gdal, ogr

# Settings
TEST_TO_TRAIN_CLASS_NAME = {
    'MZ': 'MAIZ',
    'MN': 'MANI',
    'PN': 'PN',
    'SJ': 'SOJA',
    'SRG': 'SORGO',
}

CLASS_NAME_TO_INT = {
    'ALFA': 1,
    'MAIZ': 2,
    'MANI': 3,
    'MONTE': 4,
    'PN': 5,
    'RASTROJO': 6,
    'SOJA': 7,
    'SORGO': 8
}

INT_TO_CLASS_NAME = {
    '1': 'ALFA',
    '2': 'MAIZ',
    '3': 'MANI',
    '4': 'MONTE',
    '5': 'PN',
    '6': 'RASTROJO',
    '7': 'SOJA',
    '8': 'SORGO'
}

# In[15]:

# Copy new files into each folder
def separate_files_by_date():
    dates = ['150201', '150217', '150321']
    files_source = 'real_data/split/'
    files_dest = 'real_data/split/split_by_date/%s/train/'
    for date in dates:
        print("copy files from %s to %s" % (files_source, files_dest % date))
        for f in os.listdir(files_source):
            fname = f.split(".")[0]
            if fname.endswith(date) and not fname.startswith('ROI'):
                copyfile(os.path.join(files_source, f), os.path.join(files_dest % date, f))
        unify_files_names(files_dest % date)


# In[6]:

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


# In[7]:

def test_class_name_to_train_label(class_name):
    """Convert the test class name to the label assigned to the pixels during training."""
    # training_reference must be reference dict generated during the training process. 
    return CLASS_NAME_TO_INT[TEST_TO_TRAIN_CLASS_NAME[class_name]]

def train_label_to_test_class_name(training_reference):
    """Convert the label assigned to the pixels during training to the test class name."""
    return INT_TO_CLASS_NAME[training_reference]

# In[8]:
def extract_test_mask(vector_data_path, rows, cols, geo_transform, projection, attr):
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
    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=%s" % attr])
    return target_ds


# In[9]:

def add_label_from_reference_name(vector_data_path, for_lot_id=False):
    """
    Add a new attribute to the given vector file's geometries, converting the test class name in
    ROI_E_14_1 to the corresponding training class label.
    
    NOTE: The given vector file is modified.
    
    """
    ds = gdal.OpenEx(vector_data_path, gdal.OF_UPDATE)
    if ds is None:
        print("Open failed.")
        sys.exit(1)
    lyr = ds.GetLayer()

    lyr.ResetReading()
    lyr_defn = lyr.GetLayerDefn()
    
    new_field_defn = ogr.FieldDefn("reference", ogr.OFTInteger)

    if lyr.CreateField ( new_field_defn ) != 0:
        raise Error("Creating reference_label field failed.")
    
    field_roi_e_14_1 = lyr_defn.GetFieldIndex('ROI_E_14_1')
    field_reference_label = lyr_defn.GetFieldIndex('reference')
    for feat in lyr:
        if not for_lot_id:
            field_roi = feat.GetField(field_roi_e_14_1)
            field_reference = feat.GetField(field_reference_label)
            reference_label = test_class_name_to_train_label(field_roi)
        else:
            # Id is zero indexed, but category starts from 1.
            reference_label = feat.GetFID() + 1 
        
        feat.SetField(field_reference_label, reference_label)
        lyr.SetFeature(feat)
    ds = None


# In[11]:

def unify_files_names(folder_path):
    for f in os.listdir(folder_path):
        if f.startswith('SJ'):
            print("Rename: %s to %s" % (f, f.replace('SJ', 'SOJA')))
            os.rename(os.path.join(folder_path, f), os.path.join(folder_path, f.replace('SJ', 'SOJA')))
        elif f.startswith('MN'):
            print("Rename: %s to %s" % (f, f.replace('MN', 'MANI')))
            os.rename(os.path.join(folder_path, f), os.path.join(folder_path, f.replace('MN', 'MANI')))


# In[16]:

def get_value_from_class_file(fname):
    print('File: %s' % fname)
    aux = fname.split('/')
    fname = aux[len(aux) - 1]
    print('File 1: %s' % fname)
    return CLASS_NAME_TO_INT[fname.split('_')[0]]