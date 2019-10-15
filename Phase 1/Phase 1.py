# standard packages used to handle files
import sys
import os 
import glob
import time
# commonly used library for data manipilation
import pandas as pd
# numerical
import numpy as np
# handle images - opencv
import cv2
# machine learning library
import sklearn
import sklearn.preprocessing
#used to serialize python objects to disk and load them back to memory
import pickle
#plotting
import matplotlib.pyplot as plt
# helper functions kindly provided for you by Matthias 
import helpers
# specific helper functions for feature extraction
import features

# filepath constants
DATA_BASE_PATH = 'E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data'
OUTPUT_PATH= 'E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data\\Output'

DATA_TRAIN_PATH = os.path.join(DATA_BASE_PATH,'train')
print(DATA_TRAIN_PATH)
DATA_TEST_PATH = os.path.join(DATA_BASE_PATH,'test')

FEATURE_BASE_PATH = os.path.join(OUTPUT_PATH,'features')
print(FEATURE_BASE_PATH)
FEATURE_TRAIN_PATH = os.path.join(FEATURE_BASE_PATH,'train')
FEATURE_TEST_PATH = os.path.join(FEATURE_BASE_PATH,'test')

PREDICTION_PATH = os.path.join(OUTPUT_PATH,'predictions')

# filepatterns to write out features
FILEPATTERN_DESCRIPTOR_TRAIN = os.path.join(FEATURE_TRAIN_PATH,'train_features_{}.pkl')
FILEPATTERN_DESCRIPTOR_TEST = os.path.join(FEATURE_TEST_PATH,'test_features_{}.pkl')

# create paths in case they don't exist:
helpers.createPath(FEATURE_BASE_PATH)
helpers.createPath(FEATURE_TRAIN_PATH)
helpers.createPath(FEATURE_TEST_PATH)
helpers.createPath(PREDICTION_PATH)

"""-------------------------------------------------------------------------"""

folder_paths=glob.glob(os.path.join(DATA_TRAIN_PATH,'*'))
label_strings = np.sort(np.array([os.path.basename(path) for path in folder_paths]))
num_classes = label_strings.shape[0]
print(label_strings)

"""-------------------------------------------------------------------------"""

train_paths = dict((label_string, helpers.getImgPaths(os.path.join(DATA_TRAIN_PATH,label_string))) for label_string in label_strings)
"""=paths to all training images"""
test_paths = helpers.getImgPaths(DATA_TEST_PATH)

"""-------------------------------------------------------------------------"""

#load first image of class bobcat using opencv:
image=cv2.imread(train_paths['bobcat'][0])

# images can be plotted using matplotlib, but need to be converted from BGR to RGB
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()

"""-------------------------------------------------------------------------"""

# blue green and red channels are aligned along the third dimension of the returned numpy array
print('Image shape: {}'.format(image.shape))

# compute aspect ratio of image 
(height, width, nr_channels) = image.shape

aspect_ratio= width/float(height)

fig_height = 10

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(fig_height * aspect_ratio, fig_height))
[ax.get_xaxis().set_visible(False) for ax in axes]
[ax.get_yaxis().set_visible(False) for ax in axes]
ax0, ax1, ax2 = axes

# extract blue green and red channels from image
blue_channel = image[:,:,0]
green_channel = image[:,:,1]
red_channel = image[:,:,2]

ax0.imshow(np.dstack([red_channel,np.zeros_like(blue_channel),np.zeros_like(blue_channel)]))
ax1.imshow(np.dstack([np.zeros_like(green_channel),np.zeros_like(green_channel),green_channel]))
ax2.imshow(np.dstack([np.zeros_like(red_channel),blue_channel,np.zeros_like(red_channel)]))

plt.show()

"""-------------------------------------------------------------------------"""
"""-------------------------Looking at the data-----------------------------"""
"""-------------------------------------------------------------------------"""
fig, axes = plt.subplots(nrows=num_classes, ncols=3, constrained_layout=True, figsize=(20,20))

[ax.get_xaxis().set_visible(False) for ax_row in axes for ax in ax_row]
[ax.get_yaxis().set_visible(False) for ax_row in axes for ax in ax_row]

for (idx,label_string) in enumerate(label_strings):
    images=[cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB) for path in train_paths[label_string][:3]]

    for colidx in range(3):
        axes[idx,colidx].imshow(images[colidx])
        if colidx == 1: # if this is the center column
            axes[idx,colidx].set_title(label_string)

plt.show() 

"""-------------------------------------------------------------------------"""
"""--------------------Bag Of Visual Words Model----------------------------"""
"""-------------------------------------------------------------------------"""

"""Detecting points of interest"""

for label_string in label_strings:
    
    current_image=cv2.imread(train_paths[label_string][0])
    corner_image, corner_coords=features.extractShiTomasiCorners(current_image,num_features=500, min_distance=5, visualize=True)    
    
    plt.imshow(cv2.cvtColor(corner_image,cv2.COLOR_BGR2RGB))
    plt.show()

"""Good representations of interesting patches"""

# add all features which you would like computed and their callbacks to this dictionary
# features where a pickle file already exists will not be recomputed
descriptor_dict={'daisy':features.extractDAISYCallback, # SIFT replacement, very fast, can be computed dense if necessary
                 'orb':features.extractORBCallback, # another fast SIFT replacement, oriented BRIEF w. FAST keypoints  
                 'freak':features.extractFREAKCallback, # biologically motivated descriptor
                 'lucid':features.extractLUCIDCallback,  
                 'vgg':features.extractVGGCallback, # Trained as proposed by VGG lab, don't confuse it with VGG-Net features
                 'boost_desc':features.extractBoostDescCallback} # Image descriptor learned with boosting
                 
if features.checkForSIFT():
    descriptor_dict['sift'] = features.extractSIFTCallback # One descriptor to rule them all, unfortunately patented

"""-------------------------------------------------------------------------"""

train_descriptor_dict = descriptor_dict.copy()

# if the corresponding files already exist, do not extract them again
train_descriptor_dict = dict((key,value) for (key,value) in descriptor_dict.items() if not os.path.isfile(FILEPATTERN_DESCRIPTOR_TRAIN.format(key)))

if len(train_descriptor_dict) > 0: 

    train_features = []
    train_labels = []
    
    # convert train images
    train_features_by_descriptor = dict((key,[]) for (key,value) in train_descriptor_dict.items())
    
    for label_string in label_strings:
        print('extracting train features for class {} :'.format(label_string))

        extracted_features = features.extractFeatures(train_paths[label_string],train_descriptor_dict, label_string)

        # append descriptors of corresponding label to correct descriptor list 
        for key in train_features_by_descriptor.keys():
            train_features_by_descriptor[key]+=extracted_features[key]
  
    for descriptor_key in train_features_by_descriptor.keys():
        with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_key),'wb') as pkl_file_train:
            pickle.dump(train_features_by_descriptor[descriptor_key], pkl_file_train, protocol=pickle.HIGHEST_PROTOCOL)
    
"""-------------------------------------------------------------------------"""

test_paths = helpers.getImgPaths(DATA_TEST_PATH)

test_descriptor_dict=dict((key,value) for (key,value) in descriptor_dict.items() if not os.path.isfile(FILEPATTERN_DESCRIPTOR_TEST.format(key)))

if len(test_descriptor_dict) > 0: 
    test_features = []
    
    print('extracting test features:') 
    
    test_features_by_descriptor = features.extractFeatures(test_paths,test_descriptor_dict, None) 
    
    for descriptor_key in test_features_by_descriptor.keys():
        with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_key),'wb') as pkl_file_test:
            pickle.dump(test_features_by_descriptor[descriptor_key], pkl_file_test, protocol=pickle.HIGHEST_PROTOCOL)

"""-------------------------------------------------------------------------"""

descriptor_desired='freak'
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    train_features_from_pkl=pickle.load(pkl_file_train)

"""-------------------------------------------------------------------------"""

with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desired),'rb') as pkl_file_test:
    test_features_from_pkl = pickle.load(pkl_file_test)
    
"""-------------------------------------------------------------------------"""
"""-------------------Constructing a BOVW codebook--------------------------"""
"""-------------------------------------------------------------------------""" 
# learn the codebook for the 'freak' features from the training data
codebook_size = 500
clustered_codebook = helpers.createCodebook(train_features_from_pkl, codebook_size = codebook_size)

# encode all train images 
train_data=[]
train_labels=[]

for image_features in train_features_from_pkl:
    bow_feature_vector = helpers.encodeImage(image_features.data,clustered_codebook)
    train_data.append(bow_feature_vector)
    train_labels.append(image_features.label)

plt.plot(train_labels, train_data)
plt.show()

# encode all test images 
test_data=[]
for image_features in test_features_from_pkl:
    bow_feature_vector=helpers.encodeImage(image_features.data,clustered_codebook)
    test_data.append(bow_feature_vector)

# use a labelencoder to obtain numerical labels
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labels[:10])
train_labels = label_encoder.transform(train_labels)
print(train_labels[:10])

"""-------------------------------------------------------------------------"""
"""-----------------------Making a submission-------------------------------"""
"""-------------------------------------------------------------------------"""
# Compute a naive prediction
predictions_uniform = np.ones((len(test_paths),num_classes))/num_classes
# Build a submission
pred_file_path = os.path.join(PREDICTION_PATH, helpers.generateUniqueFilename('uniform','csv'))
helpers.writePredictionsToCsv(predictions_uniform,pred_file_path,label_strings)

# Compute class priors
labels_oh = helpers.toOneHot(train_labels, min_int=np.min(train_labels), max_int=np.max(train_labels))
class_probabilities = labels_oh.mean(axis=0)
predictions_average = [class_probabilities] * len(test_paths)

# Build a submission again:
pred_file_path = os.path.join(PREDICTION_PATH, helpers.generateUniqueFilename('average','csv'))
helpers.writePredictionsToCsv(predictions_average,pred_file_path,label_strings)


"""-------------------------------------------------------------------------"""
"""---------------------1. Problem/Data Analysis----------------------------"""
"""-------------------------------------------------------------------------"""
def getAmountOfPhotosPerClass(labels, paths):
    amountOfPicturesPerClass = {}
    for label in labels:
        amountOfPicturesPerClass[label] = len(paths[label])
    
    return amountOfPicturesPerClass

#calculate average number of images with labels a list of labels you want
def avgAmountOfImagesForCertainClasses(labels, paths):
    amountOfPicturesPerClass = {}
    for label in labels:
        amountOfPicturesPerClass[label] = len(paths[label])
    
    totalphotos = 0
    for label in labels:
        totalphotos = totalphotos + amountOfPicturesPerClass[label]
    
    avgOfAllClasses = (totalphotos/len(labels))
    
    return avgOfAllClasses
    
def showAmountOfPicturesAvailalbeForTrainingPerClass(labels,paths):
    d = getAmountOfPhotosPerClass(labels,paths)
    
    height = []
    for i in d.values():
        height.append(i)
    
    bars = []
    for i in d.keys():
        bars.append(i)
    
    y_pos = np.arange(len(bars))
    # Create horizontal bars
    plt.barh(y_pos, height)
    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    plt.title("Amount of pictures availalbe for training per class", **titlepar)
    yLabelPar = {'rotation' : '0',
                 'weight' : 'bold'}
    yLabel = plt.ylabel("Class", **yLabelPar)
    plt.xlabel("Amount of pictures").set_weight('bold')
    for i, v in enumerate(height):
        plt.text(v + 2.5, i - 0.2, str(v), color='black',  fontweight='bold')
    # Show graphic
    plt.show()

showAmountOfPicturesAvailalbeForTrainingPerClass(label_strings, train_paths)

"""We can clearly see that there are huge differences in amount of training data available per class, we should take this into account"
        --> huge class imabalance"""

def getImageShape(labels, paths):
    shapes = []
    for label in labels:
        for path in paths[label]:
            image  = cv2.imread(path)
            shapes.append(image.shape)
    return shapes

"""def getImageSize(labels, paths):
    sizes = []
    for label in labels:
        for path in paths[label]:
            image  = cv2.imread(path)
            sizes.append(image.size)
    return sizes

getImageSize(["wolf"], train_paths)
"""

def getAmountOfPixels(labels, paths):
    shapes = getImageShape(labels, paths)
    pixels=[]
    for shape in shapes:
        pixels.append(shape[0]*shape[1])
    return pixels

def showAmountOfPicturesPerPixelSizeCategory(labels,paths):
    #shapes = getImageShape(labels,paths)
    #sizes  = getImageSize(labels,paths)
    pixels = getAmountOfPixels(labels,paths)
    pixelsSorted = np.sort(pixels)
    pixelsSortedSmall = pixelsSorted/1000
    pixelsSortedSmallRound = np.round(pixelsSortedSmall, 0)
    
    AmountOfPhotosPerPixelSize = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    for p in pixelsSortedSmallRound:
        if(p>=np.min(pixelsSortedSmallRound) and p<=500):
            AmountOfPhotosPerPixelSize[1] = AmountOfPhotosPerPixelSize[1]+1
        elif(p>500 and p<=1000):
            AmountOfPhotosPerPixelSize[2] = AmountOfPhotosPerPixelSize[2]+1
        elif(p>1000 and p<=1500):
            AmountOfPhotosPerPixelSize[3] = AmountOfPhotosPerPixelSize[3]+1
        elif(p>1500 and p<=2000):
            AmountOfPhotosPerPixelSize[4] = AmountOfPhotosPerPixelSize[4]+1
        elif(p>2000 and p<=2500):
            AmountOfPhotosPerPixelSize[5] = AmountOfPhotosPerPixelSize[5]+1
        elif(p>2500 and p<=3000):
            AmountOfPhotosPerPixelSize[6] = AmountOfPhotosPerPixelSize[6]+1
        elif(p>3000 and p<=np.max(pixelsSortedSmallRound)):
            AmountOfPhotosPerPixelSize[7] = AmountOfPhotosPerPixelSize[7]+1
             
    heightPixel = []
    for i in AmountOfPhotosPerPixelSize.values():
        heightPixel.append(i)
    
    s = 0
    for i in range(len(heightPixel)):
        s = s+heightPixel[i]
    
    heightPixelPercentage = []
    for i in range(len(heightPixel)):
        heightPixelPercentage.append((round(((heightPixel[i])/s), 4)*100))
    
    barsPixel = ["0-500","501-1000","1001-1500","1501-2000","2001-2500","2501-3000","3000-..."]
    
    y_pos = np.arange(len(barsPixel))
    # Create horizontal bars
    plt.barh(y_pos, heightPixel)
    
    # Create names on the y-axis
    plt.yticks(y_pos, barsPixel)
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    plt.title("Amount of pictures per pixel size category", **titlepar)
    yLabelPar = {'rotation' : '0',
                 'weight' : 'bold'}
    plt.ylabel("Pixelsize category", **yLabelPar).set_position([-1.02, 1])
    plt.xlabel("Amount of pictures").set_weight('bold')
    
    for i, v in enumerate(heightPixelPercentage):
        plt.text(v + 2, i - 0.1, str(str(v) + "%"), color='black',  fontweight='bold')
    # Show graphic
    plt.show()
    
showAmountOfPicturesPerPixelSizeCategory(label_strings, train_paths)
"""Looking at the sizes and amount of pixels of the images, we can conclude that they are quite equal.
        Almost 80% of the images is in the same range, while the other 20% is just a bit smaller or bigger."""

def getTotalAmountOfImagesInCertainClasses(labels, paths):
    amount = 0
    a = getAmountOfPhotosPerClass(labels, paths)
    for label in labels:
        amount = amount + a[label]
    return amount

avgAmountOfImagesPerClass = avgAmountOfImagesForCertainClasses(label_strings, train_paths)
print(avgAmountOfImagesPerClass)
amountOfDalmatians = getTotalAmountOfImagesInCertainClasses(["dalmatian"], train_paths)
print(amountOfDalmatians)
amountOfWolfs = getTotalAmountOfImagesInCertainClasses(["wolf"], train_paths)
print(amountOfWolfs)
totalAmountOfImages = getTotalAmountOfImagesInCertainClasses(label_strings,train_paths)
print(totalAmountOfImages)

data = np.c_[train_data, train_labels]
pd_data = pd.DataFrame(data)
"""nrs = []
for nr in range((codebook_size+1)):
    nrs.append(str('Feature ' + str(nr)))
print(nrs)
pd_data.columns = nrs"""
pd_data.hist(figsize = (60,60)) 
titlepar = {'weight' : 'bold',
        'size' : '15'}
plt.title("Feature histograms per codebook cluster", **titlepar)
plt.show()
"""Feature 500 is clearly differently distributed --> will not be significant"""
    
"""-------------------------------------------------------------------------"""
"""----------------2. Preprocessing & Feature extraction--------------------"""
"""-------------------------------------------------------------------------"""

"""Data augmentation: create more images"""

path=r'E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data\\GettingStarted'
os.chdir(path)
import Augmentor
import shutil

#New filepaths
DATA_BASE_PATH = 'E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data'
OUTPUT_PATH= 'E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data\\Output'

DATA_TRAIN_PATHAug = os.path.join(DATA_BASE_PATH,'trainAug')
DATA_TEST_PATH = os.path.join(DATA_BASE_PATH,'test')

FEATURE_BASE_PATHAug = os.path.join(OUTPUT_PATH,'featuresAug')
FEATURE_TRAIN_PATHAug = os.path.join(FEATURE_BASE_PATHAug,'trainAug')
FEATURE_TEST_PATHAug = os.path.join(FEATURE_BASE_PATHAug,'testAug')

PREDICTION_PATHAug = os.path.join(OUTPUT_PATH,'predictionsAug')

# filepatterns to write out features
FILEPATTERN_DESCRIPTOR_TRAINAug = os.path.join(FEATURE_TRAIN_PATHAug,'train_features_{}.pkl')
FILEPATTERN_DESCRIPTOR_TESTAug = os.path.join(FEATURE_TEST_PATHAug,'test_features_{}.pkl')

# create paths in case they don't exist:
helpers.createPath(DATA_TRAIN_PATHAug)
helpers.createPath(FEATURE_BASE_PATHAug)
helpers.createPath(FEATURE_TRAIN_PATHAug)
helpers.createPath(FEATURE_TEST_PATHAug)
helpers.createPath(PREDICTION_PATHAug)
helpers.createPath(FILEPATTERN_DESCRIPTOR_TESTAug)

"""Copy all existing images to new folder "trainAug" so that we do not confuse the features of the augmented and non-augmented dataset"""
folderNames = os.listdir(DATA_TRAIN_PATH)
for f in folderNames:
    helpers.createPath(os.path.join(DATA_TRAIN_PATHAug, f))
    files = os.listdir(os.path.join(DATA_TRAIN_PATH, f))
    for file in files:
        folderDirectory = os.path.join(DATA_TRAIN_PATH, f)
        shutil.copy(os.path.join(folderDirectory, file), os.path.join(DATA_TRAIN_PATHAug, f))

folder_pathsAug=glob.glob(os.path.join(DATA_TRAIN_PATHAug,'*'))
label_stringsAug = np.sort(np.array([os.path.basename(path) for path in folder_pathsAug]))
num_classesAug = label_stringsAug.shape[0]

"""-------------------------------------------------------------------------"""
train_pathsAug = dict((label_string, helpers.getImgPaths(os.path.join(DATA_TRAIN_PATHAug,label_string))) for label_string in label_stringsAug)
"""=paths to all training images"""
test_pathsAug = helpers.getImgPaths(DATA_TEST_PATH)
print(label_stringsAug)

"""We know that dalmatians and wolves are the minorities in this case
    --> begin with creating more of these (flip, zoom, ...) till the inequalities are balanced out more or less"""
    
NORMAL_TRAIN_IMAGES_PATH = (r"E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data\\trainAug")
avgOfAllClasses = avgAmountOfImagesForCertainClasses(['bobcat', 'chihuahua', 'collie', 'dalmatian', 'german_shepherd', 'leopard','lion', 'persian_cat', 'siamese_cat', 'tiger', 'wolf'], train_pathsAug)
for i in label_stringsAug:
    if avgAmountOfImagesForCertainClasses([i], train_pathsAug)<avgOfAllClasses:
        THIS_TRAIN_IMAGES_PATH = os.path.join(NORMAL_TRAIN_IMAGES_PATH, i)
        aug = Augmentor.Pipeline(THIS_TRAIN_IMAGES_PATH)
        aug.ground_truth(THIS_TRAIN_IMAGES_PATH)
        aug.flip_left_right(probability=0.5)
        aug.zoom_random(probability=0.5, percentage_area=0.8)
        aug.greyscale(probability=0.3)
        aug.black_and_white(probability=0.3)
        if(avgAmountOfImagesForCertainClasses([i], train_pathsAug)<250):
            aug.sample(200)
        else:
            aug.sample(100)
        THIS_OUTPUT_PATH = os.path.join(THIS_TRAIN_IMAGES_PATH, r"output")
        files = os.listdir(THIS_OUTPUT_PATH)
        for f in files:
            shutil.move(os.path.join(THIS_OUTPUT_PATH, f), THIS_TRAIN_IMAGES_PATH)
        os.rmdir(THIS_OUTPUT_PATH)

"""Check again for class imbalance"""
train_pathsAug = dict((label_string, helpers.getImgPaths(os.path.join(DATA_TRAIN_PATHAug,label_string))) for label_string in label_stringsAug)
"""=paths to all training images"""
test_pathsAug = helpers.getImgPaths(DATA_TEST_PATH)

"""Crop the images to 90% from the centre --> 10% unuseful data removed"""
NORMAL_TRAIN_IMAGES_PATH = (r"E:\\Allerlei\\Sander\\Andere\\School\\Universiteit\\Master\\Machine Learning\\Competition\\Data\\trainAug")
avgOfNormalClasses = avgAmountOfImagesForCertainClasses(['bobcat', 'chihuahua', 'collie', 'dalmatian', 'german_shepherd', 'leopard','lion', 'persian_cat', 'siamese_cat', 'tiger', 'wolf'], train_pathsAug)
for i in label_stringsAug:
    if avgAmountOfImagesForCertainClasses([i], train_pathsAug)<avgOfNormalClasses:
        THIS_TRAIN_IMAGES_PATH = os.path.join(NORMAL_TRAIN_IMAGES_PATH, i)
        aug = Augmentor.Pipeline(THIS_TRAIN_IMAGES_PATH)
        aug.ground_truth(THIS_TRAIN_IMAGES_PATH)
        aug.crop_centre(probability = 1, percentage_area = 0.1)
        THIS_OUTPUT_PATH = os.path.join(THIS_TRAIN_IMAGES_PATH, r"output")
        files = os.listdir(THIS_OUTPUT_PATH)
        for f in files:
            shutil.move(os.path.join(THIS_OUTPUT_PATH, f), THIS_TRAIN_IMAGES_PATH)
        os.rmdir(THIS_OUTPUT_PATH)
        
train_pathsAug = dict((label_string, helpers.getImgPaths(os.path.join(DATA_TRAIN_PATHAug,label_string))) for label_string in label_stringsAug)
"""=paths to all training images"""
test_pathsAug = helpers.getImgPaths(DATA_TEST_PATH)

showAmountOfPicturesAvailalbeForTrainingPerClass(label_stringsAug, train_pathsAug)

"""-------------------------------------------------------------------------"""
"""--------------------Bag Of Visual Words Model----------------------------"""
"""-------------------------------------------------------------------------"""

"""Good representations of interesting patches"""

# add all features which you would like computed and their callbacks to this dictionary
# features where a pickle file already exists will not be recomputed
descriptor_dictAug={'daisy':features.extractDAISYCallback, # SIFT replacement, very fast, can be computed dense if necessary
                 'orb':features.extractORBCallback, # another fast SIFT replacement, oriented BRIEF w. FAST keypoints  
                 'freak':features.extractFREAKCallback, # biologically motivated descriptor
                 'lucid':features.extractLUCIDCallback,  
                 'vgg':features.extractVGGCallback, # Trained as proposed by VGG lab, don't confuse it with VGG-Net features
                 'boost_desc':features.extractBoostDescCallback} # Image descriptor learned with boosting
                 
if features.checkForSIFT():
    descriptor_dictAug['sift'] = features.extractSIFTCallback # One descriptor to rule them all, unfortunately patented

"""-------------------------------------------------------------------------"""

train_descriptor_dictAug = descriptor_dictAug.copy()

# if the corresponding files already exist, do not extract them again
train_descriptor_dictAug = dict((key,value) for (key,value) in descriptor_dictAug.items() if not os.path.isfile(FILEPATTERN_DESCRIPTOR_TRAINAug.format(key)))

if len(train_descriptor_dictAug) > 0: 

    train_featuresAug = []
    train_labelsAug = []
    
    # convert train images
    train_features_by_descriptorAug = dict((key,[]) for (key,value) in train_descriptor_dictAug.items())
    
    for label_stringAug in label_stringsAug:
        print('extracting train features for class {} :'.format(label_stringAug))

        extracted_featuresAug = features.extractFeatures(train_pathsAug[label_stringAug],train_descriptor_dictAug, label_stringAug)

        # append descriptors of corresponding label to correct descriptor list 
        for key in train_features_by_descriptorAug.keys():
            train_features_by_descriptorAug[key]+=extracted_featuresAug[key]
  
    for descriptor_keyAug in train_features_by_descriptorAug.keys():
        with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_keyAug),'wb') as pkl_file_train:
            pickle.dump(train_features_by_descriptorAug[descriptor_keyAug], pkl_file_train, protocol=pickle.HIGHEST_PROTOCOL)
    
"""-------------------------------------------------------------------------"""
test_pathsAug = helpers.getImgPaths(DATA_TEST_PATH)

test_descriptor_dictAug=dict((key,value) for (key,value) in descriptor_dictAug.items() if not os.path.isfile(FILEPATTERN_DESCRIPTOR_TESTAug.format(key)))

if len(test_descriptor_dictAug) > 0: 
    test_featuresAug = []
    
    print('extracting test features:') 
    
    test_features_by_descriptorAug = features.extractFeatures(test_pathsAug,test_descriptor_dictAug, None) 
    
    for descriptor_keyAug in test_features_by_descriptorAug.keys():
        with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_keyAug),'wb') as pkl_file_test:
            pickle.dump(test_features_by_descriptorAug[descriptor_keyAug], pkl_file_test, protocol=pickle.HIGHEST_PROTOCOL)

"""-------------------------------------------------------------------------"""

descriptor_desiredAug='freak'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugFreak=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugFreak = pickle.load(pkl_file_test)

"""-------------------------------------------------------------------------"""

descriptor_desiredAug='orb'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugOrb=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TEST.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugOrb = pickle.load(pkl_file_test)
    
"""-------------------------------------------------------------------------"""

descriptor_desiredAug='daisy'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugDaisy=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugDaisy = pickle.load(pkl_file_test)
    
"""-------------------------------------------------------------------------"""

descriptor_desiredAug='lucid'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugLucid=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugLucid = pickle.load(pkl_file_test)
    
"""-------------------------------------------------------------------------"""

descriptor_desiredAug='boost_desc'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugBoost_desc=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugBoost_desc = pickle.load(pkl_file_test)

"""-------------------------------------------------------------------------"""

descriptor_desiredAug='vgg'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugVgg=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugVgg = pickle.load(pkl_file_test)
    
"""-------------------------------------------------------------------------"""

descriptor_desiredAug='sift'
with open(FILEPATTERN_DESCRIPTOR_TRAINAug.format(descriptor_desiredAug),'rb') as pkl_file_train:
    train_features_from_pklAugSift=pickle.load(pkl_file_train)

with open(FILEPATTERN_DESCRIPTOR_TESTAug.format(descriptor_desiredAug),'rb') as pkl_file_test:
    test_features_from_pklAugSift = pickle.load(pkl_file_test)

"""Create codebook with Freak"""
codebook_size = 500
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugFreak = helpers.createCodebook(train_features_from_pklAugFreak, codebook_size = codebook_size)
# encode all train images 
train_dataAugFreak=[]
train_labelsAugFreak=[]

for image_featuresAug in train_features_from_pklAugFreak:
    bow_feature_vectorAugFreak = helpers.encodeImage(image_featuresAug.data,clustered_codebookAugFreak)
    train_dataAugFreak.append(bow_feature_vectorAugFreak)
    train_labelsAugFreak.append(image_featuresAug.label)

# encode all test images 
test_dataAugFreak=[]
for image_featuresAug in test_features_from_pklAugFreak:
    bow_feature_vectorAugFreak=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugFreak)
    test_dataAugFreak.append(bow_feature_vectorAugFreak)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugFreak[:10])
train_labelsAugFreak = label_encoder.transform(train_labelsAugFreak)
print(train_labelsAugFreak[:10])

"""Create codebook with orb"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugOrb = helpers.createCodebook(train_features_from_pklAugOrb, codebook_size = codebook_size)
# encode all train images 
train_dataAugOrb=[]
train_labelsAugOrb=[]

for image_featuresAug in train_features_from_pklAugOrb:
    bow_feature_vectorAugOrb = helpers.encodeImage(image_featuresAug.data,clustered_codebookAugOrb)
    train_dataAugOrb.append(bow_feature_vectorAugOrb)
    train_labelsAugOrb.append(image_featuresAug.label)

# encode all test images 
test_dataAugOrb=[]
for image_featuresAug in test_features_from_pklAugOrb:
    bow_feature_vectorAugOrb=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugOrb)
    test_dataAugOrb.append(bow_feature_vectorAugOrb)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugOrb[:10])
train_labelsAugOrb = label_encoder.transform(train_labelsAugOrb)
print(train_labelsAugOrb[:10])

"""Create codebook with Daisy"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugDaisy = helpers.createCodebook(train_features_from_pklAugDaisy, codebook_size = codebook_size)
# encode all train images 
train_dataAugDaisy=[]
train_labelsAugDaisy=[]

for image_featuresAug in train_features_from_pklAugDaisy:
    bow_feature_vectorAugDaisy = helpers.encodeImage(image_featuresAug.data,clustered_codebookAugDaisy)
    train_dataAugDaisy.append(bow_feature_vectorAugDaisy)
    train_labelsAugDaisy.append(image_featuresAug.label)

# encode all test images 
test_dataAugDaisy=[]
for image_featuresAug in test_features_from_pklAugDaisy:
    bow_feature_vectorAugDaisy=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugDaisy)
    test_dataAugDaisy.append(bow_feature_vectorAugDaisy)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugDaisy[:10])
train_labelsAugDaisy = label_encoder.transform(train_labelsAugDaisy)
print(train_labelsAugDaisy[:10])    

"""Create codebook with Lucid"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugLucid= helpers.createCodebook(train_features_from_pklAugLucid, codebook_size = codebook_size)
# encode all train images 
train_dataAugLucid=[]
train_labelsAugLucid=[]

for image_featuresAug in train_features_from_pklAugLucid:
    bow_feature_vectorAugLucid = helpers.encodeImage(image_featuresAug.data,clustered_codebookAugLucid)
    train_dataAugLucid.append(bow_feature_vectorAugLucid)
    train_labelsAugLucid.append(image_featuresAug.label)

# encode all test images 
test_dataAugLucid=[]
for image_featuresAug in test_features_from_pklAugLucid:
    bow_feature_vectorAugLucid=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugLucid)
    test_dataAugLucid.append(bow_feature_vectorAugLucid)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugLucid[:10])
train_labelsAugLucid = label_encoder.transform(train_labelsAugLucid)
print(train_labelsAugLucid[:10])
    
"""Create codebook with boost_desc"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugBoost_desc= helpers.createCodebook(train_features_from_pklAugBoost_desc, codebook_size = codebook_size)
# encode all train images 
train_dataAugBoost_desc=[]
train_labelsAugBoost_desc=[]

for image_featuresAug in train_features_from_pklAugBoost_desc:
    bow_feature_vectorAugBoost_desc= helpers.encodeImage(image_featuresAug.data,clustered_codebookAugBoost_desc)
    train_dataAugBoost_desc.append(bow_feature_vectorAugBoost_desc)
    train_labelsAugBoost_desc.append(image_featuresAug.label)

# encode all test images 
test_dataAugBoost_desc=[]
for image_featuresAug in test_features_from_pklAugBoost_desc:
    bow_feature_vectorAugBoost_desc=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugBoost_desc)
    test_dataAugBoost_desc.append(bow_feature_vectorAugBoost_desc)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugBoost_desc[:10])
train_labelsAugBoost_desc = label_encoder.transform(train_labelsAugBoost_desc)
print(train_labelsAugBoost_desc[:10])  
  
"""Create codebook with vgg"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugVgg= helpers.createCodebook(train_features_from_pklAugVgg, codebook_size = codebook_size)
# encode all train images 
train_dataAugVgg=[]
train_labelsAugVgg=[]

for image_featuresAug in train_features_from_pklAugVgg:
    bow_feature_vectorAugVgg= helpers.encodeImage(image_featuresAug.data,clustered_codebookAugVgg)
    train_dataAugVgg.append(bow_feature_vectorAugVgg)
    train_labelsAugVgg.append(image_featuresAug.label)

# encode all test images 
test_dataAugVgg=[]
for image_featuresAug in test_features_from_pklAugVgg:
    bow_feature_vectorAugVgg=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugVgg)
    test_dataAugVgg.append(bow_feature_vectorAugVgg)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugVgg[:10])
train_labelsAugVgg = label_encoder.transform(train_labelsAugVgg)
print(train_labelsAugVgg[:10])    

"""Create codebook with SIFT"""
# learn the codebook for the 'freak' features from the training data
clustered_codebookAugSift= helpers.createCodebook(train_features_from_pklAugSift, codebook_size = codebook_size)
# encode all train images 
train_dataAugSift=[]
train_labelsAugSift=[]

for image_featuresAug in train_features_from_pklAugSift:
    bow_feature_vectorAugSift= helpers.encodeImage(image_featuresAug.data,clustered_codebookAugSift)
    train_dataAugSift.append(bow_feature_vectorAugSift)
    train_labelsAugSift.append(image_featuresAug.label)

# encode all test images 
test_dataAugSift=[]
for image_featuresAug in test_features_from_pklAugSift:
    bow_feature_vectorAugSift=helpers.encodeImage(image_featuresAug.data,clustered_codebookAugSift)
    test_dataAugSift.append(bow_feature_vectorAugSift)
    
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labelsAugSift[:10])
train_labelsAugSift = label_encoder.transform(train_labelsAugSift)
print(train_labelsAugSift[:10])    

"""-------------------------------------------------------------------------------------"""
"""Create some models to get a first impression which descriptor and model might be best"""
"""-------------------------------------------------------------------------------------"""  
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

OneVsRestLogReg= OneVsRestClassifier(LogisticRegression())
LogReg = LogisticRegression()
LogRegCV = LogisticRegressionCV()
LinSVC = LinearSVC()
LDA = LinearDiscriminantAnalysis()

models=[]
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Logistic Regression CV implemented', LogisticRegressionCV()))
models.append(('OneVsRestClassifier on Logistic Regression', OneVsRestClassifier(LogisticRegression())))
models.append(('Linear SVC', LinearSVC()))
models.append(('LDA', LinearDiscriminantAnalysis()))

from sklearn.model_selection import train_test_split
"""Theory: train = 50%, val = 25%, test = 25%
    Here: test set given separately, so we do not really need to take that into account when we split our (train) data provided
    Therefore, we will keep 25% of the (training) data provided for our validation set, the remainding 75% will be our pure training data
    Note: these are the default values in sklearn"""

"""For every type of feature extraction"""
x_trainAugDaisy, x_testAugDaisy, y_trainAugDaisy, y_testAugDaisy = train_test_split(train_dataAugDaisy, train_labelsAugDaisy, random_state = 2)
x_trainAugOrb, x_testAugOrb, y_trainAugOrb, y_testAugOrb = train_test_split(train_dataAugOrb, train_labelsAugOrb, random_state = 2)
x_trainAugFreak, x_testAugFreak, y_trainAugFreak, y_testAugFreak = train_test_split(train_dataAugFreak, train_labelsAugFreak, random_state = 2)
x_trainAugLucid, x_testAugLucid, y_trainAugLucid, y_testAugLucid = train_test_split(train_dataAugLucid, train_labelsAugLucid, random_state = 2)
x_trainAugVgg, x_testAugVgg, y_trainAugVgg, y_testAugVgg = train_test_split(train_dataAugVgg, train_labelsAugVgg, random_state = 2)
x_trainAugBoost_desc, x_testAugBoost_desc, y_trainAugBoost_desc, y_testAugBoost_desc = train_test_split(train_dataAugBoost_desc, train_labelsAugBoost_desc, random_state = 2)
x_trainAugSift, x_testAugSift, y_trainAugSift, y_testAugSift = train_test_split(train_dataAugSift, train_labelsAugSift, random_state = 2)

d = np.c_[train_dataAugFreak, train_labelsAugFreak]
pd_d = pd.DataFrame(d)
titlepar = {'weight' : 'bold',
        'size' : '15'}
plt.title("Feature histograms per codebook cluster using Freak features", **titlepar)
pd_d.hist(figsize = (60,60)) 
plt.show()

"""See which descriptor might be most suitable with basic LogisticRegression"""
descriptorNames = ["Daisy", "Orb", "Freak", "Lucid", "Vgg", "Boost_desc", "Sift"]
trainDataPerName = {}
trainDataPerName["Daisy"] = x_trainAugDaisy
trainDataPerName["Orb"] = x_trainAugOrb
trainDataPerName["Freak"] = x_trainAugFreak
trainDataPerName["Lucid"] = x_trainAugLucid
trainDataPerName["Vgg"] = x_trainAugVgg
trainDataPerName["Boost_desc"] = x_trainAugBoost_desc
trainDataPerName["Sift"] = x_trainAugSift
trainLabelsPerName = {}
trainLabelsPerName["Daisy"] = y_trainAugDaisy
trainLabelsPerName["Orb"] = y_trainAugOrb
trainLabelsPerName["Freak"] = y_trainAugFreak
trainLabelsPerName["Lucid"] = y_trainAugLucid
trainLabelsPerName["Vgg"] = y_trainAugVgg
trainLabelsPerName["Boost_desc"] = y_trainAugBoost_desc
trainLabelsPerName["Sift"] = y_trainAugSift

result = []
kfold = KFold(n_splits=10, random_state=7)
for key in trainDataPerName.keys():
    result.append(cross_val_score(LogisticRegression(), trainDataPerName[key], trainLabelsPerName[key], cv=kfold, scoring=scoring).mean())

plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Comparison of different descriptors",**titlepar)
plt.xlabel("Descriptor", **xpara)
plt.ylabel("Score", **ypara).set_position([-1, 0.45])
plt.plot(descriptorNames, result, 'o-', color="b")
plt.show()

"""First impression: use default values of linear models for Freak features"""
results=[]
names=[]
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_trainAugFreak, y_trainAugFreak, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
# boxplot algorithm comparison
fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
plt.ylabel("Cross validation score", **ypara).set_position([-1.5, 1.01])
plt.title("Cross validation score per linear model (default values) Freak", **titlepar)
plt.ylim(ymin = 0.30, ymax = 0.50)
xpara = {'rotation' : '40',
         'ha' : 'right'}
ax.set_xticklabels(names, **xpara)
plt.show()

def first_impression_model_comparison(modelsList, x_train, y_train, cv):
    results=[]
    names=[]
    scoring = 'accuracy'
    for name, model in modelsList:
        cv_results = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    ypara = {'weight' : 'bold',
             'rotation' : '0',}
    plt.ylabel("Cross validation score", **ypara).set_position([-1.5, 1.01])
    plt.title("Cross validation score per model", **titlepar)
    plt.ylim(ymin = 0.30, ymax = 0.50)
    xpara = {'rotation' : '40',
             'ha' : 'right'}

"""First impression: use default values of linear models for Sift features"""
results=[]
names=[]
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_trainAugSift, y_trainAugSift, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
# boxplot algorithm comparison
fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
plt.ylabel("Cross validation score", **ypara).set_position([-1.5, 1.01])
plt.title("Cross validation score per linear model (default values) SIFT", **titlepar)
plt.ylim(ymin = 0.40, ymax = 0.60)
xpara = {'rotation' : '40',
         'ha' : 'right'}
ax.set_xticklabels(names, **xpara)
plt.show()

"""We will continue with the OneVsRestClassifier(LogisticRegression())"""
"""We shall not use Linear SVC (no predict_proba) or Logistic Regression with CV implemented (difficult to train the parameters we want)"""
modelAugSift1 = OneVsRestClassifier(LogisticRegression())
modelAugSift1.fit(x_trainAugSift, y_trainAugSift)

"""-------------------------------------------------------------------------"""
"""------------3.Train/validate/test split & validation approach------------"""
"""-------------------------------------------------------------------------"""  
"""Check if we should use StratifiedKFold"""
scoring = "accuracy"
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(modelAugSift1, x_trainAugSift,y_trainAugSift, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("modelAugSift1", cv_results.mean(), cv_results.std())
print(msg)
    """0.475291 (0.015403) with 10 folds"""

scoring = "accuracy"
stratKFold10 = StratifiedKFold(n_splits = 10)
cv_results = cross_val_score(modelAugSift1, x_trainAugSift,y_trainAugSift, cv=stratKFold10, scoring=scoring)
msg = "%s: %f (%f)" % ("modelAugSift1", cv_results.mean(), cv_results.std())
print(msg)
    """0.476313 (0.014329) with 10 folds"""
    """We shall not use StratifiedKFold, because the mean and std are very close and the biggest reason why StratifiedKFold 
            is used is to solve class imbalance, which we already solved."""
      
"""Tune regularization parameter C and penalty (should always be done when we choose a certain model)"""
from sklearn.model_selection import GridSearchCV
modelAugSift1.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
penaltyTypesToTest = ["l1", "l2"]
#classWeightsToTest = ['None', 'Balanced']
#solverToTest = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# Set the parameters by cross-validation
tuned_parameters = [{'estimator__C': CvaluesToTest,
                        'estimator__penalty': penaltyTypesToTest}]
                            #'estimator__class_weight': classWeightsToTest,
                             #   'estimator__solver': solverToTest,
kfold10 = KFold(n_splits=10, random_state=7)
CV = GridSearchCV(modelAugSift1, tuned_parameters, cv=kfold10)
CV.fit(x_trainAugSift, y_trainAugSift)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestPenalty = CV.best_params_['estimator__penalty']
"""bestClassWeight = CV.best_params_['estimator__class_weight']
bestSolver = CV.best_params_['estimator__solver']"""

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
       
plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Tune regularization parameter and penalty method",**titlepar)
plt.xlabel("Value of regularization parameter C", **xpara)
plt.ylabel("Score", **ypara)
l1means = []
l2means = []
l1stds = []
l2stds = []
for i in range(len(cv_means)):
    if (i%2)==0:
        l2means.append(cv_means[i])
        l2stds.append(cv_stds[i])
    else:
        l1means.append(cv_means[i])
        l1stds.append(cv_stds[i])
plt.plot()
plt.plot(CvaluesToTest, l1means, 'o-', color="r",
         label="L2 Penalty")
plt.plot(CvaluesToTest, l2means, 'o-', color="g",
         label="L1 Penalty")
plt.legend(loc="best")
plt.show()
    """C=10 :  0.507 (0.033) and penalty = L2 (default)"""

"""Only set C=10 and keep penalty at L2"""

modelAugSift2 = OneVsRestClassifier(LogisticRegression(C=10))
modelAugSift2.fit(x_trainAugSift, y_trainAugSift)

"""Optimize amount of folds"""
foldsizes = [2, 5, 10, 20, 50, ]
dMean = {}
dStd = {}
names = []
results = []
for foldSize in foldsizes:
    kfold = KFold(n_splits=foldSize, random_state=7)
    cv_results = cross_val_score(modelAugSift2, x_trainAugSift, y_trainAugSift, cv=kfold, scoring=scoring)   
    dMean[foldSize] = cv_results.mean()
    dStd[foldSize] = cv_results.std()
    results.append(cv_results)
    names.append(foldSize)
    print(foldSize)
    
minScore = {}
maxScore = {}
for key in dMean.keys():
    minScore[key] = dMean[key] - dStd[key]
    maxScore[key] = dMean[key] + dStd[key]

plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0'}
xpara = {'weight' : 'bold'}
plt.title("Tune amount of folds",**titlepar)
plt.xlabel("Amount of folds", **xpara)
plt.ylabel("Cross-validation score", **ypara).set_position([-1.5, 1.01])
plt.plot(dMean.keys(), dMean.values(), 'o-', color="b",label="Mean")
plt.plot(dMean.keys(), minScore.values(), 'o-', color="r",label="Minimum")
plt.plot(dMean.keys(), maxScore.values(), 'o-', color="g",label="Maximum")
plt.legend(loc="best")
plt.show()

    """We can see that the mean cross-validation score with different amount of folds peaks at 10 folds and afterwards stays quite steady.
        After 10 folds, the maximum cross-validation score that can be achieved increases, but the minimum decreases as well.
            As we want to make a model that is as accurate as possible, we will minimize our std while maximizing the mean.
                Therefore, 10 folds is optimal."""
                
kfold10 = KFold(n_splits=10)
cv_results10 = cross_val_score(modelAugSift2, x_trainAugSift, y_trainAugSift, cv=kfold10, scoring=scoring)   
msg10 = "%s: %f (%f)" % ("modelAugSift2", cv_results10.mean(), cv_results10.std())
print(msg10)
    """0.506588 (0.016501) with KFold with 10 folds"""
    
"""-------------------------------------------------------------------------"""
"""---------------4. Dimensionality reduction/feature selection-------------"""
"""-------------------------------------------------------------------------"""     
"""Feature selection"""
    """Forward selection"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        
        f = np.zeros(codebook_size)
        kfold10 = KFold(n_splits=10, random_state=7)

        for i in range(codebook_size):
            selector = SelectKBest(chi2, k=i+1).fit(x_trainAugSift, y_trainAugSift)
            x_new=selector.transform(x_trainAugSift)
            model = modelAugSift2
            f[i] = cross_val_score(model, x_new, y_trainAugSift, cv=kfold10).mean()
            print("Average accuracy with ",i+1," features: ",f[i])
    
        plt.figure()
        titlepar = {'weight' : 'bold',
                'size' : '15'}
        ypara = {'weight' : 'bold',
                 'rotation' : '0'}
        xpara = {'weight' : 'bold'}
        plt.title("Forward feature selection",**titlepar)
        plt.xlabel("Amount of features", **xpara)
        plt.ylabel("Cross-validation score", **ypara).set_position([-1.5, 1.01])
        plt.plot(np.arange(1.0,codebook_size+1,1.0),f)
        plt.show()
        
        best_features = np.argmax(f)
        print("Optimal performance of ",f[best_features],
              ", for ",best_features+1," features")
        
        """Optimal performance of  0.5065884778621252 for  500  features"""
        
        featureSelector = SelectKBest(chi2, k=best_features+1).fit(x_trainAugSift, y_trainAugSift)
        selector_modelForward = OneVsRestClassifier(LogisticRegression(C=10))
        selector_modelForward.fit(featureSelector.transform(x_trainAugSift), y_trainAugSift)
        kfold10 = KFold(n_splits=10, random_state=7)
        scoreslin = cross_val_score(selector_modelForward, featureSelector.transform(x_trainAugSift), y_trainAugSift, cv=kfold10)
        print("Average CV accuracy of linear model: ",scoreslin.mean(),", stdev: ",scoreslin.std())
            """Average CV accuracy of linear model:  0.5065884778621252 , stdev:  0.01650078506495287"""

        pred_select = selector_modelForward.predict(featureSelector.transform(x_testAugSift))
        predForwardFeatureSiftNoLDA = selector_modelForward.predict_proba(featureSelector.transform(test_data))
        
        train_score_select = selector_modelForward.score(featureSelector.transform(x_trainAugSift), y_trainAugSift)
        test_score_select = selector_modelForward.score(featureSelector.transform(x_testAugSift), y_testAugSift)
        
        print("Accuracy of final model: ",train_score_select,
          " (train), ",test_score_select," (test)")
            """Accuracy of final model:  0.7527799327644169  (train),  0.5050426687354539  (test)"""
            
"""Plot learning curve"""
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    ypara = {'weight' : 'bold',
             'rotation' : '0'}
    xpara = {'weight' : 'bold'}
    plt.title(title, **titlepar)
    plt.xlabel("Amount of training images", **xpara)
    plt.ylabel("Score", **ypara).set_position([-1.5, 0.45])
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross validation score")
    plt.legend(loc="best")
    return plt

kfold10 = KFold(n_splits = 10)
plot_learning_curve(OneVsRestClassifier(LogisticRegression(C=10)), "Learning curve of basic model", X = featureSelector.transform(x_trainAugSift), y = y_trainAugSift, cv = kfold10)
            
"""LDA 99%"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
train_dataAugSiftsource = sc.fit_transform(featureSelector.transform(train_dataAugSift))
test_dataAugSiftsource = sc.transform(featureSelector.transform(test_dataAugSift))

LDA = LinearDiscriminantAnalysis(n_components=None)
X_LDA = LDA.fit(train_dataAugSiftsource, train_labelsAugSift)
LDA_variance_ratios = LDA.explained_variance_ratio_
def selectcomponents(variance_ratio, goal_variance: float) -> int:
    total_variance = 0.0
    n_components = 0
    for explained_variance in variance_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_variance:
            break      
    return n_components
# Run function
LDA = LinearDiscriminantAnalysis(n_components = selectcomponents(LDA_variance_ratios, 0.99))
train_dataAugSiftLDA= LDA.fit_transform(train_dataAugSiftsource, train_labelsAugSift)
test_dataAugSiftLDA = LDA.transform(test_dataAugSiftsource)

from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
x_trainAugSiftLDA, x_testAugSiftLDA, y_trainAugSiftLDA, y_testAugSiftLDA = train_test_split(train_dataAugSiftLDA, train_labelsAugSift, random_state=7)

modelAugSift3 = OneVsRestClassifier(LogisticRegression(C=10))
modelAugSift3.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)
train_score = modelAugSift3.score(x_trainAugSiftLDA, y_trainAugSiftLDA)
test_score = modelAugSift3.score(x_testAugSiftLDA, y_testAugSiftLDA)
print("Accuracy of model after LDA: ",train_score," (train), ",test_score," (test)")
    """Accuracy of model after LDA:  0.6865787432117921  (train),  0.6625290923196276  (test) 99%"""   
    
plot_learning_curve(modelAugSift3, "Learning curve after LDA", train_dataAugSiftLDA, train_labelsAugSift)

"""----------------------------------------------------------------------------------------"""
"""5. Machine learning technique(s) used, training and hyperparameter optimisation approach"""
"""----------------------------------------------------------------------------------------"""     
"""Tune regularization parameter C and penalty"""
from sklearn.model_selection import GridSearchCV
modelAugSift3.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
penaltyTypesToTest = ["l1", "l2"]
tuned_parameters = [{'estimator__C': CvaluesToTest,
                        'estimator__penalty': penaltyTypesToTest}]
kfold10 = KFold(n_splits=10, random_state=7)
CV = GridSearchCV(modelAugSift3, tuned_parameters, cv=kfold10)
CV.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestPenalty = CV.best_params_['estimator__penalty']
"""bestClassWeight = CV.best_params_['estimator__class_weight']
bestSolver = CV.best_params_['estimator__solver']"""

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
       
plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Tune regularization parameter and penalty method",**titlepar)
plt.xlabel("Value of regularization parameter C", **xpara)
plt.ylabel("Score", **ypara)
l1means = []
l2means = []
l1stds = []
l2stds = []
for i in range(len(cv_means)):
    if (i%2)==0:
        l2means.append(cv_means[i])
        l2stds.append(cv_stds[i])
    else:
        l1means.append(cv_means[i])
        l1stds.append(cv_stds[i])
plt.plot()
plt.plot(CvaluesToTest, l1means, 'o-', color="r",
         label="L2 Penalty")
plt.plot(CvaluesToTest, l2means, 'o-', color="g",
         label="L1 Penalty")
plt.xlim( xmax = 4)
plt.legend(loc="best")
plt.show()

"""As penalty L2 is prefered in this case over L1, and the scores are very close, we'll set it back to its default C = 1 and penalty = L2
    Besides, we prefer to minimize the std, which in this case is minimized for these values: 0.680 (+/-0.057)"""

"""Tune regularization parameter C and class weight"""
from sklearn.model_selection import GridSearchCV
modelAugSift3.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
classWeightsToTest = ['balanced', None]
tuned_parameters = [{'estimator__C': CvaluesToTest,
                            'estimator__class_weight': classWeightsToTest}]
kfold10 = KFold(n_splits=10, random_state=7)
CV = GridSearchCV(modelAugSift3, tuned_parameters, cv=kfold10)
CV.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestclassWeight = CV.best_params_['estimator__class_weight']

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
       
plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Tune regularization parameter and class weight method",**titlepar)
plt.xlabel("Value of regularization parameter C", **xpara)
plt.ylabel("Score", **ypara).set_position([-1.0, 0.45])
balancedmeans = []
nonemeans = []
balancedstds = []
nonestds = []
for i in range(len(cv_means)):
    if (i%2)==0:
        nonemeans.append(cv_means[i])
        nonestds.append(cv_stds[i])
    else:
        balancedmeans.append(cv_means[i])
        balancedstds.append(cv_stds[i])
plt.plot()
plt.plot(CvaluesToTest, nonemeans, 'o-', color="r",
         label="Class weight = None")
plt.plot(CvaluesToTest, balancedmeans, 'o-', color="g",
         label="Class weight = Balanced")
plt.xlim(xmax = 4)
plt.legend(loc="best")
plt.show()

"""As the std of class weight mathod 'balanced' is in general higher than that of class weight 'None', and the scores are very close, we'll set it back to its default C = 1 and class weight 'None'
       0.680 (+/-0.057)"""

"""Tune regularization parameter C and solver method"""
from sklearn.model_selection import GridSearchCV
modelAugSift3.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
solverToTest = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# Set the parameters by cross-validation
tuned_parameters = [{'estimator__C': CvaluesToTest,
                            'estimator__solver': solverToTest}]
kfold10 = KFold(n_splits=10, random_state=7)
CV = GridSearchCV(modelAugSift3, tuned_parameters, cv=kfold10)
CV.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestSolver = CV.best_params_['estimator__solver']

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

"""We'll keep this at its default values as well
    0.680 (+/-0.057)"""
    
modelAugSift4 = OneVsRestClassifier(LogisticRegression())
modelAugSift4.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)

train_score = modelAugSift4.score(x_trainAugSiftLDA, y_trainAugSiftLDA)
test_score = modelAugSift4.score(x_testAugSiftLDA, y_testAugSiftLDA)

print("Accuracy of model after step 5: ",train_score," (train), ",test_score," (test)")
    """Accuracy of model after step 5:  0.6870959400051719  (train),  0.6656322730799069  (test)"""  

"""-------------------------------------------------------------------------"""
"""-------------------6.1 Analysis of intermediate results-------------------"""
"""-------------------------------------------------------------------------"""   
"""Plot ROC"""
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
kfold10 = KFold(n_splits=10)
def plotROC(x, y, nameofTest, test, cv):
    y_bin = label_binarize(y, classes=[0,1,2,3,4,5,6,7,8,9,10])
    n_classes = y_bin.shape[1]
    
    pipeline= Pipeline([('scaler', StandardScaler()), (nameofTest, test)])
    y_score = cross_val_predict(pipeline, x, y, cv=cv ,method='predict_proba')
    f = dict()
    t = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        f[i], t[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(f[i], t[i])
    colors =(['blue', 'red', 'green','yellow', 'black','coral','orange','cyan','peachpuff','deeppink','forestgreen'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(f[i], t[i], color=color, lw=3,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(label_strings[i], roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    titlepar = {'weight' : 'bold',
            'size' : '15'}
    ypara = {'weight' : 'bold',
             'rotation' : '0',}
    xpara = {'weight' : 'bold'}
    plt.xlabel('False Positive Rate', **xpara)
    plt.ylabel('True Positive \n Rate', **ypara).set_position([-1, 0.43])
    plt.title('ROC for image classification', **titlepar)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


    """The goal is to maximize the AUC (perfection = 1) of each class, we can see that only the collie and german_shepherd class are lagging behind a bit,
            but still have an AUC of 0.89 and 0.88 respectively, which is pretty good.
                The classes with the biggest class imbalance originaly (dalmatian and wolf) seem to have catched up with a AUC of 0.99 and 0.97 respectively.
                        We just have to check if we are not overfitting now"""
"""Learning curve"""

plot_learning_curve(modelAugSift4, "Learning curve of best model yet (train data)", X = x_trainAugSiftLDA, y = y_trainAugSiftLDA, cv = kfold10)
plot_learning_curve(modelAugSift4, "Learning curve of best model yet (validation data)", X = x_testAugSiftLDA, y = y_testAugSiftLDA, cv = kfold10)
plot_learning_curve(modelAugSift4, "Learning curve of best model yet", X = train_dataAugSiftLDA, y = train_labelsAugSift, cv = kfold10)

    """It seems that more data will not help a great deal and thus not be worth the money/effort as the graphs are already pretty converged and more data might even have a negative effect. 
        Our bias is simply nog good enough."""

"""Confusion matrix"""
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_conf_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    titlepar = {'weight' : 'bold',
            'size' : '20'}
    ypara = {'weight' : 'bold',
             'rotation' : '0',
             'size' : '14'}
    xpara = {'weight' : 'bold',
             'size' : '14'}
    plt.ylabel('True label', **ypara)
    plt.xlabel('Predicted label', **xpara)
    plt.title(title, **titlepar)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, weight='bold')
    plt.yticks(tick_marks, classes, weight = 'bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

modelAugSift4 = OneVsRestClassifier(LogisticRegression())
modelAugSift4.fit(x_trainAugSiftLDA, y_trainAugSiftLDA)
y_true = y_testAugSiftLDA
y_pred = modelAugSift4.predict(x_testAugSiftLDA)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize = (20, 20))
plot_conf_matrix(cnf_matrix, classes=label_strings,
                      title='Confusion matrix (%)')

# Plot normalized confusion matrix
plt.figure(figsize = (20, 20))
plot_conf_matrix(cnf_matrix, classes=label_strings, normalize=True,
                      title='Normalized onfusion matrix')

"""Pretty good, only the collies, german shephards, persian cats and siamese cats are lagging behind in accuracy."""

"""Possible statistical solutions: add some noise to the data, find a way to extract the interesting pathces and thus do the feature extraction more accurately
        use stratify in train_test_split or StratifiedKFold to make sure any imbalances are taken in to account"""
def plot_confusion_matrix(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_true = y_test
    y_pred = model.predict(x_test)
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize = (20, 20))
    plot_conf_matrix(cnf_matrix, classes=label_strings,
                          title='Confusion matrix (%)')
    
    # Plot normalized confusion matrix
    plt.figure(figsize = (20, 20))
    plot_conf_matrix(cnf_matrix, classes=label_strings, normalize=True,
                          title='Normalized onfusion matrix')
"""-------------------------------------------------------------------------"""
"""-----------------------------6.2 Improvements----------------------------"""
"""-------------------------------------------------------------------------"""   

"""Sift feature extraction with stratify"""
x_trainAugSiftStrat, x_testAugSiftStrat, y_trainAugSiftStrat, y_testAugSiftStrat = train_test_split(train_dataAugSift, train_labelsAugSift, random_state = 2, stratify = train_labelsAugSift)
"""First impression: use default values of linear models for Sift features"""
results=[]
names=[]
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_trainAugSiftStrat, y_trainAugSiftStrat, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
plt.ylabel("Cross validation score", **ypara).set_position([-1.5, 1.01])
plt.title("Cross validation score per linear model \n (default values) SIFT, stratify in split", **titlepar)
plt.ylim(ymin = 0.40, ymax = 0.60)
xpara = {'rotation' : '40',
         'ha' : 'right'}
ax.set_xticklabels(names, **xpara)
plt.show()

"""We will continue with the OneVsRestClassifier(LogisticRegression())"""
"""We shall not use Linear SVC (no predict_proba) or Logistic Regression with CV implemented (difficult to train the parameters we want)"""
modelAugSiftStrat1 = OneVsRestClassifier(LogisticRegression())
modelAugSiftStrat1.fit(x_trainAugSiftStrat, y_trainAugSiftStrat)

"""Check if we should use StratifiedKFold"""
scoring = "accuracy"
kfold = KFold(n_splits=10, random_state=7)
cv_results = cross_val_score(modelAugSiftStrat1, x_trainAugSiftStrat,y_trainAugSiftStrat, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ("modelAugSiftStrat1", cv_results.mean(), cv_results.std())
print(msg)
    """First: 0.475291 (0.015403) with 10 folds
        Now: 0.485908 (0.018153) with 10 folds
            --> higher mean AND higher variance"""

scoring = "accuracy"
stratKFold10 = StratifiedKFold(n_splits = 10)
cv_results = cross_val_score(modelAugSiftStrat1, x_trainAugSiftStrat,y_trainAugSiftStrat, cv=stratKFold10, scoring=scoring)
msg = "%s: %f (%f)" % ("modelAugSiftStrat1", cv_results.mean(), cv_results.std())
print(msg)
    """First: 0.476313 (0.014329) with 10 folds
        Now: 0.483342 (0.013622)"""
    """We shall use StratifiedKFold, because the std can be lowered, while increasing the mean value."""
      
"""Tune regularization parameter C and penalty (should always be done when we choose a certain model)"""
from sklearn.model_selection import GridSearchCV
modelAugSiftStrat1.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
penaltyTypesToTest = ["l1", "l2"]
tuned_parameters = [{'estimator__C': CvaluesToTest,
                        'estimator__penalty': penaltyTypesToTest}]
stratKFold10 = StratifiedKFold(n_splits = 10)
CV = GridSearchCV(modelAugSiftStrat1, tuned_parameters, cv=stratKFold10)
CV.fit(x_trainAugSiftStrat, y_trainAugSiftStrat)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestPenalty = CV.best_params_['estimator__penalty']

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
       
plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Tune regularization parameter \n and penalty method",**titlepar)
plt.xlabel("Value of regularization parameter C", **xpara)
plt.ylabel("Score", **ypara)
l1means = []
l2means = []
l1stds = []
l2stds = []
for i in range(len(cv_means)):
    if (i%2)==0:
        l2means.append(cv_means[i])
        l2stds.append(cv_stds[i])
    else:
        l1means.append(cv_means[i])
        l1stds.append(cv_stds[i])
plt.plot()
plt.plot(CvaluesToTest, l1means, 'o-', color="r",
         label="L2 Penalty")
plt.plot(CvaluesToTest, l2means, 'o-', color="g",
         label="L1 Penalty")
plt.legend(loc="best")
plt.show()
    """First: C=10 :  0.507 (0.033) and penalty = L2 (default)
        Now: C=10 : 0.511 (+/-0.051) and penalty = L2 (default)
            --> small increase in mean, bigger relative increase in std!"""

"""Only set C=10 and keep penalty at L2"""

modelAugSiftStrat2 = OneVsRestClassifier(LogisticRegression(C=10))
modelAugSiftStrat2.fit(x_trainAugSiftStrat, y_trainAugSiftStrat)

"""Optimize amount of folds"""
foldsizes = [2, 5, 10, 20, 50, ]
dMean = {}
dStd = {}
names = []
results = []
for foldSize in foldsizes:
    stratKFold10 = StratifiedKFold(n_splits = foldSize)   
    cv_results = cross_val_score(modelAugSiftStrat2, x_trainAugSiftStrat, y_trainAugSiftStrat, cv=stratKFold10, scoring=scoring)   
    dMean[foldSize] = cv_results.mean()
    dStd[foldSize] = cv_results.std()
    results.append(cv_results)
    names.append(foldSize)
    print(foldSize)
    
minScore = {}
maxScore = {}
for key in dMean.keys():
    minScore[key] = dMean[key] - dStd[key]
    maxScore[key] = dMean[key] + dStd[key]

plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0'}
xpara = {'weight' : 'bold'}
plt.title("Tune amount of folds",**titlepar)
plt.xlabel("Amount of folds", **xpara)
plt.ylabel("Cross-validation score", **ypara).set_position([-1.5, 1.01])
plt.plot(dMean.keys(), dMean.values(), 'o-', color="b",label="Mean")
plt.plot(dMean.keys(), minScore.values(), 'o-', color="r",label="Minimum")
plt.plot(dMean.keys(), maxScore.values(), 'o-', color="g",label="Maximum")
plt.legend(loc="best")
plt.show()

    """We can see that the mean cross-validation score with different amount of folds peaks at 20 folds and afterwards stays quite steady.
        After 20 folds, the maximum cross-validation score that can be achieved increases, but the minimum decreases as well.
            As we want to make a model that is as accurate as possible, we will minimize our std while maximizing the mean.
                Therefore, 20 folds is optimal."""
                
stratKFold20= StratifiedKFold(n_splits = 20)
cv_results20 = cross_val_score(modelAugSiftStrat2, x_trainAugSiftStrat, y_trainAugSiftStrat, cv=stratKFold20, scoring=scoring)   
msg20 = "%s: %f (%f)" % ("modelAugSiftStrat2", cv_results20.mean(), cv_results20.std())
print(msg20)
    """First: 0.506588 (0.016501) with KFold with 10 folds
        Now: 0.517962 (0.028181) with StratifiedKFold with 20 folds
            --> std almost doubled!!!"""
    
"""-------------------------------------------------------------------------"""
    
"""Feature selection"""
    """Forward selection"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        
        f = np.zeros(codebook_size)
        stratKFold20= StratifiedKFold(n_splits = 20)

        for i in range(codebook_size):
            selector = SelectKBest(chi2, k=i+1).fit(x_trainAugSiftStrat, y_trainAugSiftStrat)
            x_new=selector.transform(x_trainAugSiftStrat)
            model = modelAugSiftStrat2
            f[i] = cross_val_score(model, x_new, y_trainAugSiftStrat, cv=stratKFold20).mean()
            print("Average accuracy with ",i+1," features: ",f[i])
    
        plt.figure()
        titlepar = {'weight' : 'bold',
                'size' : '15'}
        ypara = {'weight' : 'bold',
                 'rotation' : '0'}
        xpara = {'weight' : 'bold'}
        plt.title("Forward feature selection",**titlepar)
        plt.xlabel("Amount of features", **xpara)
        plt.ylabel("Cross-validation score", **ypara).set_position([-1.5, 1.01])
        plt.plot(np.arange(1.0,codebook_size+1,1.0),f)
        plt.show()
        
        best_features = np.argmax(f)
        print("Optimal performance of ",f[best_features],
              ", for ",best_features+1," features")
        
        """First: Optimal performance of  0.5065884778621252 for  500  features
            Now: Optimal performance of  0.5192735089102127 , for  489  features"""
        
        featureSelectorStrat = SelectKBest(chi2, k=best_features+1).fit(x_trainAugSiftStrat, y_trainAugSiftStrat)
        selector_modelForwardStrat = OneVsRestClassifier(LogisticRegression(C=10))
        selector_modelForwardStrat.fit(featureSelectorStrat.transform(x_trainAugSiftStrat), y_trainAugSiftStrat)
        stratKFold20= StratifiedKFold(n_splits = 20)       
        scoreslinStrat = cross_val_score(selector_modelForwardStrat, featureSelectorStrat.transform(x_trainAugSiftStrat), y_trainAugSiftStrat, cv=stratKFold20)
        print("Average CV accuracy of linear model: ",scoreslinStrat.mean(),", stdev: ",scoreslinStrat.std())
            """First: Average CV accuracy of linear model:  0.5065884778621252 , stdev:  0.01650078506495287
                Now: Average CV accuracy of linear model:  0.5192735089102127 , stdev:  0.027735293911190194"""
        
        pred_selectStrat = selector_modelForwardStrat.predict(featureSelectorStrat.transform(x_testAugSiftStrat))
        predForwardFeatureSiftStrat = selector_modelForwardStrat.predict_proba(featureSelectorStrat.transform(test_data))
        
        train_score_selectStrat = selector_modelForwardStrat.score(featureSelectorStrat.transform(x_trainAugSiftStrat), y_trainAugSiftStrat)
        test_score_selectStrat  = selector_modelForwardStrat.score(featureSelectorStrat.transform(x_testAugSiftStrat), y_testAugSiftStrat)
        
        print("Accuracy of final model: ",train_score_selectStrat,
          " (train), ",test_score_selectStrat," (test)")
            """First: Accuracy of final model:  0.7527799327644169  (train),  0.5050426687354539  (test)
                Now Accuracy of final model:  0.7483837600206878  (train),  0.49418153607447635  (test)
                    --> both values decreased!!! --> worse"""
            
"""Plot learning curve"""
stratKFold20= StratifiedKFold(n_splits = 20)       
plot_learning_curve(OneVsRestClassifier(LogisticRegression(C=10)), "Learning curve of 'optimized' model \n before LDA", X = featureSelectorStrat.transform(x_trainAugSiftStrat), y = y_trainAugSiftStrat, cv = stratKFold20)
            
"""LDA 99%"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
train_dataAugSiftsourceStrat = sc.fit_transform(featureSelectorStrat.transform(train_dataAugSift))
test_dataAugSiftsourceStrat = sc.transform(featureSelectorStrat.transform(test_dataAugSift))

LDA = LinearDiscriminantAnalysis(n_components=None)
X_LDA = LDA.fit(train_dataAugSiftsourceStrat, train_labelsAugSift)
LDA_variance_ratiosStrat = LDA.explained_variance_ratio_
# Run function
LDA = LinearDiscriminantAnalysis(n_components = selectcomponents(LDA_variance_ratiosStrat, 0.99))
train_dataAugSiftStratLDA= LDA.fit_transform(train_dataAugSiftsourceStrat, train_labelsAugSift)
test_dataAugSiftStratLDA = LDA.transform(test_dataAugSiftsourceStrat)

from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
x_trainAugSiftStratLDA, x_testAugSiftStratLDA, y_trainAugSiftStratLDA, y_testAugSiftStratLDA = train_test_split(train_dataAugSiftStratLDA, train_labelsAugSift, random_state=7, stratify = train_labelsAugSift)

modelAugSift4 = OneVsRestClassifier(LogisticRegression(C=10))
modelAugSift4.fit(x_trainAugSiftStratLDA, y_trainAugSiftStratLDA)
train_scoreStrat = modelAugSift4.score(x_trainAugSiftStratLDA, y_trainAugSiftStratLDA)
test_scoreStrat = modelAugSift4.score(x_testAugSiftStratLDA, y_testAugSiftStratLDA)
print("Accuracy of model after LDA: ",train_scoreStrat," (train), ",test_scoreStrat," (test)")
    """First: Accuracy of model after LDA:  0.6865787432117921  (train),  0.6625290923196276  (test) 99%
        Now: Accuracy of model after LDA:  0.6720972329971554  (train),  0.6780449961210241  (test)"""   
    
"""----------------------------------------------------------------------------------------"""

"""Tune regularization parameter C and penalty"""
from sklearn.model_selection import GridSearchCV
modelAugSift4.get_params(deep=True)
CvaluesToTest = [1.0e-2,1.0e-1,1.0,2.0,5.0,10.0]
penaltyTypesToTest = ["l1", "l2"]
tuned_parameters = [{'estimator__C': CvaluesToTest,
                        'estimator__penalty': penaltyTypesToTest}]
stratKFold20= StratifiedKFold(n_splits = 20)       
CV = GridSearchCV(modelAugSift4, tuned_parameters, cv=stratKFold20)
CV.fit(x_trainAugSiftStratLDA, y_trainAugSiftStratLDA)

print("Best parameter set found on development set: ",CV.best_params_)
# store the best optimization parameter for later reuse
bestC2 = CV.best_params_['estimator__C']
bestPenalty = CV.best_params_['estimator__penalty']
"""bestClassWeight = CV.best_params_['estimator__class_weight']
bestSolver = CV.best_params_['estimator__solver']"""

print("Grid scores on training data set:")
print()
cv_means = CV.cv_results_['mean_test_score']
cv_stds = CV.cv_results_['std_test_score']
for mean, std, params in zip(cv_means, cv_stds, CV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
       
plt.figure()
titlepar = {'weight' : 'bold',
        'size' : '15'}
ypara = {'weight' : 'bold',
         'rotation' : '0',}
xpara = {'weight' : 'bold'}
plt.title("Tune regularization parameter and penalty method \n of 'optimized' model",**titlepar)
plt.xlabel("Value of regularization parameter C", **xpara)
plt.ylabel("Score", **ypara).set_position([-1,1.01])
l1means = []
l2means = []
l1stds = []
l2stds = []
for i in range(len(cv_means)):
    if (i%2)==0:
        l2means.append(cv_means[i])
        l2stds.append(cv_stds[i])
    else:
        l1means.append(cv_means[i])
        l1stds.append(cv_stds[i])
plt.plot()
plt.plot(CvaluesToTest, l1means, 'o-', color="r",
         label="L2 Penalty")
plt.plot(CvaluesToTest, l2means, 'o-', color="g",
         label="L1 Penalty")
plt.xlim( xmax = 4)
plt.legend(loc="best")
plt.show()

"""First: As penalty L2 is prefered in this case over L1, and the scores are very close, we'll set it back to its default C = 1 and penalty = L2
           Besides, we prefer to minimize the std, which in this case is minimized for these values: 0.680 (+/-0.057)
    Now: As penalty L2 is prefered in this case over L1, and the scores are very close, we'll set it back to its default C = 1 and penalty = L2
           Besides, we prefer to minimize the std, which in this case is minimized for these values: 0.668 (+/-0.085)
    --> This was not a good optimization!
    """
    
modelAugSift5 = OneVsRestClassifier(LogisticRegression())
modelAugSift5.fit(x_trainAugSiftStratLDA, y_trainAugSiftStratLDA)

train_scoreStrat = modelAugSift5.score(x_trainAugSiftStratLDA, y_trainAugSiftStratLDA)
test_scoreStrat = modelAugSift5.score(x_testAugSiftStratLDA, y_testAugSiftStratLDA)

print("Accuracy of model after step 5: ",train_scoreStrat," (train), ",test_scoreStrat," (test)")
    """First: Accuracy of model after step 5:  0.6870959400051719  (train),  0.6656322730799069  (test)
        Now: Accuracy of model after step 5:  0.6723558313938454  (train),  0.6788207913110939  (test)"""  

"""-------------------------------------------------------------------------"""
