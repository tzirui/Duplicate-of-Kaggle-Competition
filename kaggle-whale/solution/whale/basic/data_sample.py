import collections
import random
import numpy as np
import pandas as pd
import os


# function name: sampleImages
# dict_of_list: key is string, value is list of string
# lower_limit: ignore the whale type when image number under or equal this threshold 
# sample_num: number of images to sample
def sampleImages(dict_of_list, sample_num, lower_limit=0):
    """randomly pick out images from large training set

    Args:
        dict_of_list: key is string (whale type), value is list of string (whale image files)
        lower_limit: ignore the whale type when image number under or equal this threshold 
        sample_num: number of images to sample

    Returns:
        A dict whose key is the file name and value is type of this whale
    """

    sample_Dict={}

    while(sample_num>0):
        # randomly choose the whale type
        rand_type = random.choice(list(whaleDict.keys()))

        # ignore low counts types
        if lower_limit > 0 and len(whaleDict[rand_type]) <= lower_limit:
            continue

        # randomly choose whale image of specific type
        rand_file = random.choice(whaleDict[rand_type])
        
        # ignore duplicate image
        if rand_file in sample_Dict:
            continue
        
        sample_Dict[rand_file] = rand_type
        sample_num -= 1
        
    # return the list of sampled images 
    return sample_Dict


# import the train.csv into DataFrame
# file_path: path to 'train.csv'
file_path = './data/train.csv'
train_df = pd.read_csv(file_path)

# group images by whale type (ID)
group = train_df.groupby(train_df['Id'])
group.get_group('new_whale')['Image']

# reconstruction DataFrame --> Dict
# Key: whaleId
# Value: list of whale image names

whaleDict = collections.defaultdict(list)
for whaleId in group.groups:
    for imgName in group.get_group(whaleId)['Image']:
        whaleDict[whaleId].append(imgName)

# pick out 
sample_Dict=sampleImages(whaleDict,10000)
