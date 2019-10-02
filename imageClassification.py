"""Classify pictures of animals
"""
#%%
import re
# from fastai import *
from fastai.vision import get_image_files, ImageDataBunch, untar_data, get_transforms, imagenet_stats

batchsize = 64
path = untar_data("./data/oxford-iiit-pet")
annotationPath = path / "annotations"
imagePath = path / "images"

fileNames = get_image_files(imagePath)
regexPattern = re.compile(r"/([^/]+)_\d+.jpg$")

data = ImageDataBunch.from_name_re(
    imagePath,
    fileNames,
    regexPattern,
    ds_tfms=get_transforms(),
    size=224,
    bs=batchsize,
    num_workers=0,
).normalize(imagenet_stats)


#%%

#%%

#%%
