"""Classify pictures of animals
"""
#%%
import re
# from fastai import *
# from fastai.vision import (
#     error_rate,
#     models,
#     create_cnn,
#     get_image_files,
#     get_transforms,
#     ImageDataBunch,
#     imagenet_stats,
#     untar_data,
# )

from fastai.vision import *



batchSize = 64
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
    bs=batchSize,
    num_workers=0,
).normalize(imagenet_stats)


#%%
learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

#%%
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

#%%
interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=2)

#%%
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')

learn.lf_find()
learn.recorder.plot()

learn.unfreeze()
learn.fit_one_cylce(2, max_lr=slice(1e-6. 1e-4))
