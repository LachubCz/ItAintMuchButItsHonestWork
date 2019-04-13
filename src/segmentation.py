from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import sys
from tools import parse_data



g



sys.exit(1)




BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)




# load your data
x_train, y_train, x_val, y_val =1 

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val),
)

score = model.evaluate(x_test, y_test, batch_size=128)