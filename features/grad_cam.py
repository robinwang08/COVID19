
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from vis.utils import utils
import keras
from vis.visualization import visualize_cam, visualize_saliency, overlay
from keras.models import load_model, Model
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as misc
import cv2
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from vis.utils import utils
import keras
from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils.utils import load_img, normalize, find_layer_idx
from keras.models import load_model, Model
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score
from sklearn import manifold
import pandas
from config import config
from glob import glob
import matplotlib.pyplot as plt
import efficientnet.keras as efn

model = load_model('/media/user1/model.h5')

layer_idx = utils.find_layer_idx(model, 'dense_6')
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)
penultimate_layer_idx = utils.find_layer_idx(model, 'top_conv')

#model.summary()

basepath = '/media/user1/from_gradcam'
toPath = '/media/user1/to_gradcam/'
basedir = os.path.normpath(basepath)
files = glob(basedir+'/*.npy')

for file in files:
    image_file = file

    pathget = os.path.normpath(file)
    folder = pathget.split(os.sep)
    savePath = toPath + folder[7] + '.jpg'

    img = np.load(image_file)
    img = misc.imresize(img, (224, 224), 'bilinear')
    image = np.stack((img, img, img), axis=2)
    seed_input = image / 255.

    grad_top1 = visualize_cam(model, layer_idx,
                              filter_indices=0, seed_input=seed_input,
                              penultimate_layer_idx=penultimate_layer_idx,
                              backprop_modifier=None)

    #grad_top1 = grad_top1 / 255
    seed_input = seed_input * 255

    heat = cv2.resize(grad_top1, (seed_input.shape[0], seed_input.shape[1]))
    result = cv2.addWeighted(seed_input, 0.5, heat, 0.4, 0, dtype = cv2.CV_8U)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(result)
    plt.savefig(savePath)
    plt.close('all')


