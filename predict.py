
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import collections

import urllib.request

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]

def predict_10_crop(img, ix, top_n=5, plot=False, preprocess=True, debug=False):
    print('Img shape:', np.array(img).shape)
    flipped_X = np.fliplr(img)
    size = 150
    crops = [
        img[:size,:size, :], # Upper Left
        img[:size, img.shape[1]-size:, :], # Upper Right
        img[img.shape[0]-size:, :size, :], # Lower Left
        img[img.shape[0]-size:, img.shape[1]-size:, :], # Lower Right
        center_crop(img, (size, size)),

        flipped_X[:size,:size, :],
        flipped_X[:size, flipped_X.shape[1]-size:, :],
        flipped_X[flipped_X.shape[0]-size:, :size, :],
        flipped_X[flipped_X.shape[0]-size:, flipped_X.shape[1]-size:, :],
        center_crop(flipped_X, (size, size))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])

    print('Crop. shape:', np.array(crops).shape)
    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
        print('True Label:', y_test[ix])
    return preds, top_n_preds

def predict(img, model, top_n=5, debug=False):
    resized_img = np.resize(img, (1, 244, 244, 3))
    y_pred = model.predict(np.array(resized_img))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
    return preds, top_n_preds

def predict_remote_image(url, model, ix_to_class, debug=False):
    with urllib.request.urlopen(url) as f:
        pic = plt.imread(f, format='jpg')
        preds = predict(np.array(pic), model, 0, debug=debug)[0]
        best_pred = collections.Counter(preds).most_common(1)[0][0]
        print(ix_to_class[best_pred])
        # plt.imshow(pic)

def predict_image(path, model, ix_to_class, debug=False):
    pic = img.imread(path)
    preds = predict(np.array(pic), model, 0, debug=debug)[0]
    best_pred = collections.Counter(preds).most_common(1)[0][0]
    print(ix_to_class[best_pred])

# model = load_model(filepath='./model.2.hdf5')

# class_to_ix = {}
# ix_to_class = {}
# with open('food-5/meta/classes.txt', 'r') as txt:
#     classes = [l.strip() for l in txt.readlines()]
#     class_to_ix = dict(zip(classes, range(len(classes))))
#     ix_to_class = dict(zip(range(len(classes)), classes))
#     class_to_ix = {v: k for k, v in ix_to_class.items()}
# sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

# # predict_remote_image(url='https://lmld.org/wp-content/uploads/2012/07/Chocolate-Ice-Cream-3.jpg', model=model, ix_to_class=ix_to_class, debug=True)
# # predict_remote_image(url='https://images-gmi-pmc.edge-generalmills.com/75593ed5-420b-4782-8eae-56bdfbc2586b.jpg', model=model, ix_to_class=ix_to_class, debug=True)
# predict_image(path='food-5/images/apple_pie/98352.jpg', model=model, ix_to_class=ix_to_class, debug=True)
# predict_image(path='food-5/images/ice_cream/ice_cream/988684.jpg', model=model, ix_to_class=ix_to_class, debug=True)
