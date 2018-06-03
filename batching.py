import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import gc

batch_size = 63
folder = 'train_preprocessed/'

img_height = 299
img_width = 299

import os
names = os.listdir(folder)
names.remove("train.csv")

import pandas as pd
y = pd.read_csv("train_preprocessed/train.csv")
print(y.head())

for i in range(0, 150) :
    y_temp = y.iloc[i * batch_size : i * batch_size + batch_size ,1:]
    y_temp.to_csv("batched/Y/Y_batch" + str(i) + ".csv" , index=False) 

from tqdm import tqdm

for i in range(0, 150) :
    train_img = []

    for img_path in tqdm(names[batch_size * i : batch_size * i + batch_size]):
        img = load_img(folder + img_path, target_size=(img_height, img_width))
        train_img.append(img_to_array(img))

    X_temp = np.array(train_img, dtype=np.float16) / 255
    mean = X_temp.mean(axis=0)
    std = X_temp.std(axis=0)
    X_temp = (X_temp - mean)/std

    np.save("batched/X/X_batch" + str(i),X_temp)
    del train_img
    del X_temp
    gc.collect()
