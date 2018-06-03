from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
import numpy as np
import pandas as pd

folder = 'batched_train/'

input_shape = (2048,)

#Building network
input_layer = Input(shape=input_shape)


l = Dense(200, activation="relu")(input_layer)
l = Dropout(0.3)(l)
l = BatchNormalization()(l)
l = Dense(300, activation="relu")(l)
l = Dropout(0.3)(l)
l = BatchNormalization()(l)
l = Dense(400, activation="relu")(l)
l = Dropout(0.3)(l)
l = Dense(85, activation="softmax")(l)

model = Model(inputs=input_layer, outputs=l)

model.compile(optimizer='SGD',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

scores = list()

import gc
itr = 149
num_epoch = 1000
for ep in range(0, num_epoch) :
    for i in range(0, 150) :
        X_batch = np.load(folder + "X_features/X_batch" + str(i) + ".npy")
        Y_batch = pd.read_csv(folder + "Y/Y_batch" + str(i) + ".csv")
        Y_batch = Y_batch.values
        model.train_on_batch(X_batch, Y_batch)
        
        if i % 2 == 0 & itr <= 199 :
            X_val = np.load(folder + "X_features/X_batch" + str(itr) + ".npy")
            Y_val = pd.read_csv(folder + "Y/Y_batch" + str(itr) + ".csv")
            Y_val = Y_val.values
            print(model.test_on_batch(X_val, Y_val))
            scores.append(model.test_on_batch(X_val, Y_val)) 
            model.save_weights("weights2.h5")
            model.save("model2.h5")
            itr += 1   
        if itr >= 199 :
            itr = 149
    
        
del X_batch
del Y_batch
import gc
gc.collect()

model.save_weights("weights.h5")
model.save("model1.h5")
with open("score.txt", 'w+b') as scorefile:
    for item in scores :
        scorefile.writeline(str(item[0]) + "," + str(item[1]))