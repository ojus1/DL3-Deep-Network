import numpy as np
from keras.applications import inception_v3

input_shape = (299,299,3)

model =  inception_v3.InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape, pooling="max")

raw_folder = "batched_test/X/"

for i in range(20, 100) :
    X_raw = np.load(raw_folder + "X_batch" + str(i) + ".npy")
    X_features = model.predict_on_batch(X_raw)
    np.save("batched_test/X_features/X_batch" + str(i), X_features)
