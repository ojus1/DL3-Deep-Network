import numpy as np
import pandas as pd
from keras.models import Model, load_model

folder = 'batched_test/X_features/X_batch'


model = load_model("model1.h5")
model.load_weights("weights.h5")
model.summary()

Y_pred = pd.DataFrame()

for i in range(1, 100) :
    X = np.load(folder + str(i) + '.npy')
    Y_pred= pd.concat([Y_pred, pd.DataFrame(model.predict_on_batch(X))],axis=0)

Y_pred = Y_pred.values
print(Y_pred)

Y_pred = pd.DataFrame(np.rint(Y_pred))

#col_names = pd.read_csv("../../datasets/DL3_dataset/meta-data/sample_submission.csv")

#Y_pred = pd.concat([col_names, Y_pred], axis=1)
Y_pred.to_csv('Predictions.csv', index=False)