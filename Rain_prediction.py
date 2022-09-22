import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("testset.csv")
print(df.shape)

df.drop(df.columns[[0,1,2,3,4,5,6,7,10,12,13,14,15,16,17,18,19]] ,axis='columns',inplace=True)
print(df.sample(5))

#print(df.dtypes)
X = np.array(df)
Y = X[:, 1]
X = np.delete(X, 1, 1)
#print(X)
#print(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=5)

import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

yp = model.predict(X_test)
yp[:5]

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


