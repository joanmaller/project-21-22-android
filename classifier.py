import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = []
labels = []

input_file = open("data_X.json")
data = json.load(input_file)
input_file.close()

input_file = open("labels_y.json")
labels = json.load(input_file)
input_file.close()

X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import LinearSVC 
clf = LinearSVC(C=0.7, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
          verbose=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc='Liner SVC Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred)*100)
print(acc)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

#Now we test with a KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=1)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

acc2 = 'KNN Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred2)*100)
print(acc2)
print(confusion_matrix(y_test, y_pred2))

#We try with Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(500,  activation='relu'))
model.add(Dense(270, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#We set a 10% Validation set
X_train,X_val,y_train,y_val = train_test_split(np.array(data),np.array(labels),test_size = 0.1)
history = model.fit(X_train,y_train,
              batch_size=10,
              epochs=50,
              validation_data=(X_val, y_val),
              shuffle=True)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

#y_pred = model.predict_classes(X_ts) # Before tensorflow 2.6
scr = model.predict(X_test) # We extract the score for each class ...
y_pred3 = np.rint(scr)     # ... and then we round it to the nearest integer
acc3='CNN Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred3)*100)
print(acc3)
score = model.evaluate(X_test, y_test, batch_size=250)
print(score)