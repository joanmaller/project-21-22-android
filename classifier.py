import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import settings
import joblib
import keras
from keras.utils.vis_utils import plot_model



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
clf = LinearSVC(C=0.8, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=10000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
          verbose=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
acc='Liner SVC Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred)*100)
print(acc)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

if not os.path.exists(settings.MODELS):
    os.mkdir(settings.MODELS)

joblib.dump(clf, settings.SVM_MODEL_PATH)
print("[I]\tSVM model saved to", settings.SVM_MODEL_PATH)

# --- try evaluation with secml ---
# snippet based on tutorial from:
# https://github.com/pralab/secml/blob/master/tutorials/13-Android-Malware-Detection.ipynb

from secml.ml.classifiers import CClassifierSVM
from secml.ml.peval.metrics import CMetricTPRatFPR, CMetricF1, CRoc
from secml.array import CArray
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval
from secml.data import CDataset

secml_clf = CClassifierSVM(C=0.1)
secml_clf.fit(X_train, y_train)

ts = CDataset(x=X_test, y=y_test)

secml_pred, score_pred = secml_clf.predict(X_test, return_decision_function=True)
secml_acc = 'SecML SVC Accuracy: %.2f %%' % (accuracy_score(y_test, secml_pred.get_data())*100)
print(secml_acc)

fpr_th = 0.02  # 2% False Positive Rate
dr = CMetricTPRatFPR(fpr=fpr_th).performance_score(y_true=ts.Y, score=score_pred[:, 1].ravel())
print("Detection rate @ 2% FPR: {:.2%}".format(dr))

params = {
    "classifier": secml_clf,
    "distance": 'l2',
    "double_init": False,
    "lb": 'x0',
    "ub": 1,
    "attack_classes": 'all',
    "y_target": 0,
    "solver_params": {'eta': 1, 'eta_min': 1, 'eta_max': None, 'eps': 1e-4}
}

evasion = CAttackEvasionPGDLS(**params)
n_mal = 10

# Attack DS
mal_idx = ts.Y.find(ts.Y == 1)[:n_mal]
adv_ds = ts[mal_idx, :]

# Security evaluation parameters
param_name = 'dmax'  # This is the `eps` parameter
dmax_start = 0
dmax = 28
dmax_step = 4

param_values = CArray.arange(
    start=dmax_start, step=dmax_step, stop=dmax + dmax_step)

sec_eval = CSecEval(
    attack=evasion,
    param_name=param_name,
    param_values=param_values)

print("Running security evaluation...")
sec_eval.run_sec_eval(adv_ds)
print("Security evaluation completed!")




#Now we test with a KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=5, weights='distance',
        algorithm='auto', leaf_size=100000)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

acc2 = 'KNN Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred2)*100)
print(acc2)
print(confusion_matrix(y_test, y_pred2))

joblib.dump(clf2, settings.KNN_MODEL_PATH)
print("[I]\tKNN model saved to", settings.KNN_MODEL_PATH)




#We try with Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(
        initial_learning_rate = 0.001,
        decay_steps = 1000,
        decay_rate = 0.25)

opt = Adam(learning_rate=lr_schedule)

#, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)
n_features = np.shape(data)[1]

model = Sequential()
model.add(Dense(n_features,  activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(n_features/10, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(n_features/100, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model.compile(optimizer=opt,loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
#model.summary() # check what's the issue 


#plot CNN model

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  #test if the new import is working

#We set a 10% Validation set
X_train,X_val,y_train,y_val = train_test_split(np.array(data),np.array(labels),test_size = 0.1)

history = model.fit(X_train,y_train,
              batch_size=50,
              epochs=25,
              validation_data=(X_val, y_val),
              shuffle=True)


#Plot AUC
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

#Plot AUC
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model auc')
plt.ylabel('AUC (Area under Curve)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


# Plot history: Loss
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.title('L1/L2 Activity Loss')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

scr = model.predict(X_test) # We extract the score for each class ...
y_pred3 = np.rint(scr)     # ... and then we round it to the nearest integer
acc3='CNN Accuracy: %.2f %%' % (accuracy_score(y_test, y_pred3)*100)
print(acc3)
score = model.evaluate(X_test, y_test, batch_size=250)
print(score)
print(confusion_matrix(y_test, y_pred3))


model.save(settings.DNN_MODEL_PATH)
print("[I]\tDNN model saved to", settings.DNN_MODEL_PATH)

