import os
import sys
import json
import joblib
import settings
import numpy as np
from staticAnalyzer import run
from tensorflow.keras.models import load_model
from secml.ml.classifiers import CClassifierSVM
from secml.array import CArray
from secml.explanation import CExplainerGradientInput
from secml.data import CDataset


def text_pred(num):
    return "malware" if num == 1 else "goodware"


feature_file = "known_features.json"

#usage: python analyze_apk.py path_to_apk

if len(sys.argv) != 2 or not sys.argv[1].endswith("apk"):
    print("[E]\tUsage: python",sys.argv[0],"path_to_apk")
    sys.exit()

print("[I]\tExctracting features from APK...")
apk_features = run(sampleFile=sys.argv[1], workingDir=".").keys()


known_features = set()
file = open(feature_file, "r")
known_features.update(json.load(file))
file.close()


data = list()

for known_f in sorted(known_features):
    if known_f in sorted(apk_features):
        data.append(1)
    else:
        data.append(0)

X = np.array(data).reshape(1, -1)

svm_clf = joblib.load(settings.SVM_MODEL_PATH)
svm_pred = svm_clf.predict(X)

knn_clf = joblib.load(settings.KNN_MODEL_PATH)
knn_pred = knn_clf.predict(X)

dnn_clf = load_model(settings.DNN_MODEL_PATH)
dnn_pred = dnn_clf.predict(X)

print("\n\t--- PREDICTIONS ---",
        "\n\tSVM says:", text_pred(svm_pred[0]),
        "\n\tKNN says:", text_pred(knn_pred[0]),
        "\n\tDNN says:", text_pred(np.rint(dnn_pred[0])),
        "\n\n")


secml_clf = joblib.load(settings.SECML_MODEL_PATH)
attr = CExplainerGradientInput(secml_clf).explain(x=CArray(data), y=svm_pred[0])
attr = attr / attr.norm(order=1)

attr_argsort = abs(attr).argsort().ravel()[::-1]

n_plot = 5

for i in attr_argsort[:n_plot]:
    print(attr[i].item()*100, "\t", sorted(known_features)[i])

