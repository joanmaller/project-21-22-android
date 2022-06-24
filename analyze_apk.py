import os
import sys
import json
import joblib
import settings
import numpy as np
from staticAnalyzer import run


feature_file = "known_features.json"

#usage: python analyze_apk.py path_to_apk

if len(sys.argv) != 2 or not sys.argv[1].endswith("apk"):
    print("[E]\tUsage: python",sys.argv[0],"path_to_apk")
    sys.exit()

print("[I]\tExctracting features from APK...")
apk_features = run(sampleFile=sys.argv[1], workingDir=".")


known_features = set()
file = open(feature_file, "r")
known_features.update(json.load(file))
file.close()


data = list()

for known_f in known_features:
    if known_f in apk_features:
        data.append(1)
    else:
        data.append(0)

X = np.array(data).reshape(1, -1)

svm_clf = joblib.load(settings.SVM_MODEL_PATH)
pred = svm_clf.predict(X)

print(pred)
