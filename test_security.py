import json
import settings
import joblib

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from secml.ml.classifiers import CClassifierSVM
from secml.array import CArray
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml.kernels import CKernelLinear
from secml.adv.attacks.poisoning import CAttackPoisoningSVM


X_train = joblib.load(settings.X_TRAIN)
X_test = joblib.load(settings.X_TEST)
y_train = joblib.load(settings.Y_TRAIN)
y_test = joblib.load(settings.Y_TEST)


tr = CDataset(x=X_train, y=y_train)
ts = CDataset(x=X_test, y=y_test)

print("\nTrainining samples:", y_train.size,
        "\nTesting samples:", y_test.size)


# --- Evasion---

secml_clf = joblib.load(settings.SECML_MODEL_PATH)

solver_params = {'eta': 1, 'eta_min': 1, 'eta_max': None, 'eps': 1e-4}

evasion = CAttackEvasionPGDLS(
        classifier=secml_clf,
        distance='l2',
        double_init=False,
        lb=0,
        ub=1,
        attack_classes='all',
        y_target=0,
        solver_params=solver_params
        )

n_mal = 0.1*y_test.size

# Attack DS
mal_idx = ts.Y.find(ts.Y == 1)[:n_mal]
adv_ds = ts[mal_idx, :]

# Security evaluation parameters
param_name = 'dmax'  # This is the `eps` parameter
dmax_start = 0
dmax = 5
dmax_step = 1

param_values = CArray.arange(
    start=dmax_start, step=dmax_step, stop=dmax + dmax_step)

sec_eval = CSecEval(
    attack=evasion,
    param_name=param_name,
    param_values=param_values)

print("\n[I]\tEvaluating security on test set...")
sec_eval.run_sec_eval(adv_ds)
print("[I]\tEvaluation completed.")

fig = CFigure(height=5, width=5)
fig.sp.plot_sec_eval(sec_eval.sec_eval_data, marker='o',
        label='SVM', show_average=True)

fig.show()

# --- Dataset Poisoning ---

svm_poisoning = CAttackPoisoningSVM(
        classifier=secml_clf,
        distance='l2',
        training_data=tr,
        val=ts,
        lb=0,
        ub=1,
        solver_params = solver_params,
        y_target=0)

svm_poisoning.x0 = tr[0,:].X #attacker's initial sample features
svm_poisoning.xc = tr[0,:].X #attacker's sample features
svm_poisoning.yc = tr[0,:].Y #attacker's sample label

svm_poisoning.n_points = 0.25*y_train.size #number of poisoned points

print("[I]\tPoisoning training set...")
pois_y_pred, pois_scores, _, _ = svm_poisoning.run(ts.X, ts.Y)
print("[I]\tPoisoning completed.")
pois_acc='Poisoned SVC Accuracy: %.2f %%' % (accuracy_score(y_test, pois_y_pred.get_data())*100)
print(pois_acc)

plt.rcParams.update({'font.size': 24})

cm_pois = confusion_matrix(y_test, pois_y_pred.get_data())
disp_cm_pois = ConfusionMatrixDisplay(confusion_matrix=cm_pois, display_labels=["goodware", "malware"])
disp_cm_pois.plot()
plt.title("Poisoned SecML SVM Confusion Matrix")
plt.show()
