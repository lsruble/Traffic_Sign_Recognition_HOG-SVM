from sklearn.preprocessing import StandardScaler
import csv
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preprocessing import preprocessing_data, label_to_integer, integer_to_label
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import cv2 as cv2
from sklearn.svm import SVC
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

# Configuration des transformations pour l'augmentation des données
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.6),
    A.MedianBlur(blur_limit=7, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    ToTensorV2()
])

# Prétraitement des données
imgs_names, imgs_bb, classes_indices = preprocessing_data()
signs = []
labels = []
img_name = []

# Extraction des panneaux de signalisation des images
for i in range(len(imgs_names)):
    img = imread(imgs_names[i])
    for bb in imgs_bb[i]:
        if not bb[4].startswith('ff'):
            labels.append(bb[4])
            signs.append(img[bb[1]:bb[3], bb[0]:bb[2], :])
            img_name.append(imgs_names[i])

# Augmentation des données
augmented_signs_tensors = []
augmented_labels = []
all_img_name = img_name.copy()

for i in range(2):
    all_img_name += img_name
    for image, label in zip(signs, labels):
        try:
            augmented = transform(image=image, labels=label)
            augmented_signs_tensors.append(augmented['image'])
            augmented_labels.append(augmented['labels'])
        except:
            print("Erreur augmentation")

# Conversion des tenseurs en arrays numpy
augmented_signs = [tensor.numpy() for tensor in augmented_signs_tensors]
augmented_signs = [np.transpose(array, (1, 2, 0)) for array in augmented_signs]
signs = signs + augmented_signs        
labels = labels + augmented_labels 

# Extraction des caractéristiques HOG
fd = []
for j, sign in enumerate(signs):      
    try:
        sign = cv2.resize(sign, (40, 40))
        fd_k, _ = hog(sign, orientations=8, pixels_per_cell=(5,5), cells_per_block=(2, 2), visualize=True, multichannel=True)
        fd.append(fd_k)
    except:
        print(f"error hog {labels[j]} {j}")
        labels.pop(j)
        signs.pop(j)

print(f"Nombre total d'échantillons : {len(signs)}, {len(labels)}")

# Préparation des données pour l'entraînement
hog_features = np.array(fd)
labels = np.array(labels).reshape(-1, 1)
data_frame = np.hstack((hog_features, labels))
data_frame, signs = shuffle(data_frame, signs)

# Séparation des données en ensembles d'entraînement et de test
percentage = 80
partition = int(len(hog_features) * percentage / 100)
x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()

print("Début de l'entraînement")

# Entraînement et évaluation du modèle SVM
print("Entraînement SVM")
clf = SVC(C=8, tol=1e-3, probability=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))    
joblib.dump(clf, 'svm_model.joblib') 

# Entraînement et évaluation du modèle Random Forest
print("Entraînement Random Forest")
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train, y_train)
y_pred_rf = rf_clf.predict(x_test)
y_prob_rf = rf_clf.predict_proba(x_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
joblib.dump(rf_clf, 'random_forest_model.joblib')