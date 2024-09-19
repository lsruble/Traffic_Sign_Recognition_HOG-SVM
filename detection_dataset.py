# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:03:15 2024
@author: louis
"""
import cv2
import numpy as np
import joblib
import csv
import os.path
from tqdm import tqdm
from utils_processing import *
from data_preprocessing import preprocessing_test_data

# Constantes
MODEL_PATH = 'svm_hog.joblib'
RESULTS_PATH = 'results_map_test.txt'
RESULTS_FOLDER = 'results_test'
THRE_PROB = 0.80

def process_image(image_path, model):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    img_gray = convert_to_max_grayscale(image)
    (_, binary_image) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    regions = do_MSER(image) + do_gray_mser(img_gray) + binary_processing(image)
    if not regions:
        return [], image
    
    x_test = process_box_hog(image, regions)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    
    for r, region in enumerate(regions):
        predicted_label = y_pred[r] 
        predicted_class_idx = label_to_integer(predicted_label)
        predicted_prob = y_prob[r][predicted_class_idx]
        regions[r] = tuple(region) + (label_to_integer(predicted_label), predicted_prob)
    
    regions = nms(regions, 0.8, 0.3)
    return regions, image


def main():
    # Chargement du modèle et préparation des données
    model = joblib.load(MODEL_PATH)
    imgs_names = preprocessing_test_data()
    
    # Création du dossier de résultats 
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Traitement des images
    with open(RESULTS_PATH, 'w') as file:
        writer = csv.writer(file)
        for i, img_name in enumerate(tqdm(imgs_names, desc="Traitement des images")):
            regions, image = process_image(img_name, model)
            
            if regions:
                image_with_predictions = draw_predictions(image, regions)
                cv2.imwrite(f"{RESULTS_FOLDER}/{i+1}.jpg", image_with_predictions)
                for region in regions:
                    if region[5] > THRE_PROB:
                        x1, y1, x2, y2 = region[:4]
                        nbr = int(extract_image_number(img_name))
                        label_txt = integer_to_label(region[4])
                        json_line = [nbr, x1, y1, x2, y2, round(region[5], 2), label_txt]
                        writer.writerow(json_line)

if __name__ == "__main__":
    main()