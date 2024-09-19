# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:15:33 2024
@author: louis
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import csv
import matplotlib.pyplot as plt
import string as str

def calculate_text_size(text, font):
    """Calcule la taille du texte pour une police donnée."""
    ascent, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = ascent + descent
    return text_width, text_height

def get_brightness(color):
    """Calcule la luminosité d'une couleur."""
    r, g, b = ImageColor.getrgb(color)
    return (r * 299 + g * 587 + b * 114) / 1000 

def visualize_image(filename, csv_filename):
    """Visualise une image avec des boîtes englobantes à partir d'un fichier CSV."""
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)

    if os.path.getsize(csv_filename) > 0:
        with open(csv_filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if row and not row[4].lower().startswith('ff'):
                    xmin, ymin, xmax, ymax = map(int, row[0:4])
                    class_name = row[4]
        
                    class_colors = {
                        'danger': 'green',
                        'interdiction': 'green',
                        'obligation': 'green',
                        'stop': 'green',
                        'ceder': 'green',
                        'frouge': 'red',
                        'forange': 'orange',
                        'fvert': 'green'
                    }
        
                    brightness_threshold = 150  
                    box_color = class_colors.get(class_name, 'white')
                    text_color = 'black' if get_brightness(box_color) > brightness_threshold else 'white'
        
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=box_color)
        
                    font_size = 30
                    font = ImageFont.truetype("arial.ttf", font_size)
        
                    text_width, text_height = calculate_text_size(class_name, font)
        
                    draw.rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill=box_color)
                    draw.text((xmin, ymin - text_height), class_name, fill=text_color, font=font)
    return img

label_to_int = {
    'ceder': 0,
    'danger': 1,
    'forange': 2,
    'frouge': 3,
    'fvert': 4,
    'interdiction': 5,
    'obligation': 6,
    'stop': 7,
}

def inverse_dict(dictionnaire):
    """Inverse un dictionnaire."""
    return {v: k for k, v in dictionnaire.items()}

def label_to_integer(label):
    """Convertit un label en entier."""
    return label_to_int.get(label, None)

def integer_to_label(entier):
    """Convertit un entier en label."""
    inverse = inverse_dict(label_to_int)
    return inverse.get(entier, None)

def preprocessing_data(base_path="train"):
    """Prétraite les données d'entraînement."""
    file_index = ["{:04d}".format(i) for i in range(1, 880)]
    imgs_names = []
    imgs_bb = []
    classes_indices=[[] for _ in range(11)]
    for i, file_name in enumerate(file_index):
        try:
            with open(os.path.join(base_path, 'labels', f'{file_name}.csv'), 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                bounding_boxes = []
                for row in csvreader:
                    if row and not row[4].lower().startswith('ff'):
                        xmin, ymin, xmax, ymax = map(int, row[0:4])
                        class_name = row[4].lower()
                        bounding_boxes.append([xmin, ymin, xmax, ymax, class_name])
                        classes_indices[label_to_integer(class_name)].append(i)

                imgs_bb.append((bounding_boxes))
                imgs_names.append(os.path.join(base_path, 'images', f'{file_name}.jpg'))
        except FileNotFoundError:
             classes_indices[9].append(i)
    print("Prétraitement des images")       
    return imgs_names, imgs_bb, classes_indices

def preprocessing_val_data(base_path="val"):
    """Prétraite les données de validation."""
    return preprocessing_data(base_path)

def preprocessing_test_data(base_path="test"):
    """Prétraite les données de test."""
    file_index = ["{:04d}".format(i) for i in range(1, 880)]
    imgs_names = []
    
    for i, file_name in enumerate(file_index):
        img_path = os.path.join(base_path, f"{file_name}.jpg")
        if os.path.exists(img_path):
            imgs_names.append(img_path)
    return imgs_names