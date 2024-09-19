import numpy as np
import cv2
import glob
import bisect
import copy
import matplotlib.pyplot as plt
import os
from skimage.feature import hog


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
    '''
    Fonction pour inverser un dictionnaire
    '''
    return {v: k for k, v in dictionnaire.items()}

def label_to_integer(label):
    '''
    Fonction pour convertir un label en entier
    '''
    return label_to_int.get(label, None)

def integer_to_label(entier):
    '''
    Fonction pour convertir un entier en label
    '''
    inverse = inverse_dict(label_to_int)
    return inverse.get(entier, None)

def draw_predictions(image, regions):
    '''
    Fonction dessinant une bounding box sur les panneaux routiers
    '''
    for region in regions:
        if region[5] > 0.83:
            x1, y1, x2, y2 = region[:4]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, integer_to_label(region[4]), (x1+10, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def extract_image_number(image_path):
    '''
    Fonction pour extraire un numéro d'un str'
    '''
    parts = image_path.split('/')
    filename = parts[-1]
    number_str = filename[:-4]    
    return number_str

def expand_region(x1, y1, x2, y2, scale_factor=1.25, image_width=None, image_height=None):
    '''
    Fonction agrandissant une bounding box
    '''
    w = x2 - x1
    h = y2 - y1

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    new_width = w * scale_factor
    new_height = h * scale_factor

    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    if image_width is not None:
        new_x1 = max(0, new_x1)
        new_x2 = min(image_width - 1, new_x2)

    if image_height is not None:
        new_y1 = max(0, new_y1)
        new_y2 = min(image_height - 1, new_y2)

    return new_x1, new_y1, new_x2, new_y2

def obtenir_sign_bleu(id_zone):
    """
    Reconnais contours bleus d'un panneau
    """
    global sign_35, sign_38, sign_45

    if id_zone == 35.0:
        return sign_35
    if id_zone == 38.0:
        return sign_38
    if id_zone == 45.0:
        return sign_45

def obtenir_sign_rouge(id_zone):
    """
    Reconnais contours rouges d'un panneau
    """
    global sign_1, sign_14, sign_17, sign_19, sign_21

    if id_zone == 1.0:
        return sign_1
    if id_zone == 14.0:
        return sign_14
    if id_zone == 17.0:
        return sign_17
    if id_zone == 19.0:
        return sign_19
    if id_zone == 21.0:
        return sign_21

def ajuster_image(image_source, tolerance=1, valeurs_entree=[0,255], valeurs_sortie=(0,255)):
    """
    Ajustement de l'image
    """
    tolerance = max(0, min(100, tolerance))

    if tolerance > 0:
        hist = np.histogram(image_source,bins=list(range(256)),range=(0,255))[0]
        cum = hist.copy()
        for i in range(0, 255): cum[i] = cum[i - 1] + hist[i]
        total = image_source.shape[0] * image_source.shape[1]
        low_bound = total * tolerance / 100
        upp_bound = total * (100 - tolerance) / 100
        valeurs_entree[0] = bisect.bisect_left(cum, low_bound)
        valeurs_entree[1] = bisect.bisect_left(cum, upp_bound)

    scale = (valeurs_sortie[1] - valeurs_sortie[0]) / (valeurs_entree[1] - valeurs_entree[0])
    vs = image_source-valeurs_entree[0]
    vs[image_source<valeurs_entree[0]]=0
    vd = vs*scale+0.5 + valeurs_sortie[0]
    vd[vd>valeurs_sortie[1]] = valeurs_sortie[1]
    dst = vd

    return dst

def traiter_image(image):
    """
    Extraction des contours des panneaux
    """

    flou_r = cv2.GaussianBlur(image[:,:,2],(5,5),0)
    flou_g = cv2.GaussianBlur(image[:,:,1],(5,5),0)
    flou_b = cv2.GaussianBlur(image[:,:,0],(5,5),0)

    normalise_r = ajuster_image(flou_r)    
    normalise_g = ajuster_image(flou_g)    
    normalise_b = ajuster_image(flou_b)    
    
    total_normalise = normalise_r + normalise_g + normalise_b

    id_rouge = np.maximum(0,(np.minimum((normalise_r-normalise_b),(normalise_r-normalise_g))))    
    id_bleu = np.maximum(0,(np.minimum((normalise_b-normalise_r),(normalise_b-normalise_g))))

    return id_rouge, id_bleu

def do_MSER(image, contours_arret=None):
    """
    Traitement MSER sur les images pour détecter les régions d'intérêt (ROI) des panneaux routiers

    """
    # Traitement initial de l'image pour isoler les composantes rouge et bleue
    id_rouge, id_bleu = traiter_image(image)
    
    # Conversion des images traitées en uint8 pour compatibilité avec MSER
    nouveau_rouge = id_rouge.copy().astype(np.uint8)
    nouveau_bleu = id_bleu.copy().astype(np.uint8)
    
    # Création des objets MSER pour la détection des régions
    mser_bleu = cv2.MSER_create()
    mser_rouge = cv2.MSER_create()
    
    # Détection des régions MSER pour les composantes rouge et bleue
    region_rouge, _ = mser_rouge.detectRegions(nouveau_rouge)
    region_bleue, _ = mser_bleu.detectRegions(nouveau_bleu)
    
    # Calcul des enveloppes convexes pour les régions détectées
    enveloppe_rouge = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in region_rouge]
    enveloppe_bleue = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in region_bleue]
    
    ROI = []
    
    # Traitement des régions bleues
    for contour in enveloppe_bleue:
        aire = cv2.contourArea(contour)
        x_bleu, y_bleu, largeur, hauteur = cv2.boundingRect(contour)
        ratio_aspect = float(largeur) / hauteur
        # Filtrage des régions basé sur l'aire et le ratio d'aspect
        if aire > 1500 and 0.33 < ratio_aspect < 1.2:
            x1, y1 = x_bleu, y_bleu
            x2, y2 = x1 + largeur, y1 + hauteur
            # Ajout de la région élargie à la liste des ROI
            ROI.append(expand_region(x1, y1, x2, y2, 1.15, image.shape[1], image.shape[0]))
    
    # Traitement des régions rouges (similaire aux régions bleues)
    for contour in enveloppe_rouge:
        aire = cv2.contourArea(contour)
        x_rouge, y_rouge, largeur, hauteur = cv2.boundingRect(contour)
        ratio_aspect = float(largeur) / hauteur
        if aire > 1500 and 0.33 < ratio_aspect < 1.2:
            x1, y1 = x_rouge, y_rouge
            x2, y2 = x1 + largeur, y1 + hauteur
            ROI.append(expand_region(x1, y1, x2, y2, 1.15, image.shape[1], image.shape[0]))
    
    return ROI

def IoU(box1, box2):
    """
    Fonction IOU
    """
    x1, y1, x2, y2 = box1	
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    width = (max(x3, x1) - min(x4, x2))
    height = (max(y3, y1) - min(y4, y2))
    if (width>0) or (height >0):
        return 0.0

    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
   
    return iou




def nms(boxes, prob_threshold=0.7, iou_threshold=0.4):
    """
    Fonction NMS
    """
    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    best_boxes = []
    for box in boxes:
        if box[5] >= prob_threshold:
            keep = True
            for kept_box in best_boxes:
                iou = IoU(box[:4], kept_box[:4])
                if iou > iou_threshold:
                    # Si la boîte actuelle est entièrement contenue dans une boîte gardée
                    # avec une probabilité plus élevée, ignorez-la
                    if is_contained(box[:4], kept_box[:4]) and box[5] < kept_box[5]:
                        keep = False
                    # Sinon, si la boîte gardée est entièrement contenue dans la boîte actuelle
                    # avec une probabilité plus faible, supprimez la boîte gardée
                    elif is_contained(kept_box[:4], box[:4]) and kept_box[5] < box[5]:
                        best_boxes.remove(kept_box)
                    else:
                        keep = False
                        break
            if keep:
                best_boxes.append(box)
    
    to_remove = set()
    
    for i, box1 in enumerate(best_boxes):
        for j, box2 in enumerate(best_boxes):
            if i != j:
                if is_contained(box1, box2):
                    to_remove.add(i)
                elif is_contained(box2, box1):
                    to_remove.add(j)
    
    best_boxes = [box for i, box in enumerate(best_boxes) if i not in to_remove]
    
    return best_boxes

def is_contained(box1, box2):
    """
    Fonction checkant si une bounding box est contenue dans une autre
    """  
    return box1[0] > box2[0] and box1[1] > box2[1] and box1[2] < box2[2] and box1[3] < box2[3]

def major_color_hsv(image_array):
    '''
    Fonction donnant la couleur principale de l'image entre le bleu et le rouge'
    '''

    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    blue_count = cv2.countNonZero(mask_blue)
    red_count = cv2.countNonZero(mask_red)

    if blue_count > red_count:
        return 1.
    else:
        return 0.
    

def eval_detection(y_true, y_pred, iou_threshold=0.5):
    """
    Évalue les métriques TP, FP, FN, précision et rappel pour un ensemble de frames.
    
    Parameters:
    y_true (list) : Liste de listes représentant les bounding boxes ground truth pour chaque frame
                  
    y_pred (list) : Liste de listes représentant les bounding boxes prédites pour chaque frame
                  
    iou_threshold (float) : Seuil d'IoU pour considérer une détection comme TP.
    
    Retourne:
    tuple : Tuple contenant les métriques TP, FP, FN, précision et rappel pour chaque frame.
    """
    
    num_frames = len(y_true)
    false_positives_dir = "falses_positives"
    os.makedirs(false_positives_dir, exist_ok=True)
    # Initialisation des métriques
    tp = np.zeros(num_frames, dtype=int)
    fp = np.zeros(num_frames, dtype=int)
    print(len(fp))
    fn = np.zeros(num_frames, dtype=int)
    precision = np.zeros(num_frames)
    recall = np.zeros(num_frames)
    
    
    for frame_idx in range(num_frames):
        y_true_frame = y_true[frame_idx]
        y_pred_frame = y_pred[frame_idx]
        false_positives = []  # Liste pour stocker les faux positifs de cette frame

        for true_box in y_true_frame:
            detected = False
            
            for pred_box in y_pred_frame:
                if true_box[-1] == pred_box[-1]:
                    box1 = true_box[:4]
                    box2 = pred_box[:4]
                    iou_score = IoU(box1, box2)
                    
                    if iou_score >= iou_threshold:
                        tp[frame_idx] += 1
                        detected = True
                        break
                    else:
                        fp[frame_idx] += 1
                        false_positives.append(pred_box[:4] + ["NotSign"])

            
            if not detected:
                fn[frame_idx] += 1

        precision[frame_idx] = tp[frame_idx] / (tp[frame_idx] + fp[frame_idx]) if tp[frame_idx] + fp[frame_idx] > 0 else 0
        recall[frame_idx] = tp[frame_idx] / (tp[frame_idx] + fn[frame_idx]) if tp[frame_idx] + fn[frame_idx] > 0 else 0
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)
        
        
        frame_file = os.path.join(false_positives_dir, f"{frame_idx}.csv")
        with open(frame_file, "w") as f:
            if false_positives:
                for fps in false_positives:
                    f.write(",".join(map(str, fps)) + "\n")
                else :
                     f.write("")
            
        
    # Calcul de la précision globale
    if total_tp + total_fp > 0:
        global_precision = total_tp / (total_tp + total_fp)
    else:
        global_precision = 0.0
    
    # Calcul du rappel global
    if total_tp + total_fn > 0:
        global_recall = total_tp / (total_tp + total_fn)
    else:
        global_recall = 0.0
    print("Totaux :")
    print("TP :", total_tp)
    print("FP :", total_fp)
    print("FN :", total_fn)
    print("Précision globale :", global_precision)
    print("Rappel global :", global_recall)
    

def get_original_bbox(x1, y1, x2, y2, original_width, original_height):
    """
    Calcule les coordonnées d'une boîte englobante dans l'image d'origine à partir des coordonnées
    dans une image redimensionnée à 500x500.

    Args:
        x1, y1, x2, y2 (int): Coordonnées de la boîte englobante dans l'image redimensionnée (500x500).
        original_width, original_height (int): Dimensions de l'image d'origine.

    Returns:
        tuple: Coordonnées de la boîte englobante dans l'image d'origine (x1, y1, x2, y2).
    """
    new_width, new_height = 500, 500
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    original_x1 = int(x1 / scale_x)
    original_y1 = int(y1 / scale_y)
    original_x2 = int(x2 / scale_x)
    original_y2 = int(y2 / scale_y)

    return original_x1, original_y1, original_x2, original_y2

def convert_to_max_grayscale(image):
    """
    Fonction max R,B pour traitement des panneaux
    """    
    image_test = image.copy()[:,:,0]
    sum_channels = np.sum(image, axis=2)
    sum_channels[sum_channels == 0] = 1
    
    red_fraction = image[:, :, 0] / sum_channels
    blue_fraction = image[:, :, 2] / sum_channels
    
    max_fraction = np.maximum(red_fraction, blue_fraction)
    max_fraction = np.clip(max_fraction, 0, 1)
    
    # Accentuer les contrastes
    max_fraction = np.power(max_fraction, 0.5)  
    
    image_test = (max_fraction * 255).astype(np.uint8)
    return image_test

def do_gray_mser(image):
    
    # Détecter les MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(image)
    
    # Filtrer les régions en fonction des caractéristiques
    MIN_AREA = 1500
    MAX_AREA = 200000
    MIN_ASPECT_RATIO = 0.33
    MAX_ASPECT_RATIO = 1.333
    
    filtered_regions = []
    for region in regions:
        area = cv2.contourArea(region)
        rect = cv2.minAreaRect(region)
        width, height = rect[1]
        aspect_ratio = float(width) / (height+1)
        if MIN_AREA < area < MAX_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            x, y, w, h = cv2.boundingRect(region)
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            filtered_regions.append(expand_region(x1, y1, x2, y2, 1.25, image.shape[1], image.shape[0]))

    return filtered_regions


def sliding_window(image, gray, binary_image, initial_window_size, step_size, min_window_size=20):
    """
    Algorithme fenetre glissante
    """
    height, width = image.shape[:2]
    coordinates = []

    def contains_zero(region):
        return np.any(region == 0)

    window_size = initial_window_size

    while window_size >= min_window_size:
        for y in range(0, height - window_size + 1, step_size):
            for x in range(0, width - window_size + 1, step_size):
                region = binary_image[y:y + window_size, x:x + window_size]
                if contains_zero(region):
                    coordinates.append((x, y, x + window_size, y + window_size))
        window_size //= 2  # Réduire la taille de la fenêtre par deux à chaque itération

    return coordinates



def binary_processing(image):
        '''
        Fonction effectuant un traitement binaire sur l'image et renvoyant l'ensemble des régions d'intéret'
        '''
        original_height, original_width = image.shape[:2]
        img_gray = convert_to_max_grayscale(image)
        regions = []
        #Differents traitement pour extraire les formes des images binaires
        (thresh, binary_image2) = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        (thresh, im_bw) = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_or(otsu, adaptive)
        binary_image= cv2.bitwise_or(binary_image, binary_image2)

        
        #Extraction des contours
        contours3, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2,_ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours2+contours+contours3:
            area = cv2.contourArea(contour)/(image.shape[0]*image.shape[1])
        
            x_red, y_red, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if area > 0.0005 and area < 0.8 and aspect_ratio < 1.33 and aspect_ratio >0.3:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, x + w, y + h))
                
        #Agrandissement des régions
        for j,r in enumerate(regions) :
            x1,y1,x2,y2 = r
            regions[j] = expand_region(x1, y1, x2, y2,1.2, image.shape[1], image.shape[0])
        return regions
    

def process_box_hog(img , boxes):
    '''
    Fonction mettant toutes les ROIS sous forme de vecteur hog
    
    Parameters
    ----------
    img : image rgb
    boxes : all ROIS

    Returns
    -------
    hog_features : vecteur 1568
    '''
    fd=[]
    n=4
    for i in range(len(boxes)):
        bb = boxes[i]
        if bb != []:
            box = (img[bb[1]:bb[3],bb[0]:bb[2],:])
            box = cv2.resize(box, (40,40))
            fd_k, hog_k = hog(box, orientations=8, pixels_per_cell=(5,5), cells_per_block=(2, 2),multichannel=True, visualize=True)
            fd.append(fd_k)

    hog_features = np.array(fd)
    data = np.hstack(hog_features)
    return hog_features