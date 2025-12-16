# Face Recognition using LBP and SVM

Projet de reconnaissance faciale utilisant les caractéristiques Local Binary Pattern (LBP) et un classifieur SVM (Support Vector Machine).

## Description

Ce projet implémente un système de reconnaissance faciale qui :
- Détecte les visages dans une image en utilisant les cascades de Haar d'OpenCV
- Extrait les caractéristiques LBP (Local Binary Pattern) des visages détectés
- Classe les visages à l'aide d'un modèle SVM entraîné sur un dataset personnalisé

## Fonctionnalités

- **Détection de visages** : Utilisation de `haarcascade_frontalface_default.xml` pour détecter les visages dans une image
- **Extraction de caractéristiques LBP** : Division de l'image en régions et calcul des histogrammes LBP pour chaque région
- **Classification SVM** : Entraînement d'un modèle SVM linéaire pour la reconnaissance faciale
- **Prédiction** : Prédiction de l'identité d'un visage à partir d'une image

## Structure du projet

```
.
├── main.ipynb                          # Notebook principal avec le code complet
├── haarcascade_frontalface_default.xml # Classificateur Haar pour la détection de visages
├── Dataset EquipeMaroc/                # Dataset d'entraînement
└── README.md                           # Ce fichier
```

## Prérequis

Les bibliothèques Python suivantes sont nécessaires :

```bash
pip install opencv-python
pip install numpy
pip install scikit-image
pip install scikit-learn
pip install matplotlib
pip install imutils
pip install pillow
```

## Utilisation

### 1. Préparation du dataset

Organisez vos images dans une structure de dossiers où chaque sous-dossier représente une classe (personne) :

```
Dataset EquipeMaroc/
├── personne1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── personne2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 2. Entraînement du modèle

Exécutez les cellules du notebook `main.ipynb` pour :
- Charger les images du dataset
- Extraire les caractéristiques LBP
- Entraîner le modèle SVM
- Évaluer les performances

### 3. Prédiction

Utilisez le modèle entraîné pour prédire l'identité d'un visage dans une nouvelle image :

```python
# Charger l'image
image = cv2.imread("detected_face_1.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (200, 200))

# Extraire les caractéristiques LBP
lbp_features = extract_lbp_features_images(image, N_POINTS, RADIUS, 10, 10)

# Standardiser et prédire
lbp_features_scaled = scaler.transform(lbp_features.reshape(1, -1))
prediction = classifier.predict(lbp_features_scaled)
```

## Paramètres LBP

- **RADIUS** : Rayon du voisinage (par défaut : 4)
- **N_POINTS** : Nombre de points du voisinage (par défaut : 16)
- **METHOD** : Méthode LBP utilisée (par défaut : 'uniform')
- **num_rows/num_cols** : Nombre de régions pour diviser l'image (par défaut : 10x10)

## Résultats

Le modèle atteint une précision d'environ **90%** sur le dataset de test avec les paramètres par défaut.

## Auteur

Projet réalisé dans le cadre du cours de Vision par Ordinateur (Computer Vision).

## Licence

Ce projet est fourni à des fins éducatives.

