# Importation des bibliothèques nécessaires
# import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Description de l'application
st.write('''
# Application simple pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris
''')


# Fonction pour recueillir les entrées de l'utilisateur
def user_input():
    # Création des sliders dans la barre latérale
    sepal_length = st.sidebar.slider('La longeur du Sepal', 4.3, 7.9, 5.3)
    sepal_width = st.sidebar.slider('La largeur du Sepal', 2.0, 4.4, 3.3)
    petal_length = st.sidebar.slider('La longeur du Petal', 1.0, 6.9, 2.3)
    petal_width = st.sidebar.slider('La largeur du Petal', 0.1, 2.5, 1.3)

    # Création d'un dictionnaire avec les entrées de l'utilisateur
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    # Conversion du dictionnaire en DataFrame
    fleur_parametres = pd.DataFrame(data, index=[0])
    return fleur_parametres


# Appel de la fonction pour obtenir les entrées de l'utilisateur
df = user_input()

# Affichage des entrées de l'utilisateur
st.subheader('On veut trouver la catégorie de cette fleur')
st.write(df)

# Chargement du jeu de données Iris
iris = datasets.load_iris()

# Création et entraînement du modèle
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

# Prédiction de la catégorie de la fleur
prediction = clf.predict(df.values)

# Affichage de la prédiction
st.subheader("La catégorie de la fleur d'iris est:")
st.write(iris.target_names[prediction])