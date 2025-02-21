# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# Paso 1
# Descompriomir los archivos train y test
train = "files/input/train_default_of_credit_card_clients.csv"
test = "files/input/test_default_of_credit_card_clients.csv"

# Leer los archivos
train = pd.read_csv(train)
test = pd.read_csv(test)

# Renombrar la columna "default payment next month" a "default"
train.rename(columns={"default payment next month": "default"}, inplace=True)
test.rename(columns={"default payment next month": "default"}, inplace=True)

# Remover la columna "ID"
train.drop(columns=["ID"], inplace=True)
test.drop(columns=["ID"], inplace=True)

# Eliminar los registros con informacion no disponible
train.dropna(inplace=True)
test.dropna(inplace=True)

# Para la columna EDUCATION, valores > 4 indican niveles superiores

train["EDUCATION"] = train["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
test["EDUCATION"] = test["EDUCATION"].apply(lambda x: x if x <= 4 else 4)

# Paso 2
x_train = train.drop(columns=["default"])
y_train = train["default"]
x_test = test.drop(columns=["default"])
y_test = test["default"]

# Paso 3
# Crear el pipeline
# Definir las columnas categóricas y numéricas
categorical_features = train.select_dtypes(include=['object']).columns.tolist()
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Transformadores
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = MinMaxScaler()

# Preprocesador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

# Pipeline completo
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=3)),  # Selección de características
    ('classifier', LogisticRegression())
])

# Paso 4
# Optimizar hiperparametros
param_grid = {
    'feature_selection__k': [3, 5, 7],
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy')
grid_search.fit(x_train, y_train)

# Paso 5
# Guardar el modelo
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    f.write(gzip.compress(pickle.dumps(grid_search.best_estimator_)))

# Paso 6 y 7
# crear una función para calcular las métricas y la matriz de confusión

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, balanced_accuracy, recall, f1

def calculate_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"predicted_0": tn, "predicted_1": fp}, {"predicted_0": fn, "predicted_1": tp}

# Calcular las métricas y la matriz de confusión
metrics = {}

# Conjunto de entrenamiento
y_pred_train = grid_search.predict(x_train)
precision, balanced_accuracy, recall, f1 = calculate_metrics(y_train, y_pred_train)
cm_matrix = calculate_confusion_matrix(y_train, y_pred_train)
metrics["train"] = {
    "precision": precision,
    "balanced_accuracy": balanced_accuracy,
    "recall": recall,
    "f1_score": f1,
    "cm_matrix": cm_matrix
}

# Conjunto de prueba
y_pred_test = grid_search.predict(x_test)
precision, balanced_accuracy, recall, f1 = calculate_metrics(y_test, y_pred_test)
cm_matrix = calculate_confusion_matrix(y_test, y_pred_test)
metrics["test"] = {
    "precision": precision,
    "balanced_accuracy": balanced_accuracy,
    "recall": recall,
    "f1_score": f1,
    "cm_matrix": cm_matrix
}

# Guardar las métricas junto a la matriz de confusión en un archivo JSON
with open("files/output/metrics.json", "w") as f:
    for key, value in metrics.items():
        f.write(json.dumps({"type": "metrics", "dataset": key, **value}) + "\n")

# Fin del script