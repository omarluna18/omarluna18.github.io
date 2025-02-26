# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:56:41 2025

@author: OMAR.LUNA
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('C:/Users/omar.luna/Desktop/Ciencia de Datos/Data.csv')

dataset.head(10)
#Para seleccionar la última columna como array de numpy
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#### Tratamiento de los NAs
from sklearn.impute import SimpleImputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean',axis =0) "Obsoleto"
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer = SimpleImputer.fit(X[:,1:3])  "obsoleto"
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
#print(X)

#### Codificar los datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
# Convertimos la columna categórica en números
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Aplicamos OneHotEncoder para crear las variables Dummy
column_transformer = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],
    remainder='passthrough'  #para dejar las demas columnas tal cual
)

X = column_transformer.fit_transform(X)

#### Dividir el dataset en conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)

