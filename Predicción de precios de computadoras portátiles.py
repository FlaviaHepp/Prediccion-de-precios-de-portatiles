
#Importamos librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


#Leer datos:
df = pd.read_csv('laptop_price - dataset.csv')
df.head(5)

#Verificar datos:
df.info()

df.isna().sum()

print(f'Valores duplicados: {df.duplicated().sum().item()}')

print(df.describe(include='all'))

compania = df['Company'].value_counts()

compania = compania.sort_values(ascending=True)
compania.plot(kind='barh', edgecolor='violet', title='Fabricante de ordenadores portátiles\n')
plt.xticks(rotation=0)
pd.DataFrame(compania.sort_values(ascending=False))

#Tipo Nombre
Tipo_Nombre = df['TypeName'].value_counts()

Tipo_Nombre.plot(kind='bar', color='blue', edgecolor='white', title='Tipo y Nombre\n')
plt.xticks(rotation=0)
pd.DataFrame(Tipo_Nombre)

#Pulgadas
Pulgadas = df['Inches'].value_counts()

Inches = Pulgadas.sort_values(ascending=True)
Inches.plot(kind='barh', color='green', edgecolor='lime', title='Tamaño de pantalla\n')
plt.xticks(rotation=0)
pd.DataFrame(Inches.sort_values(ascending=False))

#Resolución de pantalla
resolucion = df['ScreenResolution'].value_counts()

plt.figure(figsize=(8,10))
resolucion = resolucion.sort_values(ascending=True)
resolucion.plot(kind='barh', color='orange', edgecolor='white', title='Resolución de pantalla\n')
plt.xticks(rotation=45)
pd.DataFrame(resolucion.sort_values(ascending=False))

#CPU_Compañía
CPU_compania = df['CPU_Company'].value_counts()

CPU_compania.plot(kind='bar', color='aquamarine', edgecolor='white', width=0.3, title='Fabricante de CPU\n')
plt.xticks(rotation=0)
pd.DataFrame(CPU_compania)

#Frecuencia de CPU (GHz)
frecuencia = df['CPU_Frequency (GHz)'].value_counts()

plt.figure(figsize=(7,7))
freceuncia = frecuencia.sort_values(ascending=True)
frecuencia.plot(kind='barh', color='deepskyblue', edgecolor='white', title='Frecuencia de CPU (Ghz)\n')
plt.xticks(rotation=0)
pd.DataFrame(frecuencia.sort_values(ascending=False))

#RAM
RAM = df['RAM (GB)'].value_counts()

RAM.plot(kind='bar', color='chartreuse', edgecolor='white')
plt.xticks(rotation=0)
pd.DataFrame(RAM)

#Memoria
memoria = df['Memory'].value_counts()

plt.figure(figsize=(8,10))
memoria = memoria.sort_values(ascending=True)
memoria.plot(kind='barh', color='blueviolet', edgecolor='white', title= "Memoria\n")
plt.xticks(rotation=0)
pd.DataFrame(memoria.sort_values(ascending=False))

#GPU_Compañía
GPU_Compania = df['GPU_Company'].value_counts()

GPU_Compania.plot(kind='bar', color='fuchsia', edgecolor='white', width=0.3, title='Fabricante de GPU\n')
plt.xticks(rotation=0)
pd.DataFrame(GPU_Compania)

#Sistemas operativos
SO = df['OpSys'].value_counts()

plt.figure(figsize=(10,4))
SO.plot(kind='bar', color='gold', edgecolor='white', title='Sistema operativo\n')
plt.xticks(rotation=0)
pd.DataFrame(SO)

#Peso (kg)
peso = df['Weight (kg)']

sns.histplot(peso, kde=True)
plt.xticks(rotation=0)
pd.DataFrame(peso.describe())

#Precio (Euro)
precio = df['Price (Euro)']

sns.histplot(precio, kde=True)
plt.xticks(rotation=0)
pd.DataFrame(precio.describe())

df[df['Price (Euro)'] == 174]

#Análisis de datos colectivos:
#Empresa y precio (euros)
sns.barplot(x = df['Company'], y = df['Price (Euro)'])
plt.title('Empresa y precio (Euro)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Tipo Nombre y precio (Euro)
sns.barplot(x = df['TypeName'], y = df['Price (Euro)'])
plt.title('Tipo y precio (Euro)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Pulgadas y precio (euros)
sns.barplot(x = df['Inches'], y = df['Price (Euro)'])
plt.title('Pulgadas y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Resolución de pantalla y precio (euros)
plt.figure(figsize=(15,6))

sns.barplot(x = df['ScreenResolution'], y = df['Price (Euro)'])
plt.title('Resolución de pantalla y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#CPU y precio (euros)
sns.barplot(x = df['CPU_Company'], y = df['Price (Euro)'])
plt.title('CPU_Empresa y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

plt.figure(figsize=(12,6))

sns.barplot(x = df['CPU_Frequency (GHz)'], y = df['Price (Euro)'])
plt.title('Frecuencia de CPU (GHz) y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#RAM (GB) y Precio (Euros)
sns.barplot(x = df['RAM (GB)'], y = df['Price (Euro)'])
plt.xticks(rotation=0)
plt.title('RAM (GB) y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Memoria y precio (euros)
plt.figure(figsize=(10,6))
sns.barplot(x = df['Memory'], y = df['Price (Euro)'])
plt.title('Memoria y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#GPU y precio (euros)
sns.barplot(x = df['GPU_Company'], y = df['Price (Euro)'])
plt.title('GPU Compañía y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Sistema operativo y precio (euros)
sns.barplot(x = df['OpSys'], y = df['Price (Euro)'])
plt.title('Sistema operativo y precio (euros)\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Preprocesamiento:
x = df.corr(numeric_only=True)

sns.heatmap(x, annot=True, cmap='cool')
plt.show()

df.sample()

#Codificación
encoder = LabelEncoder()

df['Company'] = encoder.fit_transform(df['Company'])
df['Product'] = encoder.fit_transform(df['Product'])
df['TypeName'] = encoder.fit_transform(df['TypeName'])
df['ScreenResolution'] = encoder.fit_transform(df['ScreenResolution'])
df['CPU_Company'] = encoder.fit_transform(df['CPU_Company'])
df['CPU_Type'] = encoder.fit_transform(df['CPU_Type'])
df['Memory'] = encoder.fit_transform(df['Memory'])
df['GPU_Company'] = encoder.fit_transform(df['GPU_Company'])
df['GPU_Type'] = encoder.fit_transform(df['GPU_Type'])
df['OpSys'] = encoder.fit_transform(df['OpSys'])
df.head(5)

#Dividir entrada y salida
x = df.iloc[ : , 0:-1]
y = df.iloc[ : , -1]
x

y

#Escalada
scaler = MinMaxScaler()

x = scaler.fit_transform(x)
x

#Convertir a DataFrame
y_df = y.to_frame()

#Ajusta y transforma y
y = scaler.fit_transform(y_df)
y

#Entrenar y probar (dividir):
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

#Modelo ML:
#Regresión lineal
model_1 = LinearRegression()
model_1.fit(x_train,y_train)

y_pred = model_1.predict(x_test)
print(f'Error al cuadrado:\n {mean_squared_error(y_test, y_pred) * 100}%')

#Regresor SGD
model_2 = SGDRegressor()
model_2.fit(x_train,y_train)

y_pred = model_2.predict(x_test)
print(f'Error al cuadrado:\n {mean_squared_error(y_test, y_pred) * 100}%')

#Lasso
model_3 = Lasso()
model_3.fit(x_train,y_train)

y_pred = model_3.predict(x_test)
print(f'Error al cuadrado:\n {mean_squared_error(y_test, y_pred) * 100}%')

#Ridge
model_4 = Ridge()
model_4.fit(x_train,y_train)

y_pred = model_4.predict(x_test)
print(f'Error al cuadrado:\n {mean_squared_error(y_test, y_pred) * 100}%')

#Red elástica
model_5 = ElasticNet()
model_5.fit(x_train,y_train)

y_pred = model_5.predict(x_test)
print(f'Error al cuadrado:\n {mean_squared_error(y_test, y_pred) * 100}%')

#Puntuación del modelo:
#Lista de modelos
models = [model_1, model_2, model_3, model_4, model_5]
model_names = ['LinearRegression', 'SGDRegressor', 'Lasso', 'Ridge', 'ElasticNet']

#Calcular puntuaciones de trenes y pruebas
train_scores = [f'{round(model.score(x_train, y_train), 2) * 100} %' for model in models]
test_scores = [f'{round(model.score(x_test, y_test), 2) * 100} %' for model in models]

#Crear marco de datos
model_score = pd.DataFrame({
    'Model': model_names,
    'Train score': train_scores,
    'Test score': test_scores,
})

model_score

#Convertir columnas de RAM y Peso a numéricas
df['RAM (GB)'] = df['RAM (GB)'].astype(int)
df['Weight (kg)'] = df['Weight (kg)'].astype(float)

#Análisis de correlación
#Veamos cómo se correlacionan entre sí las distintas características numéricas. Esto puede darnos información sobre qué características 
#podrían ser importantes para predecir el precio de una computadora portátil.
#Seleccione solo columnas numéricas para el análisis de correlación
numeric_df = df.select_dtypes(include=[np.number])

#Generar un mapa de calor de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='spring', fmt='.2f')
plt.title('Mapa de calor de correlación\n', fontsize = 16, fontweight = 'bold')
plt.show()

#Ingeniería de características
#Crearemos algunas características nuevas a partir de las columnas existentes para ayudar a mejorar nuestro modelo predictivo. 
#Por ejemplo, podemos extraer las dimensiones de resolución de pantalla y calcular la cantidad total de píxeles.
#Extraer las dimensiones de resolución de pantalla
df['ScreenWidth'] = df['ScreenResolution'].str.extract(r'(\d+)x')[0].astype(int)
df['ScreenHeight'] = df['ScreenResolution'].str.extract(r'x(\d+)')[0].astype(int)

#Calcular el número total de píxeles
df['TotalPixels'] = df['ScreenWidth'] * df['ScreenHeight']

#Modelado predictivo
#Ahora, construyamos un modelo predictivo para estimar el precio de una computadora portátil en función de sus especificaciones. Para ello, utilizaremos un modelo de regresión lineal simple.
#Definir las características y la variable objetivo
features = ['Inches', 'CPU_Frequency (GHz)', 'RAM (GB)', 'Weight (kg)', 'TotalPixels']
X = df[features]
y = df['Price (Euro)']

#Dividir los datos en conjuntos de entrenamiento y prueb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#Hacer predicciones sobre el conjunto de pruebas
y_pred = modelo.predict(X_test)

#Calcular la precisión de la predicción
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMSE:\n", mse, "\nr2:\n", r2)
