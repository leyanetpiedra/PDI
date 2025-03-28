#Importar bibliotecas
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread(r"C:\Users\Leyanet Piedra\Desktop\pdi lab\practica 2\img_lowcontrast.jpg", cv2.IMREAD_GRAYSCALE) #Carga la imagen original en escala de grises


#Utilizar la función para encontrar el histograma de la imagen original que muestra la distribución de los valores de los píxeles en una imagen
#256 bins, lo que corresponde a los  valores de intensidad para una imagen en escala de grises de 8 bits (de 0 a 255)
#[0, 256], especifica el rango de valores de píxel que se deben considerar al calcular el histograma de una img en gris
histograma_original = cv2.calcHist([imagen], [0], None, [256], [0, 256])

#Aplicar ecualización a la imagen
imagen_ecualizada = cv2.equalizeHist(imagen)

#Obtener el histograma de la imagen ecualizada
histograma_ecualizada = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])

# Visualizar la imagen original y la ecualizada
plt.figure(figsize=(15, 5))

plt.subplot(2, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(imagen_ecualizada, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

# Mostrar histograma de la imagen original
plt.subplot(2, 2, 3)
plt.plot(histograma_original, color='gray')
plt.title('Histograma Original')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Cantidad de píxeles')

# Mostrar histograma de la imagen ecualizada
plt.subplot(2, 2, 4)
plt.plot(histograma_ecualizada, color='gray')
plt.title('Histograma Ecualizado')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Cantidad de píxeles')

plt.tight_layout()
plt.show()



