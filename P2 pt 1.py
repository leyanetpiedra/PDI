#Importar bibliotecas
import cv2
import numpy as np
import matplotlib.pyplot as plt

### EJERCICIO 1: TRASLACIÓN
imagen = cv2.imread(r"C:\Users\Leyanet Piedra\Desktop\pdi lab\practica 2\img_radiografia.jpg", cv2.IMREAD_GRAYSCALE) #Carga la imagen en escala de grises
ancho, alto = imagen.shape #Número de filas y columnas de la imagen

tx, ty = 50, 30  # Desplazamiento en X e Y
#Matriz de traslación | 1   0   tx | , | 0   1   ty |
M1 = np.float32([[1, 0, 50], [0,1,30]])

#Aplica la traslación a la imagen con la matriz M2
imagen_trasladada = cv2.warpAffine(imagen, M1, (ancho, alto)) 

#Matriz de traslación

tz, te = 100.5, 100.7
M2 = np.float32([[1, 0, tz], [0,1,te]])
imagen_trasladada_decimal = cv2.warpAffine(imagen, M2, (ancho, alto)) #Aplica la traslación a la imagen

plt.figure(figsize=(10,5)) #Define el tamaño de la figura para mostrar las imagenes
plt.subplot(1,3,1), plt.imshow(imagen, cmap='gray'), plt.title('Foto Original'), plt.axis("off") #Muestra la imagen original
#Muestra la imagen trasladada ambas veces
plt.subplot(1,3,2), plt.imshow(imagen_trasladada, cmap='gray'), plt.title('Imagen Trasladada x=50 y=30') , plt.axis("off")
plt.subplot(1,3,3), plt.imshow(imagen_trasladada_decimal, cmap='gray'), plt.title('Imagen Trasladada x=100.5 y=100.7') , plt.axis("off")

### EJERCICIO 2: ROTACIÓN
#Crea la matriz de rotación para girar la imagen 45 grados (en contra del reloj y alrededor del centro)
angulo = 45
centro = (ancho // 2, alto // 2)  #Centro de la imagen
escala = 1.0 #Sin cambios en el tamaño
rotacion = cv2.getRotationMatrix2D(centro, angulo, escala)
imagen_rotada = cv2.warpAffine(imagen, rotacion, (ancho, alto)) #Aplica la rotación a la imagen


plt.figure(figsize=(10,5)) #Define el tamaño de la figura para mostrar las imagenes
plt.subplot(1,3,1), plt.imshow(imagen, cmap='gray'), plt.title('Foto Original'), plt.axis("off") #Muestra la imagen original
plt.subplot(1,3,2), plt.imshow(imagen_rotada, cmap='gray'), plt.title('Imagen Rotada 45º'), plt.axis("off") #Muestra la imagen rotada


### EJERCICIO 3: ESCALA
#Redimensionamos la imagen usando la funcion (imagen, tamaño de salida, escala en x, escala en y, método de interpolación) 
escala_150 = cv2.resize(imagen, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
escala_50 = cv2.resize(imagen, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) #Redimensiona la imagen al 50% del tamaño original

plt.figure(figsize=(10,5)) #Define el tamaño de la figura para mostrar las imagenes

plt.subplot(1,3,1), plt.imshow(imagen, cmap='gray'), plt.title('Foto Original') #Muestra la imagen original
plt.subplot(1,3,2), plt.imshow(escala_150, cmap='gray'), plt.title('Imagen a escala 150') #Muestra la imagen escalada al 150%
plt.subplot(1,3,3), plt.imshow(escala_50, cmap='gray'), plt.title('Imagen a escala 50') #Muestra la imagen escalada al 50%


#Muestra todas las figuras generadas
plt.show()