import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model 

longitud, altura = 100, 100 
modelo = './modelo/modelo.h5'
pesos= './modelo/pesos.h5'

cnn = tf.keras.models.load_model(modelo, custom_objects=None, compile=True)
cnn.load_weights(pesos)

def predict(file):
    x=load_img(file, target_size=(longitud, altura))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    arreglo=cnn.predict(x)
    resultado=arreglo[0]
    respuesta=np.argmax(resultado)
    if respuesta == 0:
        print('Billete de 10 euros')
    if respuesta==1:
        print('Billete de 100 euros')
    if respuesta == 2:
        print('Billete de 20 euros')
    if respuesta == 3:
        print('Billete de 200 euros')
    if respuesta == 4:
            print('Billete de 5 euros')
    if respuesta == 5:
        print('Billete de 50 euros')
    if respuesta == 6:
        print('Billete de 500 euros')
    return respuesta


predict('descarga.jpg')

predict('images (1).jpg')

predict('500.1.jpg')

predict('200.5.jpg')


