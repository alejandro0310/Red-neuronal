import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import numpy as np
import h5py
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from math import sqrt

import os 

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'


#cnn = load_model(modelo, custom_objects={'mean_squared_abs_error': mean_squared_abs_error})
cnn = tf.keras.models.load_model(modelo, custom_objects=None, compile=True)
cnn.load_weights(pesos_modelo)

def predict(file):
    x=load_img(file, target_size=(longitud, altura))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    arreglo=cnn.predict(x)
    resultado=arreglo[0]
    respuesta=np.argmax(resultado)
    if respuesta == 0:
         print("pred: lapiz")
         Label(miframe, text="Pred: Lapiz  ").grid(row=4, column=1)
    elif respuesta==1:
        print("pred: Botella")
        Label(miframe, text="Pred: Botella  ").grid(row=4, column=1)
    return respuesta

def Boton():  
    ftypes = [('image file',"*.jpeg jpg png")]
    ttl  = "Title"
    dir1 = 'Desktop:\\red'
    root.fileName = askopenfilename(filetypes = ftypes, initialdir = dir1, title = ttl)
    print (root.fileName)
    a = os.path.basename(root.fileName)
    path = a
    resz = Image.open(path)
    resz = resz.resize((284, 178), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(resz)
    panel = Label(root, image=img)
    panel.photo = img
    panel.grid(column=1,row=1)
    #img2= Image.open("C:/Users/New/Desktop/Red2.0/inicio.jpg")
    #img2 = img2.resize((284, 178), Image.ANTIALIAS)
    #photo1 = ImageTk.PhotoImage(img2)
    #label = Label(miframe, image = photo1).grid(row=1, column=1)
    print(os.path.basename(a))

def avanza():
    #predim = mystring.get()
    #print(predim)
    a = os.path.basename(root.fileName)
    print(a)
    predict(a)
    

root = Tk()
root.title("Red Neuronal")
mystring =tk.StringVar(root)

miframe= Frame(width = 450, height=50).grid(rowspan=4, columnspan=3)

imgo= Image.open("C:/Users/New/Desktop/Red2.0/inicio.jpg")
photo = ImageTk.PhotoImage(imgo)
Label(miframe, image = photo).grid(row=1, column=1)

Label(miframe, text="Escoja su imagen").grid(row=2, column=1)
#entry_1 = Entry(miframe, textvariable = mystring).grid(row=2, column=2)
w = Button ( miframe, text ="Buscar Imagen", command = Boton).grid(row=2, column=3)
#w = Button ( miframe, text ="Abrir Imagen", command = Boton).grid(row=2, column=2)

D = Button ( miframe, text ="Analizar imagen", command = avanza).grid(row=3, column=3)

Label(miframe, text="Resultado:  ").grid(row=4, column=1)


root.mainloop()

