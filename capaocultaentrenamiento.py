import cv2 as cv
import os
import numpy as np
from time import time
dataRuta='C:/Proyectos Python/curso/Curso_Udemy/data'
listaData=os.listdir(dataRuta)
ids=[]
rostroData=[]
id=0
tiempoInicial=time()
for fila in listaData:
    rutacompleta=dataRuta+'/'+ fila
    print('Iniciando lectura...')
    for archivo in os.listdir(rutacompleta):
        
        print('Imagenes: ',fila +'/'+archivo)

        ids.append(id)
        rostroData.append(cv.imread(rutacompleta+'/'+archivo,0))
        

    id=id+1
    tiempoFinalLectura=time()
    tiempoTotalLectura=tiempoFinalLectura-tiempoInicial
    print('Tiempo total:',tiempoTotalLectura)

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()
print('Iniciando el entrenamienoto...')
entrenamientoEigenFaceRecognizer.train(rostroData,np.array(ids))
tiempoFinalLectura=time()
tiempoTotalLectura=tiempoFinalLectura-tiempoTotalLectura
print('Tiempo entrenamiento total',tiempoTotalLectura)
entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')
print('Entrenamiento concluido')



