
import cv2 as cv
import numpy as np
import os
dataRuta='C:/Proyectos Python/curso/Curso_Udemy/data'
listaData=os.listdir(dataRuta)
entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')
ruidos=cv.CascadeClassifier('C:/Proyectos Python/curso/Curso_Udemy/entrenamientosopencv ruidos//opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)
while True:
    _,captura=camara.read()
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara: 
        rostrocap=idcaptura[y:y+e2,x:x+e1]
        rostrocap=cv.resize(rostrocap, (160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocap)
        cv.putText(captura, '{}'.format(resultado),(x,y-5),1,1.3,(0,255,0),1,cv.LINE_AA)
        if resultado[1]<9000:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]),(x,y-20),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2),(255,0,0),2)
        else:
            cv.putText(captura, 'No encontrado'.format(listaData[resultado[0]]),(x,y-20),2,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2),(255,0,0),2)

    cv.imshow("Resultados", captura)
    if cv.waitKey(1)==ord('s'):
        break
camara.release()
cv.destroyAllWindows()