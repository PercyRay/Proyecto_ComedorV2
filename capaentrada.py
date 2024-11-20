import cv2 as cv
import numpy as np
import os
import imutils

modelo='FotosKevin'
ruta1='C:/Proyectos Python/curso/Curso_Udemy'
rutacompleta = ruta1 + '/'+ modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)
   

ruidos=cv.CascadeClassifier('C:/Proyectos Python/curso/Curso_Udemy/entrenamientosopencv ruidos//opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')
print(ruidos)
camara=cv.VideoCapture(0)
id=0

while True: 
   respuesta,captura=camara.read()
   if respuesta==False:break
   captura=imutils.resize(captura,width=640)
   grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
   idcaptura=captura.copy()

   cara=ruidos.detectMultiScale(grises,1.3,5)

   for(x,y,e1,e2) in cara: 
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0), 2) 
        rostrocap=idcaptura[y:y+e2,x:x+e1]
        rostrocap=cv.resize(rostrocap, (160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocap)
        id=id+1

   cv.imshow("Resultado rostro",captura)

   if id==200:
      break
camara.release()
cv.destroyAllWindows()


