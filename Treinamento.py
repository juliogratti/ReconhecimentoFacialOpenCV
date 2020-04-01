import cv2
import os
import numpy as np

eigenFace = cv2.face.EigenFaceRecognizer_create(num_components = 50, threshold = 0)
fisherFace = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComID():
    caminhos = [os.path.join('dataSet', f) for f in os.listdir('dataSet')]
    #print(caminhos)

    faces = []
    ids = []

    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face ", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComID()
#print(ids)

print("Treinando...")


eigenFace.train(faces, ids) #chama o metodo de treinamento levando como parâmetro as faces e os id's
eigenFace.write('classificadorEigen.yml')

fisherFace.train(faces, ids)
fisherFace.write('classificadorFischer.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado!")

