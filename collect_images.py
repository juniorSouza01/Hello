import os
import time
import uuid
import cv2


IMAGES_PATH = os.path.join('data', 'images')
number_images = 30
cap = cv2.VideoCapture(0)



if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera")
else:
    print("Câmera aberta")
    for imgnum in range(number_images):
        print('coletando imagens {}'.format(imgnum + 1))
        ret, frame = cap.read()
        if not ret:
            print("Erro: não foi possível ler o frame")
            break


        imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Coletando imagens', frame)
        time.sleep(0.5) # só uma pause de uma imagem para outra

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Coleta de imagens interrompida')
            break


cap.release()
cv2.destroyAllWindows()
print('Coleta de imagens finalizada')