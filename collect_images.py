import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data', 'images')
os.makedirs(IMAGES_PATH, exist_ok=True)
number_images = 30
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera. Verifique se ela está conectada e não está em uso por outro aplicativo.")
else:
    print("Câmera aberta. Preparado para coletar imagens. Pressione 'q' para sair.")
    for imgnum in range(number_images):
        ret, frame = cap.read()
        if not ret:
            print("Erro: não foi possível ler o frame da câmera. Tentando novamente...")
            time.sleep(1)
            continue
        
        cv2.putText(frame, f'Coletando imagem {imgnum + 1}/{number_images}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Coletando imagens - Pressione Q para parar', frame)

        imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        print(f'Imagem {imgnum + 1} coletada: {imgname}')
        
        time.sleep(1) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Coleta de imagens interrompida pelo usuário.')
            break

cap.release()
cv2.destroyAllWindows()
print('Coleta de imagens finalizada.')