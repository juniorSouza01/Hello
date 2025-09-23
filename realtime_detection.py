import cv2
import tensorflow as tf
import numpy as np
import time
import os

model_path = 'models/final_face_detector_model.h5' 
if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado em {model_path}. Por favor, treine o modelo primeiro.")
    exit()

model = tf.keras.models.load_model(model_path)
print(f"Modelo {model_path} carregado com sucesso.")

IMG_WIDTH = 120
IMG_HEIGHT = 120
CONFIDENCE_THRESHOLD = 0.8

def preprocess_frame(frame):

    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    normalized_frame = resized_frame / 255.0

    input_tensor = np.expand_dims(normalized_frame, axis=0)
    return input_tensor

def draw_bbox(frame, bbox_coords, class_label, img_width, img_height):

    x_min, y_min, x_max, y_max = np.multiply(bbox_coords, [img_width, img_height, img_width, img_height]).astype(int)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    

    text = f"Face: {class_label:.2f}"
    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Câmera aberta. Pressione 'q' para sair.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame da câmera.")
            break


        frame = cv2.flip(frame, 1)

        input_frame = preprocess_frame(frame)

        predictions = model.predict(input_frame)
        class_prediction = predictions[0][0][0] 
        bbox_prediction = predictions[1][0]


        if class_prediction > CONFIDENCE_THRESHOLD:

            frame = draw_bbox(frame, bbox_prediction, class_prediction, frame_width, frame_height)
            cv2.putText(frame, "Face Detectada!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Aguardando Face...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        fps_frame_count += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
            cv2.putText(frame, f"FPS: {int(fps)}", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detecção de Face em Tempo Real', frame)

 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detecção de face finalizada.")

if __name__ == '__main__':
    main()