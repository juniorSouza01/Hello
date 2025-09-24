import cv2
import tensorflow as tf
import numpy as np
import time
import os
from collections import deque

model_path = 'models/final_face_detector_model.h5'
IMG_WIDTH = 120
IMG_HEIGHT = 120
CONFIDENCE_THRESHOLD = 0.75
FPS_DISPLAY_INTERVAL = 1
SMOOTHING_FRAMES = 5


if not os.path.exists(model_path):
    print(f"Erro: Modelo não encontrado em {model_path}.")
    print("Por favor, verifique o caminho e se o modelo foi treinado e salvo corretamente.")
    exit()

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo '{model_path}' carregado com sucesso.")

except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o modelo foi salvo corretamente e se as versões do TensorFlow são compatíveis.")
    exit()

bbox_history = deque(maxlen=SMOOTHING_FRAMES)

def preprocess_frame(frame):

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0)
    return input_tensor

def draw_bbox_and_info(frame, bbox_coords, confidence, img_width, img_height, color=(0, 255, 0)):

    x_min, y_min, x_max, y_max = np.multiply(bbox_coords, [img_width, img_height, img_width, img_height]).astype(int)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

    text = f"Confianca: {confidence:.2f}"

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x_min
    text_y = y_min - 10 if y_min - 10 > text_size[1] else y_min + text_size[1] + 5
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return frame, (x_min, y_min, x_max, y_max)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n" + "="*50)
        print("ERRO: Não foi possível acessar a câmera!")
        print("Por favor, verifique os seguintes pontos:")
        print("1. A câmera está conectada e funcionando corretamente?")
        print("2. Outros aplicativos estão usando a câmera? Feche-os.")
        print("3. Você possui permissões para acessar a câmera? (Em Linux, pode ser necessário 'sudo usermod -a -G video $USER' e reiniciar.)")
        print("4. Drivers da câmera estão atualizados?")
        print("5. Tente mudar o índice da câmera em `cv2.VideoCapture(0)` para 1, 2, etc.")
        print("="*50 + "\n")
        return

    print("Câmera aberta com sucesso. Pressione 'q' para sair.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("AVISO: Não foi possível ler o frame da câmera. Tentando novamente...")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        input_frame = preprocess_frame(frame)

        predictions = model.predict(input_frame, verbose=0)

        if isinstance(predictions, list) and len(predictions) == 2:
            class_prediction = predictions[0][0][0]
            bbox_prediction = predictions[1][0]
        else:

            print("ATENÇÃO: A estrutura da saída do modelo não é a esperada (lista de 2 itens).")
            print(f"Saída do modelo: {type(predictions)}, Conteúdo (primeiro elemento): {predictions[0] if isinstance(predictions, list) and len(predictions)>0 else predictions}")

            if isinstance(predictions, np.ndarray) and predictions.shape[-1] == 5:
                class_prediction = predictions[0][0]
                bbox_prediction = predictions[0][1:]
            else:
                print("Não foi possível interpretar a saída do modelo. Verifique a estrutura.")
                break

        detected_bbox_on_frame = None

        if class_prediction > CONFIDENCE_THRESHOLD:

            bbox_history.append(bbox_prediction)
            smoothed_bbox = np.mean(bbox_history, axis=0)

            frame, detected_bbox_on_frame = draw_bbox_and_info(frame, smoothed_bbox, class_prediction, frame_width, frame_height, color=(0, 255, 0))
            cv2.putText(frame, "Face Detectada!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            bbox_history.clear()
            cv2.putText(frame, "Aguardando Face...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        fps_frame_count += 1
        if (time.time() - fps_start_time) > FPS_DISPLAY_INTERVAL:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0

        cv2.putText(frame, f"FPS: {int(current_fps)}", (frame_width - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detecção de Face em Tempo Real - Pressione Q para sair', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detecção de face interrompida pelo usuário.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detecção de face finalizada.")

if __name__ == '__main__':
    main()