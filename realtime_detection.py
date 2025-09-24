import cv2
import tensorflow as tf
import numpy as np
import time
import os
from collections import deque

MODEL_PATH = 'models/final_face_detector_model.h5'
IMG_WIDTH, IMG_HEIGHT = 120, 120
CONFIDENCE_THRESHOLD = 0.75
FPS_DISPLAY_INTERVAL = 1
SMOOTHING_FRAMES = 5
WINDOW_NAME = "Detecção de Face em Tempo Real - Pressione Q para sair"


if not os.path.exists(MODEL_PATH):
    print(f"❌ Erro: Modelo não encontrado em {MODEL_PATH}.")
    print("➡️ Verifique o caminho e se o modelo foi salvo corretamente.")
    exit(1)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modelo carregado com sucesso: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    print("➡️ Verifique compatibilidade de versões do TensorFlow/Keras.")
    exit(1)

bbox_history = deque(maxlen=SMOOTHING_FRAMES)



def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Pré-processa o frame para entrada no modelo."""
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def draw_bbox_and_info(frame, bbox_coords, confidence, img_width, img_height, color=(0, 255, 0)):
    """Desenha a bounding box e informações de confiança no frame."""
    try:
        x_min, y_min, x_max, y_max = np.multiply(bbox_coords, [img_width, img_height, img_width, img_height]).astype(int)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

        text = f"Confiança: {confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(0, x_min)
        text_y = max(text_size[1] + 5, y_min - 10)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return frame, (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"⚠️ Erro ao desenhar bounding box: {e}")
        return frame, None


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("\n" + "=" * 50)
        print("❌ ERRO: Não foi possível acessar a câmera!")
        print("➡️ Verifique:")
        print("1. Se a câmera está conectada.")
        print("2. Se outro programa não está usando a câmera.")
        print("3. Se você tem permissão (use: sudo usermod -a -G video $USER && reboot).")
        print("4. Se os drivers estão instalados.")
        print("5. Tente trocar o índice da câmera (cv2.VideoCapture(1), etc.).")
        print("=" * 50 + "\n")
        return

    print("📷 Câmera aberta com sucesso. Pressione 'q' para sair.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Aviso: Não foi possível capturar frame da câmera.")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        input_frame = preprocess_frame(frame)

        predictions = model.predict(input_frame, verbose=0)

        class_prediction, bbox_prediction = None, None
        if isinstance(predictions, list) and len(predictions) == 2:
            class_prediction = predictions[0][0][0]
            bbox_prediction = predictions[1][0]
        elif isinstance(predictions, np.ndarray) and predictions.shape[-1] == 5:
            class_prediction = predictions[0][0]
            bbox_prediction = predictions[0][1:]
        else:
            print("⚠️ Saída inesperada do modelo:", predictions)
            break

        detected_bbox_on_frame = None

        if class_prediction is not None and class_prediction > CONFIDENCE_THRESHOLD:
            bbox_history.append(bbox_prediction)
            smoothed_bbox = np.mean(bbox_history, axis=0)

            frame, detected_bbox_on_frame = draw_bbox_and_info(
                frame, smoothed_bbox, class_prediction, frame_width, frame_height, color=(0, 255, 0)
            )
            cv2.putText(frame, "Face Detectada!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            bbox_history.clear()
            cv2.putText(frame, "Aguardando Face...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        fps_frame_count += 1
        if (time.time() - fps_start_time) > FPS_DISPLAY_INTERVAL:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0

        cv2.putText(frame, f"FPS: {int(current_fps)}", (frame_width - 180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Detecção de face interrompida pelo usuário.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detecção de face finalizada.")


if __name__ == '__main__':
    main()