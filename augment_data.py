import albumentations as alb
import cv2
import json
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations',
                              label_fields=['class_labels']))


for folder in ['train','test','val']:
    os.makedirs(os.path.join('aug_data', folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join('aug_data', folder, 'labels'), exist_ok=True)

example_image_name = None
for f in os.listdir(os.path.join('data', 'train', 'images')):
    if f.endswith('.jpg'):
        example_image_name = f
        break

if example_image_name:
    print(f"Usando imagem de exemplo: {example_image_name}")
    img_path = os.path.join('data', 'train', 'images', example_image_name)
    label_path = os.path.join('data', 'train', 'labels', example_image_name.replace('.jpg', '.json'))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label = json.load(f)

        coords_raw = label['shapes'][0]['points']
        x1, y1 = coords_raw[0]
        x2, y2 = coords_raw[1]

        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1

        h, w, _ = img.shape
        coords = [x_min / w, y_min / h, x_max / w, y_max / h]
        print("Coordenadas normalizadas:", coords)

        augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

        if len(augmented['bboxes']) > 0:
            aug_bbox = augmented['bboxes'][0]
            img_h, img_w, _ = augmented['image'].shape
            x_min_aug, y_min_aug, x_max_aug, y_max_aug = np.multiply(
                aug_bbox, [img_w, img_h, img_w, img_h]).astype(int)

            aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            cv2.rectangle(aug_image_bgr, (x_min_aug, y_min_aug),
                          (x_max_aug, y_max_aug), (0, 0, 255), 2)

            # plt.imshow(cv2.cvtColor(aug_image_bgr, cv2.COLOR_BGR2RGB))
            # plt.title("Imagem Aumentada com Bounding Box")
            # plt.show() # Esta linha bloquearia a execução
        else:
            print("Nenhuma bbox após aumento (pode ter sido cortada).")
            # plt.imshow(augmented['image'])
            # plt.title("Imagem Aumentada (sem BBox)")
            # plt.show() # Esta linha bloquearia a execução

    else:
        print(f"Erro: Arquivo de rótulo não encontrado em {label_path}")
else:
    print("Nenhuma imagem JPG encontrada na pasta de treino para teste de aumento.")


print("\nIniciando o pipeline de aumento de dados...")
for partition in ['train', 'test', 'val']:
    print(f"Processando partição: {partition}")
    for image_file in os.listdir(os.path.join('data', partition, 'images')):
        if not image_file.endswith('.jpg'):
            continue

        img_path = os.path.join('data', partition, 'images', image_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        coords = [0, 0, 0.00001, 0.00001]
        label_filename = image_file.replace('.jpg', '.json')
        label_path = os.path.join('data', partition, 'labels', label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            if label['shapes']:
                coords_raw = label['shapes'][0]['points']
                x1, y1 = coords_raw[0]
                x2, y2 = coords_raw[1]

                h, w, _ = img_rgb.shape
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)

                if x_max <= x_min: x_max = x_min + 1
                if y_max <= y_min: y_max = y_min + 1

                if w > 0 and h > 0:
                    coords = [x_min / w, y_min / h, x_max / w, y_max / h]
        else:
            print(f"Aviso: rótulo {label_filename} não encontrado.")

        try:
            for x in range(60):  # Gerar 60 versões
                augmented = augmentor(image=img_rgb, bboxes=[coords], class_labels=['face'])

                aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join('aug_data', partition, 'images',
                                          f'{image_file.split(".")[0]}.{x}.jpg'), aug_image_bgr)

                annotation = {"image_original": image_file}
                if len(augmented['bboxes']) == 0:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0
                else:
                    annotation['bbox'] = augmented['bboxes'][0]
                    annotation['class'] = 1

                with open(os.path.join('aug_data', partition, 'labels',
                                       f'{image_file.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)
        except Exception as e:
            print(f"Erro ao aumentar {image_file}: {e}")

print("✅ Pipeline de aumento de dados concluído!")