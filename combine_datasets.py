import tensorflow as tf
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU encontrada ou configurada. Usando CPU.")


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    label_path_str = label_path.numpy().decode('utf-8')
    with open(label_path_str, 'r', encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']

print("Preparando datasets para treino, teste e validação...")

# carrega as imagens
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image).map(lambda x: tf.image.resize(x, (120, 120))).map(lambda x: x / 255.0)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image).map(lambda x: tf.image.resize(x, (120, 120))).map(lambda x: x / 255.0)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image).map(lambda x: tf.image.resize(x, (120, 120))).map(lambda x: x / 255.0)

# carrega os rótulos
train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))


print(f"Número de imagens de treino: {len(list(train_images.as_numpy_iterator()))}")
print(f"Número de rótulos de treino: {len(list(train_labels.as_numpy_iterator()))}")
print(f"Número de imagens de teste: {len(list(test_images.as_numpy_iterator()))}")
print(f"Número de rótulos de teste: {len(list(test_labels.as_numpy_iterator()))}")
print(f"Número de imagens de validação: {len(list(val_images.as_numpy_iterator()))}")
print(f"Número de rótulos de validação: {len(list(val_labels.as_numpy_iterator()))}")



train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)

print("\nDatasets finais criados:")
print(f"Dataset de treino: {train}")
print(f"Dataset de teste: {test}")
print(f"Dataset de validação: {val}")


first_batch = train.as_numpy_iterator().next()
print("\nShape do primeiro batch (imagens):", first_batch[0].shape)
print("Shape do primeiro batch (classes):", first_batch[1][0].shape)
print("Shape do primeiro batch (bboxes):", first_batch[1][1].shape)


# imagens e anotações
print("\nVisualizando algumas amostras do dataset de treino...")
data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_class = res[1][0][idx]
    sample_coords = res[1][1][idx]

    display_image = (sample_image * 255).astype(np.uint8)
   

    x_min, y_min, x_max, y_max = np.multiply(sample_coords, [120, 120, 120, 120]).astype(int)

    if sample_class[0] == 1: 

        img_with_bbox = np.array(display_image)
        cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        ax[idx].imshow(img_with_bbox)
        ax[idx].set_title(f"Face (Classe: {sample_class[0]})")
    else:
        ax[idx].imshow(display_image)
        ax[idx].set_title(f"Sem Face (Classe: {sample_class[0]})")

    ax[idx].axis('off')

plt.tight_layout()
plt.savefig("dataset_samples.png")
print("Visualização dos datasets salva em dataset_samples.png")

plt.show()