import tensorflow as tf
import os
import json 
import numpy as np


def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

print("Carregando imagens de treino aumentadas...")
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x / 255.0)


print("Carregando imagens de teste aumentadas...")
test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x / 255.0)


print("Carregando imagens de validação aumentadas...")
val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x / 255.0)


print("Shape da primeira imagem de treino (normalizada):", train_images.as_numpy_iterator().next().shape)
print("Carregamento de imagens concluído.")


def load_labels(label_path):

    label_path_str = label_path.numpy().decode('utf-8')
    with open(label_path_str, 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

print("Carregando rótulos de treino aumentados...")
train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)

train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))


print("Carregando rótulos de teste aumentados...")
test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))


print("Carregando rótulos de validação aumentados...")
val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float32]))


first_train_label = train_labels.as_numpy_iterator().next()
print("Primeiro rótulo de treino (classe, bbox):", first_train_label)
print("Carregamento de rótulos concluído.")