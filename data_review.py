import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU encontrada ou configurada. Usando CPU.")

print(tf.config.list_physical_devices('GPU'))

os.makedirs('data/images', exist_ok=True)

images = tf.data.Dataset.list_files('data/images/*.jpg')
print("Caminho da primeira imagem:", images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (256, 256))  
    return img

images = images.map(load_image)

print("Primeira imagem carregada (shape):", images.as_numpy_iterator().next().shape)
print("Tipo do dataset de imagens:", type(images))

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=len(plot_images), figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image.astype('uint8'))
    ax[idx].axis('off')

#verificar qual a treta depois, não está abrindo a janela
plt.savefig("preview.png")
print("Imagem salva em preview.png")