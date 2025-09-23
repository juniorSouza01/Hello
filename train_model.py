import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json
import numpy as np

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

IMG_WIDTH = 120
IMG_HEIGHT = 120
NUM_CLASSES = 1
NUM_BBOX_COORDS = 4

def build_face_detector_model():
    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base_model = MobileNetV2(
        include_top=False,
        input_tensor=input_tensor,
        weights='imagenet'
    )
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    

    classification_output = Dense(NUM_CLASSES, activation='sigmoid', name='classification_output')(x)
    

    bbox_output = Dense(NUM_BBOX_COORDS, activation='sigmoid', name='bbox_output')(x)

    model = Model(inputs=input_tensor, outputs=[classification_output, bbox_output])
    return model

def compile_and_train_model(model, train_dataset, val_dataset, epochs=50):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'classification_output': 'binary_crossentropy',
            'bbox_output': 'mean_squared_error'
        },
        metrics={
            'classification_output': 'accuracy',
            'bbox_output': ['mae']
        }
    )

    checkpoint_path = 'models/face_detector_{epoch:02d}.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]
    )
    return history

if __name__ == '__main__':

    def load_image_tf(x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    def load_labels_tf(label_path):
        label_path_str = label_path.numpy().decode('utf-8')
        with open(label_path_str, 'r', encoding="utf-8") as f:
            label = json.load(f)
        return np.array([label['class']], dtype=np.float32), np.array(label['bbox'], dtype=np.float32)

    train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
    train_images = train_images.map(load_image_tf).map(lambda x: tf.image.resize(x, (IMG_WIDTH, IMG_HEIGHT))).map(lambda x: x / 255.0)
    train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels_tf, [x], [tf.float32, tf.float32]))

    train_labels = train_labels.map(lambda classification_label, bbox_label: (tf.reshape(classification_label, [NUM_CLASSES]), tf.reshape(bbox_label, [NUM_BBOX_COORDS])))


    test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
    test_images = test_images.map(load_image_tf).map(lambda x: tf.image.resize(x, (IMG_WIDTH, IMG_HEIGHT))).map(lambda x: x / 255.0)
    test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels_tf, [x], [tf.float32, tf.float32]))

    test_labels = test_labels.map(lambda classification_label, bbox_label: (tf.reshape(classification_label, [NUM_CLASSES]), tf.reshape(bbox_label, [NUM_BBOX_COORDS])))

    val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
    val_images = val_images.map(load_image_tf).map(lambda x: tf.image.resize(x, (IMG_WIDTH, IMG_HEIGHT))).map(lambda x: x / 255.0)
    val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels_tf, [x], [tf.float32, tf.float32]))

    val_labels = val_labels.map(lambda classification_label, bbox_label: (tf.reshape(classification_label, [NUM_CLASSES]), tf.reshape(bbox_label, [NUM_BBOX_COORDS])))


    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    train_dataset = train_dataset.shuffle(5000).batch(8).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.zip((val_images, val_labels))
    val_dataset = val_dataset.shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)

    os.makedirs('models', exist_ok=True)


    model = build_face_detector_model()
    model.summary()
    
    print("\nIniciando o treinamento do modelo...")
    history = compile_and_train_model(model, train_dataset, val_dataset)
    print("Treinamento do modelo concluído!")
    
  
    model.save('models/final_face_detector_model.h5')
    print("Modelo final salvo em 'models/final_face_detector_model.h5'")