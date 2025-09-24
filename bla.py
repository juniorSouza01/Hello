import os
import shutil
import random

BASE_DIR = "/home/gelado/Desktop/Hello/data"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")


DEST_DIRS = {
    "train": {"images": os.path.join(BASE_DIR, "train/images"),
              "labels": os.path.join(BASE_DIR, "train/labels")},
    "test": {"images": os.path.join(BASE_DIR, "test/images"),
             "labels": os.path.join(BASE_DIR, "test/labels")},
    "val": {"images": os.path.join(BASE_DIR, "val/images"),
            "labels": os.path.join(BASE_DIR, "val/labels")},
}


for split in DEST_DIRS.values():
    os.makedirs(split["images"], exist_ok=True)
    os.makedirs(split["labels"], exist_ok=True)


all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
random.shuffle(all_images) 


n_total = len(all_images)
n_train = int(0.7 * n_total)
n_test = int(0.15 * n_total)
n_val = n_total - n_train - n_test  

train_files = all_images[:n_train]
test_files = all_images[n_train:n_train + n_test]
val_files = all_images[n_train + n_test:]

splits = {"train": train_files, "test": test_files, "val": val_files}


for split_name, files in splits.items():
    for img_file in files:
        json_file = img_file.replace(".jpg", ".json")

        src_img = os.path.join(IMAGES_DIR, img_file)
        src_json = os.path.join(LABELS_DIR, json_file)

        dst_img = os.path.join(DEST_DIRS[split_name]["images"], img_file)
        dst_json = os.path.join(DEST_DIRS[split_name]["labels"], json_file)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        if os.path.exists(src_json):
            shutil.copy(src_json, dst_json)

print(f"✅ Divisão concluída! Treino: {len(train_files)}, Teste: {len(test_files)}, Validação: {len(val_files)}")
