import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_DIR = "data/raw"
AUGMENTED_DIR = "data/augmented"
IMG_HEIGHT, IMG_WIDTH = 128, 128

os.makedirs(AUGMENTED_DIR, exist_ok=True)
os.makedirs(os.path.join(AUGMENTED_DIR, "images"), exist_ok=True)

df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
df["label"] = df.apply(lambda row: 0 if row["healthy"] == 1 else 1, axis=1)
df = df[["image_id", "label"]]

aug_gen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

img_folder = os.path.join(RAW_DIR, "images")
save_folder = os.path.join(AUGMENTED_DIR, "images")

aug_images, aug_labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(img_folder, row["image_id"] + ".jpg")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img.reshape((1,) + img.shape)
    
    count = 0
    for batch in aug_gen.flow(img, batch_size=1, save_to_dir=save_folder,
                              save_prefix=row["image_id"], save_format='jpg'):
        aug_images.append(batch[0].astype(np.float32)/255.0)
        aug_labels.append(row["label"])
        count += 1
        if count >= 2: 
            break

aug_images = np.array(aug_images)
aug_labels = np.array(aug_labels)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    aug_images, aug_labels, test_size=0.2, random_state=42
)

X_train_flat = np.array([img.flatten() for img in X_train_cnn])
X_test_flat = np.array([img.flatten() for img in X_test_cnn])

np.save(os.path.join(AUGMENTED_DIR, "X_flat.npy"), X_train_flat)
np.save(os.path.join(AUGMENTED_DIR, "y.npy"), y_train_cnn)
np.save(os.path.join(AUGMENTED_DIR, "X_test_flat.npy"), X_test_flat)
np.save(os.path.join(AUGMENTED_DIR, "y_test.npy"), y_test_cnn)
np.save(os.path.join(AUGMENTED_DIR, "X_cnn.npy"), X_train_cnn)
np.save(os.path.join(AUGMENTED_DIR, "y_cnn.npy"), y_train_cnn)
np.save(os.path.join(AUGMENTED_DIR, "X_test_cnn.npy"), X_test_cnn)
np.save(os.path.join(AUGMENTED_DIR, "y_test_cnn.npy"), y_test_cnn)

print(f"Augmentation complete. Saved {len(aug_images)} images and train/test splits.")
