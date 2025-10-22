import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]

IMG_HEIGHT = params["img_height"]
IMG_WIDTH = params["img_width"]
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
NUM_CLASSES = params["num_classes"]

os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
df["label"] = df.apply(lambda row: 0 if row["healthy"] == 1 else 1, axis=1)
df = df[["image_id", "label"]]

def load_images(df, folder, size):
    images_flat, images_cnn, labels = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(folder, row["image_id"] + ".jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_flat.append(img_rgb.flatten())
        images_cnn.append(img_rgb)
        labels.append(row["label"])
    return np.array(images_flat), np.array(images_cnn, dtype=np.float32)/255.0, np.array(labels)

X_flat, X_cnn, y = load_images(df, os.path.join(RAW_DIR, "images"), (IMG_HEIGHT, IMG_WIDTH))

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=params["test_size"], random_state=params["random_state"]
)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y, test_size=params["test_size"], random_state=params["random_state"]
)

np.save(os.path.join(PROCESSED_DIR, "X_flat.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test_flat.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "X_cnn.npy"), X_train_cnn)
np.save(os.path.join(PROCESSED_DIR, "X_test_cnn.npy"), X_test_cnn)
np.save(os.path.join(PROCESSED_DIR, "y.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Data preprocessing complete. Saved to data/processed/")
