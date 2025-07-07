import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder)[:200]:  # limit for faster testing
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (64, 64))
        img = img.flatten()
        images.append(img)
        labels.append(label)
    return images, labels

cat_images, cat_labels = load_images("data/cats", 0)
dog_images, dog_labels = load_images("data/dogs", 1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = SVC()
model.fit(X_train, y_train)

print("Model trained with accuracy:", model.score(X_test, y_test))
