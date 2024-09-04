# import os
# os.environ["LOKY_MAX_CPU_COUNT"] = "6"

import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from pathlib import Path


images_path = Path('images')
json_file_path = 'road_types_json.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

images = []
labels = []

for item_id, item in data.items():
    img_path = images_path / item['filename']
    if img_path.exists():  
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        # preprocessing based on VGG input 
        # 1- normalization
        # 2- image size 
        # 3- RGB
        image = preprocess_input(image)  
        
        images.append(image)
        labels.append(item['regions'][0]['region_attributes']['type'])  

images = np.array(images)
# transform categorical labels to one-hot encoded
labels_df = pd.get_dummies(labels)  
# rows nb labels columns nb classes
labels = labels_df.values
num_classes = labels.shape[1]  


# class weights bcz my data is imbalanced
original_labels = [item['regions'][0]['region_attributes']['type'] for item_id, item in data.items() if (images_path / item['filename']).exists()]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(original_labels), y=original_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)  


y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


X_train_reshaped = X_train.reshape((X_train.shape[0], -1))
smote = SMOTE()
# one hot encoded again 
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train.argmax(axis=1))

X_train_resampled = X_train_resampled.reshape((-1, 224, 224, 3))
y_train_resampled = pd.get_dummies(y_train_resampled).values

# Data Augmentation
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.4, 1.2],  
    zoom_range=[0.7, 1.0]  
)
val_datagen = ImageDataGenerator()

# augment train only
train_generator = train_datagen.flow(X_train_resampled, y_train_resampled, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# removed top layer 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Add these
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  
x = Dense(256, activation='relu')(x)  
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  
predictions = Dense(num_classes, activation='softmax')(x)  


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# when val loss stops descreasing reduce lr
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
# same not decreasing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[reduce_lr, early_stopping])

for layer in base_model.layers[-8:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[reduce_lr, early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

model.save('road_type_classifier2.keras')
class_indices = {i: label for i, label in enumerate(labels_df.columns)}
with open('class_indices2.json', 'w') as f:
    json.dump(class_indices, f)
######## 97