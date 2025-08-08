import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.02, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    "train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test = test_datagen.flow_from_directory(
    "test",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# 2. Building the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 3. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
model.fit(train, validation_data=test, epochs=5)

# 5. Summary and Save
model.summary()
model.save("pneumonia_model.h5")

                  


