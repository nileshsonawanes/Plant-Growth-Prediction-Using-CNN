import math
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data Directories
train_dir = 'D:/A DATA/Project Module 2021-22/A PYTHON New/Plant Growth Prediction/CompleteUpdatedPlantGrowth/Dataset/train'
validation_dir = 'D:/A DATA/Project Module 2021-22/A PYTHON New/Plant Growth Prediction/CompleteUpdatedPlantGrowth/Dataset/validation'

# 2. Function to Create DataFrame
def create_dataframe(directory, class_name):
    filepaths = []
    labels = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(directory, file)
            filepaths.append(filepath)
            labels.append(class_name)  # Assign the class name
    return pd.DataFrame({'filename': filepaths, 'class': labels})

# 3. Create DataFrames
train_df = create_dataframe(train_dir, 'plant')  # Now both have the SAME class
validation_df = create_dataframe(validation_dir, 'plant')  # So we can combine them

# 4. Concatenate DataFrames (Combine train and validation data)
all_df = pd.concat([train_df, validation_df], ignore_index=True)

# 5. Print DataFrame Info for Debugging
print("Combined DataFrame:")
print(all_df.head())
print(all_df['class'].value_counts())

# 6. Data Augmentation (Only on training data)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   validation_split=0.2)  # Split into train/val

# 7. Data Generators using flow_from_dataframe (Specify subset for training/validation)
train_generator = train_datagen.flow_from_dataframe(
    all_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # <--- Changed back to 'categorical'
    subset='training')         # <--- Specify training subset

validation_generator = train_datagen.flow_from_dataframe(
    all_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # <--- Changed back to 'categorical'
    subset='validation')       # <--- Specify validation subset

# 8. Print Class Indices
print("\nTrain class indices:", train_generator.class_indices)
print("Validation class indices:", validation_generator.class_indices)

# 9. Model Definition (Categorical Classification - 1 output for each class)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # <-- Num classes and softmax
])

# 10. Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # <--- Changed back
              metrics=['accuracy'])

# 11. Calculate Steps per Epoch and Validation Steps
train_steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

# 12. Train the Model
epochs = 10
for epoch in range(epochs):

    print(f"Epoch {epoch+1}/{epochs}")

    # Reset the generators at the beginning of each epoch
    train_generator.reset()
    validation_generator.reset()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=1,  # Train only for one epoch at a time
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

# 13. Save the Model
model.save('keras_model.h5')