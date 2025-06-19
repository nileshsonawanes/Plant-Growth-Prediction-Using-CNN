import os
import numpy as np
import pickle
import cv2
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from utils.disease import disease_dic
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import load_model
import gdown

# Define the model path
model_path = "keras_model.h5"

# If the model is not present locally, download it
if not os.path.exists(model_path):
    print("Downloading keras_model.h5 from Google Drive...")
    file_id = "1dKJBZUvGBxm3aPNoJ82XsjKiBkzgszmC"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model after downloading
from keras.models import load_model
model = load_model(model_path)


# Paths for model and label binarizer
model_path = 'keras_model.h5'
label_binarizer_path = 'label_transform.pkl'

# Load model and label binarizer if exist
if os.path.exists(model_path) and os.path.exists(label_binarizer_path):
    model = load_model(model_path)
    label_binarizer = pickle.load(open(label_binarizer_path, 'rb'))
else:
    model = None
    label_binarizer = None

def prepare_image(image_path, target_size=(256, 256)):
    # Load image, convert to RGB, resize and preprocess
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(image_path):
    global model, label_binarizer
    if model is None or label_binarizer is None:
        raise ValueError("Model or label binarizer not loaded. Train the model first.")
    # Prepare image
    image = prepare_image(image_path)
    # Predict probabilities
    preds = model.predict(image)[0]
    # Get index of highest probability
    i = np.argmax(preds)
    # Get label
    label = label_binarizer.classes_[i]
    # Get disease info from disease_dic
    disease_info = disease_dic.get(label, "No information available for this disease.")
    return label, disease_info

def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(dataset_path='Dataset/PlantVillage', image_size=(256, 256), batch_size=32, epochs=10):
    global model, label_binarizer

    # Declare globals at the start of the function to avoid syntax error
    global model
    global label_binarizer

    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping

    # Use ImageDataGenerator to load images in batches from directory
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,  # Increased rotation range for more augmentation
        width_shift_range=0.2,  # Increased shift range
        height_shift_range=0.2,
        shear_range=0.3,  # Increased shear range
        zoom_range=0.3,  # Increased zoom range
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip augmentation
        fill_mode="nearest",
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    n_classes = len(train_generator.class_indices)
    print(f"[INFO] Number of classes: {n_classes}")

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(list(train_generator.class_indices.keys()))
    pickle.dump(label_binarizer, open(label_binarizer_path, 'wb'))

    inputShape = (image_size[0], image_size[1], 3)
    chanDim = -1

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    # Compile model with Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Callbacks for learning rate reduction and early stopping
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    print("[INFO] Training network...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[lr_reduction, early_stopping]
    )

    print("[INFO] Saving model...")
    model.save('keras_model.h5')

    # Save label binarizer for prediction use
    pickle.dump(label_binarizer, open(label_binarizer_path, 'wb'))

    # Plot training history
    plot_training_history(history)

    # Predict on validation data for confusion matrix
    val_steps = validation_generator.samples // validation_generator.batch_size
    validation_generator.reset()
    preds = model.predict(validation_generator, steps=val_steps+1, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = validation_generator.classes

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=list(label_binarizer.classes_))

    return model, label_binarizer
