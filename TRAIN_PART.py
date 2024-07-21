import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import cv2

#Creating a data generator for training and validation sets since we have low resource device
'''
Given the large dataset and limited system resources, we use a data generator to load and preprocess images and masks in batches rather than
loading the entire dataset into memory at once. This helps in managing memory usage efficiently.
'''
def data_generator(images, masks, batch_size):
    num_samples = len(images)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset + batch_size]
            batch_masks = masks[offset:offset + batch_size]
            yield batch_images, batch_masks


#defining the U-Net Model
def unet_model(input_size):
    inputs = Input(input_size)
    #Downsampling
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    #Upsampling
    up6 = concatenate([Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def evaluate_model(model, images, masks, batch_size):
    preds = model.predict(images, batch_size=batch_size)
    preds = (preds > 0.5).astype(np.uint8)

    # Flatten the arrays for metric calculations
    flat_preds = preds.flatten()
    flat_masks = masks.flatten()

    precision = precision_score(flat_masks, flat_preds)
    recall = recall_score(flat_masks, flat_preds)
    f1 = f1_score(flat_masks, flat_preds)

    print(f"Validation Precision: {precision}")
    print(f"Validation Recall: {recall}")
    print(f"Validation F1-Score: {f1}")

    # Plot Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(flat_masks, flat_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()

    # Save the plot to a file
    plt.savefig("val_precision_recall.png")


def main():
    inp = input("Do you want to start model training? (y/n)")
    if (inp == 'y' or inp == 'Y'):
        data_file = 'training_data_arrays.npz'
        data = np.load(data_file)
        train_images = data['train_images']
        train_masks = data['train_masks']
        val_images = data['val_images']
        val_masks = data['val_masks']
        print("Data Loaded")

        # Parameters
        batch_size = 8
        input_size = (256, 256, 3)

        # Create the model
        model = unet_model(input_size)

        # Create data generators
        train_gen = data_generator(train_images, train_masks, batch_size)
        val_gen = data_generator(val_images, val_masks, batch_size)
        print("Data Generators Initialized")

        # Calculate steps per epoch
        steps_per_epoch = len(train_images) // batch_size
        validation_steps = len(val_images) // batch_size

        # Ensure that the last incomplete batch is processed
        if len(train_images) % batch_size != 0:
            steps_per_epoch += 1
        if len(val_images) % batch_size != 0:
            validation_steps += 1

        # Define EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        model_checkpoint = ModelCheckpoint('best_unet_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        print("Callbacks Initialized")

        print("Start Training")
        # Train the model
        history = model.fit(train_gen, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=validation_steps, epochs=5, callbacks=[early_stopping, reduce_lr, model_checkpoint], verbose = 1)

        print("Training Done")

        # Evaluate the model on the validation set
        val_loss, val_accuracy = model.evaluate(val_gen, steps=validation_steps)

        # Print the evaluation results
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy}")

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("train_val_accuracy.png")

    else:
        print("Tata, Bye-Bye!")

if __name__ == "__main__":
    main()
