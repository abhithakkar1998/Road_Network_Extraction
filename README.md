# Road Network Extraction
This repository implements a road network extraction pipeline using deep learning techniques. 

The pipeline leverages:
- Python as the primary programming language.
- TensorFlow-Keras for building and training the deep learning model.
- OpenCV for image processing tasks.
- The core functionality utilizes a U-Net architecture for semantic segmentation of satellite images. This allows the model to identify and extract road pixels, effectively generating a road network representation from the satellite imagery.

Features:
- Semantic segmentation for road network extraction from satellite imagery.
- U-Net architecture for efficient pixel-wise classification.
- Implementation in Python with TensorFlow-Keras and OpenCV.
 
Dataset Link: https://drive.google.com/drive/folders/14zuHotv4NZaEayzfgS_dwPz39ktKAqZQ <br/>
(NOTE: This has been provided by Suhora Space Analytics for a Task-based Assessment. I take no claim on owning this dataset. Since the dataset link is public I am hereby sharing it. I'm not responsible if this link stops working.)

## IMPLEMENTATION & WORKING
1. **Setup Environment and Import Libraries**:
Ensure you have a Python environment set up with the necessary libraries installed. These libraries include TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, and scikit-learn.
Import the necessary libraries for handling data, images, model loading, training, evaluation, and plotting.

2. **Load Metadata**:
Read the metadata.csv file which contains paths to the satellite images and the information on which split (train, valid, test) each image belongs to.
Filter the metadata to get the train and validation set images.

3. **Load Class Dictionary**:
Read the class_dict.csv file to map pixel values to class names.

4. **Preprocess the Image**:
Define a function to read and preprocess the images. This includes resizing the image to the input size expected by the model and normalizing the pixel values.

5. **Binarize the Masks**:
Define a function to binarize the mask images at a threshold of 128. Greater than 128 should be set to 255 and less than 128 should be set to 0.

6. **Create Data Generators**:
Create custom data generators to load the images and masks in batches to avoid memory issues during training.

7. **Define the U-Net Model**:
Define the U-Net model architecture using Keras.

8. **Compile the Model**:
Compile the model with an appropriate optimizer (e.g., Adam), loss function (e.g., binary crossentropy), and metrics (e.g., accuracy).

9. **Train the Model**:
Train the model using the training and validation data generators.
Use callbacks such as ModelCheckpoint and EarlyStopping to save the best model and stop training when performance stops improving.

10. **Evaluate the Model**:
Evaluate the model on the validation set.
Calculate performance metrics such as precision, recall, and F1 score using scikit-learn.

11. **Generate Masks for Test Images**:
Load the test set images.
Use the trained model to generate masks for these images.
Resize the generated masks back to the original size of the images.

12. **Display an Image and its Generated Mask**:
Define a function to display a test image and its corresponding generated mask side by side.
Use Matplotlib to create a figure and plot the test image and mask in subplots for visual comparison.


## THEORETICAL BACKGROUND & APPROACH IDEA
Please check out "LINEAR FEATURE EXTRACTION IN SATELLITE IMAGERY.docx" file for more details.

## RESULTS
The model achieved a training accuracy of approx. 96%. Below is the graphs for accuracy and loss for both training and validation.
![train_val_accuracy](https://github.com/user-attachments/assets/f2e45639-5fb0-4c49-802e-351f4b0d23d0)
