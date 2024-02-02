
!pip install np_utils
!pip install keras
!pip install opendatasets --upgrade --quiet

# IMport Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

import os


import PIL
import tensorflow as tf
import matplotlib.image as image
import seaborn as sns
import pickle

from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
import random
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import os
import opendatasets as od
dataset_url = 'https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset/data'
od.download(dataset_url)

dir_path = '../content/license-plate-digits-classification-dataset/CNN letter Dataset'
classes = sorted(os.listdir(dir_path))
NUM_CLASSES = len(classes)
print(classes)
print('Number of classes (letters and digits): ', NUM_CLASSES)

# Specify the number of images to display from each class
num_images_per_class = 5

# Plot images from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot images from the class
    plt.figure(figsize=(15, 3))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Convert BGR to RGB (matplotlib uses RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot the image
        plt.subplot(1, num_images_per_class, i + 1)
        plt.imshow(img)
        plt.title(f'Class: {class_name}')
        plt.axis('off')

    plt.show()


# Plot rescaled images from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot rescaled images from the class
    plt.figure(figsize=(15, 3))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Rescale the image
        scaling_matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 0, 0.5)
        scaled_image = cv2.warpAffine(img, scaling_matrix, (img.shape[1], img.shape[0]))

        # Convert BGR to RGB (matplotlib uses RGB)
        scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

        # Plot the rescaled image
        plt.subplot(1, num_images_per_class, i + 1)
        plt.imshow(scaled_image)
        plt.title(f'Class: {class_name}')
        plt.axis('off')

    plt.show()

# Specify the number of images to resize and display from each class
num_images_per_class = 5

# Plot resized images from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot resized images from the class
    plt.figure(figsize=(15, 3))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Resize the image with linear interpolation
        resized_image = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB (matplotlib uses RGB)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Plot the resized image
        plt.subplot(1, num_images_per_class, i + 1)
        plt.imshow(resized_image)
        plt.title(f'Class: {class_name}')
        plt.axis('off')

    plt.show()

# Plot normalized images from each class

from torchvision import transforms
# Specify the number of images to normalize and display from each class
num_images_per_class = 5

# Define the transformation for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Plot normalized images from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot normalized images from the class
    plt.figure(figsize=(15, 3))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Apply the normalization transformation
        normalized_image = transform(img)

        # Convert tensor to NumPy array and transpose dimensions
        normalized_image = normalized_image.numpy().transpose(1, 2, 0)

        # Plot the normalized image
        plt.subplot(1, num_images_per_class, i + 1)
        plt.imshow(normalized_image)
        plt.title(f'Class: {class_name}')
        plt.axis('off')

    plt.show()

# Specify the number of images to process and display from each class
num_images_per_class = 5

# Plot contrast enhancement for images from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot contrast enhancement for images from the class
    plt.figure(figsize=(15, 6))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Plot the original and equalized images side by side
        plt.subplot(2, num_images_per_class, i + 1)
        plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_images_per_class, num_images_per_class + i + 1)
        plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB))
        plt.title('Equalized')
        plt.axis('off')

    plt.suptitle(f'Contrast Enhancement - Class: {class_name}', fontsize=16)
    plt.show()

num_images_per_class = 5

# Plot global threshold for each image from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot global threshold for each image from the class
    plt.figure(figsize=(15, 6))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply global threshold
        ret, global_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        # Plot the original and thresholded images side by side
        plt.subplot(2, num_images_per_class, i + 1)
        plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_images_per_class, num_images_per_class + i + 1)
        plt.imshow(global_thresh, cmap='gray')
        plt.title('Global Threshold')
        plt.axis('off')

    plt.suptitle(f'Global Threshold - Class: {class_name}', fontsize=16)
    plt.show()

# Specify the number of images to process and display from each class
num_images_per_class = 5

# Plot adaptive threshold for each image from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot adaptive threshold for each image from the class
    plt.figure(figsize=(15, 6))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding using mean method
        adaptive_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Plot the original and thresholded images side by side
        plt.subplot(2, num_images_per_class, i + 1)
        plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_images_per_class, num_images_per_class + i + 1)
        plt.imshow(adaptive_image, cmap='gray')
        plt.title('Adaptive Threshold (Mean)')
        plt.axis('off')

    plt.suptitle(f'Adaptive Threshold (Mean) - Class: {class_name}', fontsize=16)
    plt.show()

# Specify the number of images to process and display from each class
num_images_per_class = 5

# Define the lower and upper thresholds for Canny edge detection
th_upper = 180
th_lower = 80

# Plot Canny edge detection for each image from each class
for class_name in classes:
    # Get the path to the class folder
    class_path = os.path.join(dir_path, class_name)

    # Get the list of image filenames in the class folder
    image_files = os.listdir(class_path)[:num_images_per_class]

    # Plot Canny edge detection for each image from the class
    plt.figure(figsize=(15, 6))
    for i, image_file in enumerate(image_files):
        # Read the image
        img = cv2.imread(os.path.join(class_path, image_file))

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        canny_image = cv2.Canny(gray_image, th_lower, th_upper)

        # Plot the original and Canny edge detected images side by side
        plt.subplot(2, num_images_per_class, i + 1)
        plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_images_per_class, num_images_per_class + i + 1)
        plt.imshow(canny_image, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.axis('off')

    plt.suptitle(f'Canny Edge Detection - Class: {class_name}', fontsize=16)
    plt.show()

#Model Bindings

digits_counter = {}
NUM_IMAGES = 0

for digit in digits:
    path = os.path.join(dir_path, digit)
    digits_counter[digit] = len(os.listdir(path))
    NUM_IMAGES += len(os.listdir(path))

print(digits_counter)
print('Number of all images: ', NUM_IMAGES)

rows, columns = 7, 5

k = 0
fig, axes = plt.subplots(rows, columns, figsize=(30, 30))
for row in range(rows):
    for column in range(columns):
        rand_num = np.random.randint(0, digits_counter[digits[k]])
        class_path = dir_path + '/' + str(digits[k])
        image_path = class_path + '/' + str(os.listdir(class_path)[rand_num])
        ax = axes[row, column]
        ax.set_title(digits[k], loc='center', fontsize=16)
        ax.imshow(imread(image_path), cmap='gray')
        k += 1
plt.show()

#Image Augmented

data = []
labels = []
MAX_NUM = None   # maximum number of digits images per class
IMG_WIDTH, IMG_HEIGHT = 32, 40

# images of digits '6' in folder with '2'
incorrect_img = [
    'aug20121_0.jpg',
    'aug20122_1.jpg',
    'aug20123_2.jpg',
    'aug20124_3.jpg',
    'aug20125_4.jpg',
    'aug20126_5.jpg',
    'aug20127_6.jpg',
    'aug20128_7.jpg',
    'aug20129_8.jpg',
    'aug20130_9.jpg'
]

for digit in digits:
    path = os.path.join(dir_path, digit)
    label = digits.index(digit)
    for img in os.listdir(path):
        if img in incorrect_img:
            continue
        img_path = os.path.join(path, img)
        img_array = cv.imread(img_path)
        resized = cv.resize(img_array, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
        data.append(gray)
        labels.append(label)
        if MAX_NUM is not None:
            if labels.count(label) == MAX_NUM:
                break

data = np.array(data, dtype='float32')
labels = np.array(labels, dtype='int8')

print(data.shape)

data = data / 255.0
data = data.reshape(*data.shape, 1)
labels = to_categorical(labels, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=.3)

print("Training dataset shape: ", X_train.shape, y_train.shape)
print("Validation dataset shape: ", X_val.shape, y_val.shape)
print("Testing dataset shape: ", X_test.shape, y_test.shape)

model = tf.keras.Sequential([
    Flatten(input_shape=(40, 32, 1)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(35,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=15, batch_size=256,
                    validation_data=(X_val, y_val))

hist=history.history
plt.plot(hist["accuracy"],color="b",label="train_accuracy")
plt.plot(hist["val_accuracy"],color="g",label="val_accuracy")
plt.legend(loc="lower right")
plt.show()

model.evaluate(X_val,y_val)
model.evaluate(X_test,y_test)

# Making predictions for the NN model
predictions_nn_prob = model.predict(X_test)
predictions_nn = np.argmax(predictions_nn_prob, axis=1)

# Plotting confusion matrix for NN model
conf_matrix_nn = confusion_matrix(np.argmax(y_test, axis=1), predictions_nn)
plt.figure(figsize=(14, 8))
sns.heatmap(conf_matrix_nn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Neural Network")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#Convolutional  Neural Network

input_layer=Input(shape=(40,32,1))
layer=Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="valid")(input_layer)
layer=MaxPool2D(pool_size=(2,2),strides=(1,1))(layer)
layer=BatchNormalization()(layer)

layer=Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="valid")(input_layer)
layer=MaxPool2D(pool_size=(2,2),strides=(1,1))(layer)
layer=BatchNormalization()(layer)

layer=Flatten()(layer)

layer=Dense(64,activation="relu")(layer)
layer=BatchNormalization()(layer)

layer=Dense(32,activation="relu")(layer)
output_layer=Dense(35,activation="softmax")(layer)
model2=Model(inputs=input_layer,outputs=output_layer)
model2.summary()

model2.compile(loss='categorical_crossentropy',
       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
       metrics=['accuracy'])

history2 = model2.fit(X_train, y_train,
                    epochs=15, batch_size=256,
                    validation_data=(X_val, y_val))

hist=history2.history
plt.plot(hist["accuracy"],color="b",label="train_accuracy")
plt.plot(hist["val_accuracy"],color="g",label="val_accuracy")
plt.legend(loc="lower right")
plt.show()

model2.evaluate(X_val,y_val)

model2.evaluate(X_test,y_test)

# Making predictions for the CNN model
predictions_cnn_prob = model2.predict(X_test)
predictions_cnn = np.argmax(predictions_cnn_prob, axis=1)

# Plotting confusion matrix for CNN model
conf_matrix_cnn = confusion_matrix(np.argmax(y_test, axis=1), predictions_cnn)
plt.figure(figsize=(14, 8))
sns.heatmap(conf_matrix_cnn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Convolutional Neural Network")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc

# Getting predicted probabilities for each class
y_pred_prob_cnn = model2.predict(X_test)

# Calculating fpr and tpr for each class
fpr_cnn = dict()
tpr_cnn = dict()
roc_auc_cnn = dict()

for i in range(NUM_CLASSES):
    fpr_cnn[i], tpr_cnn[i], _ = roc_curve(y_test[:, i], y_pred_prob_cnn[:, i])
    roc_auc_cnn[i] = auc(fpr_cnn[i], tpr_cnn[i])

# Micro-average ROC curve and ROC area
fpr_cnn_micro, tpr_cnn_micro, _ = roc_curve(y_test.ravel(), y_pred_prob_cnn.ravel())
roc_auc_cnn_micro = auc(fpr_cnn_micro, tpr_cnn_micro)

# Plotting ROC curve
plt.figure(figsize=(8, 6))

plt.plot(fpr_cnn_micro, tpr_cnn_micro, color='orange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc_cnn_micro))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - CNN')
plt.legend(loc="lower right")
plt.show()


#Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

# Reshaping data for DecisionTreeClassifier
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Create Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train_flattened, np.argmax(y_train, axis=1))

# Make predictions on the test dataset
predictions_dt_test = dt_classifier.predict(X_test_flattened)

# Evaluate the classifier
accuracy_train = dt_classifier.score(X_train_flattened, np.argmax(y_train, axis=1))
accuracy_test = dt_classifier.score(X_test_flattened, np.argmax(y_test, axis=1))

precision_dt_test = precision_score(np.argmax(y_test, axis=1), predictions_dt_test, average='weighted')
recall_dt_test = recall_score(np.argmax(y_test, axis=1), predictions_dt_test, average='weighted')
f1_dt_test = f1_score(np.argmax(y_test, axis=1), predictions_dt_test, average='weighted')

print("Decision Tree Classifier - Train Accuracy:", accuracy_train)
print("Decision Tree Classifier - Test Accuracy:", accuracy_test)
print("Decision Tree Classifier - Test Precision:", precision_dt_test)
print("Decision Tree Classifier - Test Recall:", recall_dt_test)
print("Decision Tree Classifier - Test F1-score:", f1_dt_test)

#Support Vector Mechine

from sklearn.metrics import precision_score, recall_score, f1_score

# Make predictions on the test dataset
predictions_svm = svm_classifier.predict(X_test_flattened)

# Calculate accuracy
accuracy_test_svm = svm_classifier.score(X_test_flattened, np.argmax(y_test, axis=1))

# Calculate precision, recall, and F1-score
precision_svm = precision_score(np.argmax(y_test, axis=1), predictions_svm, average='weighted')
recall_svm = recall_score(np.argmax(y_test, axis=1), predictions_svm, average='weighted')
f1_svm = f1_score(np.argmax(y_test, axis=1), predictions_svm, average='weighted')

# Print the results
print("SVM Classifier - Test Accuracy:", accuracy_test_svm)
print("SVM Classifier - Precision:", precision_svm)
print("SVM Classifier - Recall:", recall_svm)
print("SVM Classifier - F1-score:", f1_svm)


#Predict

# Choose the model (replace 'model' with your chosen model)
chosen_model = model  # Example: model, model2, dt_classifier, svm_classifier

# Make predictions on the test dataset
if chosen_model in [model, model2]:
    predictions = chosen_model.predict(X_test)
else:
    # For Decision Tree Classifier and SVM, reshape the input data
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)
    predictions = chosen_model.predict(X_test_flattened)

# Choose specific images from the test dataset to display (e.g., first 10 images)
num_images_to_display = 5
images_to_display = X_test[:num_images_to_display]
true_labels = np.argmax(y_test[:num_images_to_display], axis=1)
if chosen_model in [model, model2]:
    predicted_labels = np.argmax(predictions[:num_images_to_display], axis=1)
else:
    predicted_labels = predictions[:num_images_to_display]

# Display the chosen images along with their true labels and predicted labels
plt.figure(figsize=(5, 3*num_images_to_display))
for i in range(num_images_to_display):
    plt.subplot(num_images_to_display, 1, i + 1)
    plt.imshow(images_to_display[i].reshape(40, 32), cmap='gray')
    plt.title(f'True: {true_labels[i]}, Predicted: {predicted_labels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
