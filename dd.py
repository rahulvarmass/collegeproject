import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score

# Function to define YOLOv7 model architecture
def yolov7(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Backbone
    x = Conv2D(64, 3, strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Head
    x = Conv2D(1024, 3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(30, 1, strides=(1, 1), padding='same')(x)
    predictions = Reshape((30,))(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model

# Function to parse YOLO format labels
def parse_yolo_label(label_path):
    # Implement parsing logic here
    pass

# Define paths to training, validation, and evaluation data
# Define directory path for both validation and evaluation datasets
val_eval_dir = "C:\\Users\\nirma\\Downloads\\collegeproject-master\\collegeproject-master"

# Define subdirectories for validation and evaluation datasets
val_img_dir = os.path.join(val_eval_dir, "val_images")
val_label_dir = os.path.join(val_eval_dir, "val_labels")
eval_img_dir = os.path.join(val_eval_dir, "eval_images")
eval_label_dir = os.path.join(val_eval_dir, "eval_labels")


# Define hyperparameters
BATCH_SIZE = 32
IMAGE_SIZE = (416, 416)
NUM_CLASSES = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Custom data generator
class CustomDataGenerator(tf.keras.utils.Sequence):
    def init(self, image_dir, label_dir, batch_size, image_size=(416, 416), num_classes=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.image_paths = sorted(os.listdir(image_dir))
        self.label_paths = sorted(os.listdir(label_dir))
        self.num_samples = len(self.image_paths)

    def len(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def getitem(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_label_paths = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for image_path, label_path in zip(batch_image_paths, batch_label_paths):
            image = cv2.imread(os.path.join(self.image_dir, image_path))
            image = cv2.resize(image, self.image_size)
            batch_images.append(image)
            label = parse_yolo_label(os.path.join(self.label_dir, label_path))
            batch_labels.append(label)
        return np.array(batch_images), np.array(batch_labels)

# Create custom data generators for training, validation, and evaluation
train_generator = CustomDataGenerator(train_img_dir, train_label_dir, BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES)
val_generator = CustomDataGenerator(val_img_dir, val_label_dir, BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES)
eval_generator = CustomDataGenerator(eval_img_dir, eval_label_dir, BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES)

# vic, [2/13/2024 10:12 PM]
# Create and compile YOLOv7 model
model = yolov7(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

# Train the model
model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=val_generator)

# Save the trained model
model.save("nasal_fracture_detector.h5")

# Evaluate the model
evaluation_images, ground_truth_labels = eval_generator[0]
predicted_labels = model.predict(evaluation_images)
predicted_labels = np.argmax(predicted_labels, axis=1)  # Assuming one class

# Compute accuracy
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)