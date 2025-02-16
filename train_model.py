from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# Define image dimensions and number of classes
img_width, img_height = 100, 100
num_classes = 10  # Dataset
batch_size = 32
epochs = 15

# Path to the dataset
dataset_dir = "dataset"
model_save_path = "model/cnn_model.h5"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load and preprocess the validation dataset
validation_generator = validation_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build an improved CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),  # Replaces Flatten for better generalization
    Dropout(0.5),
    
    Dense(128, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# Compile the model with Adam optimizer and a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {test_accuracy * 100:.2f}%\n")

# Generate predictions
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Save confusion matrix to a file
np.savetxt(os.path.join(results_dir, "confusion_matrix.csv"), conf_matrix, delimiter=",", fmt="%d")
print(f"Confusion matrix saved to {os.path.join(results_dir, 'confusion_matrix.csv')}")

# Classification report
class_report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys())
print("\nClassification Report:")
print(class_report)

# Save classification report to a file
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(class_report)
print(f"Classification report saved to {os.path.join(results_dir, 'classification_report.txt')}")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(os.path.join(results_dir, "accuracy_loss_plot.png"))
plt.show()
print(f"Accuracy and loss plots saved to {os.path.join(results_dir, 'accuracy_loss_plot.png')}")

# Visualize misclassified images
misclassified_idx = np.where(y_pred_classes != y_true)[0]

plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx[:9]):  # Show first 9 misclassified images
    plt.subplot(3, 3, i + 1)
    img = validation_generator.filepaths[idx]
    plt.imshow(plt.imread(img))
    plt.title(f"True: {validation_generator.classes[idx]}\nPred: {y_pred_classes[idx]}")
    plt.axis('off')

plt.savefig(os.path.join(results_dir, "misclassified_images.png"))
plt.show()
print(f"Misclassified images saved to {os.path.join(results_dir, 'misclassified_images.png')}")