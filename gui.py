import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model("model/cnn_model.h5")

# Define image dimensions
img_width, img_height = 100, 100

# Load class names from the dataset
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    "dataset",  
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())

# Function to classify an image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Load and preprocess the image
            image = Image.open(file_path)
            image = image.resize((img_width, img_height))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # Predict the class
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction)

            # Display the result
            result_label.config(text=f"Predicted Class: {predicted_class_name}\nConfidence: {confidence:.2f}")

            # Display the uploaded image
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img
        except Exception as e:
            result_label.config(text="Error processing image. Image cannot be recognized")

# Create the GUI
root = tk.Tk()
root.title("Group 2 - Fruit Recognition")

# Set window size
root.geometry("500x650")

# Top Frame (for Logo + Team Members)
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

# Group Logo on Left
try:
    logo = Image.open("group_logo.png")  # Replace with actual logo file
    logo = logo.resize((120, 120))
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(top_frame, image=logo)
    logo_label.grid(row=0, column=0, padx=10)
except:
    logo_label = tk.Label(top_frame, text="[Logo Missing]", font=("Arial", 12, "bold"))
    logo_label.grid(row=0, column=0, padx=10)

# Team Members on Right
team_label = tk.Label(top_frame, text="Team Members:\nMohamad Syakir\nMuhammad Harith\nMuhammad Aqmar\nNur Atiqah", 
                      font=("Arial", 12), justify="left")
team_label.grid(row=0, column=1, sticky="w")

# Add Project Title
title_label = tk.Label(root, text="Group 2 - Fruit Recognition", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Upload Button
upload_button = tk.Button(root, text="Upload Image", font=("Arial", 12), command=classify_image)
upload_button.pack(pady=20)

# Image Display Area
image_label = tk.Label(root)
image_label.pack(pady=10)

# Prediction Result Label
result_label = tk.Label(root, text="Predicted Class: ", font=("Arial", 14))
result_label.pack(pady=20)

# Run the GUI
root.mainloop()
