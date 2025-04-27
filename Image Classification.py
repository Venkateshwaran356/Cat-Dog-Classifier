


# Data Preprocessing
#---------------------------------------------------------------( 1 ) Resize the Images---------------------------------------------------------------------

from PIL import Image
import os

# Define the target size
target_size = (150, 150)

# Path to the folder containing images
dataset_folder = 'C:\Data Science and AI\Project\Image Classification (Cats vs Dogs)\Datasets'  # Replace with your folder path

# Loop through each class folder in the dataset
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    
    # Check if it is a folder (class)
    if os.path.isdir(class_path):
        # Loop through each image file in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image formats
                image_path = os.path.join(class_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Resize the image
                img_resized = img.resize(target_size)
                

                print(f"Resized image: {filename} in class: {class_folder}")

print("All images resized successfully!")


#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------( 2 ) Data Augmentation---------------------------------------------------------------------

from PIL import ImageEnhance
import random

# Loop through each class folder in the dataset
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    
    # Check if it is a folder (class)
    if os.path.isdir(class_path):
        # Loop through each image file in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image formats
                image_path = os.path.join(class_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Resize the image
                img_resized = img.resize(target_size)

                # 1. Flip the image horizontally
                img_flipped = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
                img_flipped.save(os.path.join(class_path, 'flipped_' + filename))
                
                # 2. Rotate the image by a random angle between -45 and 45 degrees
                rotation_angle = random.randint(-45, 45)
                img_rotated = img_resized.rotate(rotation_angle)
                img_rotated.save(os.path.join(class_path, 'rotated_' + str(rotation_angle) + '_' + filename))
                
                # 3. Random zoom (resize and then crop)
                zoom_factor = random.uniform(1.0, 1.5)  # Zoom factor between 1.0 and 1.5
                width, height = img_resized.size
                img_zoomed = img_resized.resize((int(width * zoom_factor), int(height * zoom_factor)))
                img_zoomed_cropped = img_zoomed.crop((0, 0, width, height))  # Crop to original size
                img_zoomed_cropped.save(os.path.join(class_path, 'zoomed_' + filename))
                
                # 4. Random width shift (horizontal shift)
                width_shift = random.randint(-20, 20)  # Horizontal shift by -20 to 20 pixels
                img_shifted = img_resized.transform(img_resized.size, Image.AFFINE, (1, 0, width_shift, 0, 1, 0))
                img_shifted.save(os.path.join(class_path, 'shifted_' + filename))
                
                # 5. Random height shift (vertical shift)
                height_shift = random.randint(-20, 20)  # Vertical shift by -20 to 20 pixels
                img_shifted_vertical = img_resized.transform(img_resized.size, Image.AFFINE, (1, 0, 0, 0, 1, height_shift))
                img_shifted_vertical.save(os.path.join(class_path, 'shifted_vertical_' + filename))
                
                # 6. Random shear (shear transformation)
                shear_factor = random.uniform(-0.5, 0.5)  # Shear factor between -0.5 and 0.5
                img_sheared = img_resized.transform(img_resized.size, Image.AFFINE, (1, shear_factor, 0, shear_factor, 1, 0))
                img_sheared.save(os.path.join(class_path, 'sheared_' + filename))
                
                # 7. Random brightness adjustment
                enhancer = ImageEnhance.Brightness(img_resized)
                brightness_factor = random.uniform(0.7, 1.3)  # Random brightness factor between 0.7 and 1.3
                img_brightness = enhancer.enhance(brightness_factor)
                img_brightness.save(os.path.join(class_path, 'brightened_' + str(brightness_factor) + '_' + filename))

print("Data augmentation completed for all images!")

#----------------------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------( 3 ) Normalizing image---------------------------------------------------------------------
import numpy as np

# Loop through each class folder in the dataset
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    
    # Check if it is a folder (class)
    if os.path.isdir(class_path):
        # Loop through each image file in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image formats
                image_path = os.path.join(class_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Resize the image
                img_resized = img.resize(target_size)

                # Convert image to numpy array (needed for normalization)
                img_array = np.array(img_resized)

                # Normalize the image (pixel values between 0 and 1)
                img_normalized = img_array / 255.0  # Dividing by 255 scales the values to [0, 1]

                # Convert back to image (if you need to save the normalized image)
                img_normalized = Image.fromarray((img_normalized * 255).astype(np.uint8))  # Convert back to uint8 for saving
                img_normalized.save(os.path.join(class_path, 'normalized_' + filename))

                print(f"Processed and normalized image: {filename} in class: {class_folder}")

print("Image normalization completed for all images!")


#----------------------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------( 4 ) Data Splitting---------------------------------------------------------------------
import shutil

# Folder paths for train and validation data (should already exist)
train_folder = 'train_data'  # Folder for training data
val_folder = 'val_data'  # Folder for validation data

# Split ratio (80% for training, 20% for validation)
split_ratio = 0.8

# Loop through each folder (class) in the dataset
for class_folder in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_folder)
    
    # Skip if it's not a folder (in case there are non-folder items)
    if not os.path.isdir(class_path):
        continue

    # Check if the class subfolders exist in the train and validation folders
    # If they don't exist, create them
    train_class_folder = os.path.join(train_folder, class_folder)
    val_class_folder = os.path.join(val_folder, class_folder)

    # Create class subfolders in train and validation folders (if not already there)
    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(val_class_folder, exist_ok=True)

    # List all images in the class folder (only .jpg or .png files)
    images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle the images for randomness
    random.shuffle(images)

    # Split the images (80% for training, 20% for validation)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Move images to the train and validation folders
    for img in train_images:
        shutil.move(os.path.join(class_path, img), os.path.join(train_class_folder, img))

    for img in val_images:
        shutil.move(os.path.join(class_path, img), os.path.join(val_class_folder, img))

    print(f"Class '{class_folder}' split into train and validation.")

print("Data splitting completed!")

#----------------------------------------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------( 5 ) Data Generator---------------------------------------------------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to training and validation directories
train_dir = 'C:/Data Science and AI/Project/Image Classification (Cats vs Dogs)/Datasets/train_data'  # Your training folder path
val_dir = 'C:/Data Science and AI/Project/Image Classification (Cats vs Dogs)/Datasets/val_data'      # Your validation folder path

# Create ImageDataGenerator for training
train_datagen = ImageDataGenerator(rescale=1./255)

# Create ImageDataGenerator for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create the train generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),   # Resize images to 150x150
    batch_size=32,            # Number of images per batch
    class_mode='binary'       # Use 'binary' for 2 classes, 'categorical' for multiple classes
)

# Create the validation generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------( 6 ) Model Building---------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a basic CNN model
model = Sequential()

# 1st Convolution + Pooling layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolution + Pooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Fully connected layer (Dense)
model.add(Dense(64, activation='relu'))

# Output layer (1 neuron for binary classification)
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' for binary

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Show the model summary
model.summary()

# Train the model using the training data generator
history = model.fit(
    train_generator,  # Training data
    steps_per_epoch=100,  # How many batches to process before each epoch (adjust based on your dataset)
    epochs=10,  # Number of epochs (iterations over the entire dataset)
    validation_data=val_generator,  # Validation data
    validation_steps=50  # How many validation batches to process per epoch
)

# Save the trained model (optional)
model.save('CAT VS DOG_cnn_model.h5')

print("Model training complete!")


#----------------------------------------------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------( 7 ) Model Evaluation---------------------------------------------------------------------

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator, steps=50)  # Adjust steps based on your validation set size

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

#----------------------------------------------------------------------------------------------------------------------------------------------------


