from fastai.vision.all import *
import os

# Set the path to your data
path = Path('../data')

# Check directories
print(f"Checking if path exists: {path.exists()}")
print(f"Contents of data directory: {os.listdir(path)}")

# Create a filtered list of directories we want to use
train_dirs = [d for d in ['fresh', 'stale'] if d in os.listdir(path)]
print(f"Using directories for training: {train_dirs}")

# Show sample contents
for dir_name in train_dirs:
    if os.path.isdir(path/dir_name):
        print(f"Contents of {dir_name} directory: {os.listdir(path/dir_name)[:5]}")  # Show first 5 items

# Create a new path for just our training data
train_path = path

# Get image files from only fresh and stale folders
files = []
for dir_name in train_dirs:
    files.extend(get_image_files(path/dir_name))
    
print(f"Number of image files found: {len(files)}")
if len(files) > 0:
    print(f"Sample file paths: {files[:3]}")  # Show first 3 files

# Only proceed if files were found
if len(files) > 0:
    # Verify images
    print("Verifying images...")
    failed = verify_images(files)
    if len(failed) > 0:
        print(f"Found {len(failed)} corrupted images. Removing...")
        failed.map(Path.unlink)
    else:
        print("No corrupted images found.")

    # Create DataLoaders - using specific path for each class
    print("Creating DataLoaders...")
    dls = ImageDataLoaders.from_folder(
        train_path,
        train='.',  # Look in current path only
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, mult=2, max_lighting=0.2, max_zoom=1.5),
        folders=train_dirs  # Only use these folders
    )
else:
    print("No image files found. Please check your data directory.")

# Check the dataset
print(f"Classes: {dls.vocab}")
print(f"Training images: {len(dls.train_ds)}")
print(f"Validation images: {len(dls.valid_ds)}")

# Create and train the model
print("Training model...")
learn = vision_learner(dls, resnet34, metrics=error_rate)  # Using resnet34 for better features
learn.class_weights = torch.tensor([0.8, 1.0])  # Adjust if needed based on class distribution
learn.fine_tune(4, freeze_epochs=1)

# Save the model
print("Saving model...")
import os
os.makedirs('../models', exist_ok=True)
learn.export('../models/fresh_stale_model.pkl')

print("Training complete!")