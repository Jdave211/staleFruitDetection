from fastai.vision.all import *
import matplotlib.pyplot as plt
import os
import torch

# Load the model
learn = load_learner('fresh_stale_model.pkl')

# Function to predict and display an image with activation map
def predict_with_activation(img_path):
    # Get filename
    filename = os.path.basename(img_path)
    
    # Load the image
    img = PILImage.create(img_path)
    
    # Make prediction
    pred_class, pred_idx, probs = learn.predict(img)
    confidence = float(probs[pred_idx])
    
    # Create interpretation
    interp = ClassificationInterpretation.from_learner(learn)
    
    # Get the most important features using feature importance
    # We'll display the original image alongside the prediction
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display original image
    ax.imshow(img)
    ax.set_title(f"Prediction: {pred_class} ({confidence:.2%})")
    ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"File: {filename}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return {
        'filename': filename,
        'prediction': pred_class,
        'probability': confidence
    }

# Get all images from the test directory
def get_test_images(base_path='../data/test'):
    if not os.path.exists(base_path):
        print(f"Warning: Test directory {base_path} does not exist!")
        return []
        
    # Get all image files in the test directory
    test_images = [f"{base_path}/{f}" for f in os.listdir(base_path) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    return test_images

# Get test images
test_images = get_test_images()
print(f"Found {len(test_images)} test images...")

# Process images
results = []
for img_path in test_images:
    print(f"Processing: {img_path}")
    result = predict_with_activation(img_path)
    results.append(result)
    print(f"Prediction: {result['prediction']} (Confidence: {result['probability']:.2%})")
    print("---")

# Summarize results
fresh_count = sum(1 for r in results if r['prediction'] == 'fresh')
stale_count = sum(1 for r in results if r['prediction'] == 'stale')
print(f"Summary: {fresh_count} images classified as fresh, {stale_count} images classified as stale")