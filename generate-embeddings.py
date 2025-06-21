# generate_embeddings.py
import os
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

def generate_embeddings():
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    approved_dir = "APPROVED-SIGNAGE"  # Put your approved images here
    approved_embeddings = []
    approved_image_names = []
    
    for fname in os.listdir(approved_dir):
        if fname.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
            print(f"Processing: {fname}")
            
            img_path = os.path.join(approved_dir, fname)
            img_pil = Image.open(img_path).convert('RGB')
            img_resized = img_pil.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            embedding = model.predict(img_array, verbose=0)
            approved_embeddings.append(embedding[0])
            approved_image_names.append(fname)
    
    # Save embeddings
    with open("approved_embeddings.pkl", "wb") as f:
        pickle.dump(np.array(approved_embeddings), f)
    with open("approved_names.pkl", "wb") as f:
        pickle.dump(approved_image_names, f)
    
    print(f"✅ Generated embeddings for {len(approved_embeddings)} images")

if __name__ == "__main__":
    generate_embeddings()
