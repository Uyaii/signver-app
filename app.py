import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import os
import pickle
from PIL import Image
import io
import requests
from urllib.parse import urlparse
import subprocess
import sys
import shutil

# Page config
st.set_page_config(
    page_title="SIGNVER - Signage Verification",
    page_icon="💎",
    layout="wide"
)

# Custom CSS to match your website theme
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #4CAF50;
    margin-bottom: 2rem;
}
.upload-section {
    background: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.result-approved {
    background: linear-gradient(135deg, #4CAF50, #81C784);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #f44336, #ef5350);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.system-error {
    background: linear-gradient(135deg, #ff9800, #ffb74d);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Configuration
SIMILARITY_THRESHOLD = 0.85
DISTANCE_THRESHOLD = 0.5
ADMIN_PASSWORD = "admin123"  # Change this to your secure password
APPROVED_SIGNAGE_DIR = "APPROVED-SIGNAGE"  # Directory for approved signage images

@st.cache_resource
def load_model():
    """Load MobileNetV2 model (cached)"""
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

@st.cache_data
def load_approved_embeddings():
    """Load pre-generated approved signage embeddings"""
    embeddings_path = "approved_embeddings.pkl"
    names_path = "approved_names.pkl"
    
    if os.path.exists(embeddings_path) and os.path.exists(names_path):
        with open(embeddings_path, 'rb') as f:
            approved_embeddings = pickle.load(f)
        with open(names_path, 'rb') as f:
            approved_image_names = pickle.load(f)
        return approved_embeddings, approved_image_names
    else:
        return None, None

def get_embedding(img_pil, model):
    """Convert PIL image to embedding"""
    img_resized = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)
    return embedding[0]

def load_image_from_url(url):
    """Load image from URL"""
    try:
        # Validate URL
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return None, "Invalid URL format"
        
        # Download image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return None, "URL does not point to an image"
        
        # Convert to PIL Image
        img_pil = Image.open(io.BytesIO(response.content)).convert('RGB')
        return img_pil, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out - URL took too long to respond"
    except requests.exceptions.RequestException as e:
        return None, f"Failed to download image: {str(e)}"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def save_images_to_approved_folder(new_images_data):
    """Save new images to the APPROVED-SIGNAGE folder"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(APPROVED_SIGNAGE_DIR, exist_ok=True)
        
        saved_count = 0
        for img_name, img_pil in new_images_data:
            # Generate unique filename if file already exists
            base_name, ext = os.path.splitext(img_name)
            counter = 1
            final_name = img_name
            
            while os.path.exists(os.path.join(APPROVED_SIGNAGE_DIR, final_name)):
                final_name = f"{base_name}_{counter}{ext}"
                counter += 1
            
            # Save image
            img_path = os.path.join(APPROVED_SIGNAGE_DIR, final_name)
            img_pil.save(img_path)
            saved_count += 1
        
        return True, saved_count, None
        
    except Exception as e:
        return False, 0, str(e)

def run_embedding_generation_script():
    """Run the external generate_embeddings.py script"""
    try:
        # Check if the script exists
        script_path = "generate-embeddings.py"  
        if not os.path.exists(script_path):
            return False, f"Embedding generation script not found at {script_path}"
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Script failed with error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Embedding generation timed out (5 minutes)"
    except Exception as e:
        return False, f"Error running embedding script: {str(e)}"

def add_new_approved_signages(new_images_data):
    """Add new images to approved folder and regenerate embeddings"""
    try:
        # Step 1: Save images to APPROVED-SIGNAGE folder
        success, saved_count, error = save_images_to_approved_folder(new_images_data)
        if not success:
            return False, f"Failed to save images: {error}"
        
        # Step 2: Run the external embedding generation script
        success, output = run_embedding_generation_script()
        if not success:
            return False, f"Failed to generate embeddings: {output}"
        
        return True, f"Successfully added {saved_count} images and regenerated embeddings"
        
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def admin_section():
    """Admin section for adding new approved signages"""
    st.markdown("---")
    st.subheader("🔐 Admin Section")
    
    # Check if admin is authenticated
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        # Password input
        st.write("Enter admin password to add new approved signages:")
        password = st.text_input("Password:", type="password", key="admin_password")
        
        if st.button("Authenticate", key="auth_button"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("✅ Admin authenticated successfully!")
                st.rerun()
            else:
                st.error("❌ Incorrect password")
    
    else:
        # Admin is authenticated - show admin panel
        st.success("🔓 Admin authenticated")
        
        if st.button("🚪 Logout", key="logout_button"):
            st.session_state.admin_authenticated = False
            st.rerun()
        
        st.markdown("### 📁 Add New Approved Signages")
        
        # File uploader for multiple images
        new_images = st.file_uploader(
            "Upload new approved signage images:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Select multiple images to add to the approved database"
        )
        
        if new_images:
            st.write(f"📊 Selected {len(new_images)} images:")
            
            # Preview images
            cols = st.columns(min(len(new_images), 4))
            for i, uploaded_file in enumerate(new_images):
                with cols[i % 4]:
                    img_pil = Image.open(uploaded_file).convert('RGB')
                    st.image(img_pil, caption=uploaded_file.name, use_column_width=True)
            
            # Add to database button
            if st.button("🔄 Add to Database & Regenerate Embeddings", key="add_to_db"):
                with st.spinner("🔄 Saving images and regenerating embeddings..."):
                    
                    # Prepare new images data
                    new_images_data = []
                    for uploaded_file in new_images:
                        img_pil = Image.open(uploaded_file).convert('RGB')
                        new_images_data.append((uploaded_file.name, img_pil))
                    
                    # Add images and regenerate embeddings using external script
                    success, message = add_new_approved_signages(new_images_data)
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.balloons()
                        
                        # Clear cache to reload new embeddings
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        
        # Current database info
        st.markdown("### 📊 Current Database Status")
        approved_embeddings, approved_names = load_approved_embeddings()
        
        if approved_embeddings is not None:
            st.info(f"📈 Current database contains {len(approved_embeddings)} approved signages")
            
            with st.expander("📋 View all approved signages"):
                for i, name in enumerate(approved_names):
                    st.write(f"{i+1}. {name}")
        else:
            st.warning("⚠️ No approved signages in database")
        
        # Check approved folder status
        st.markdown("### 📁 Approved Signage Folder Status")
        if os.path.exists(APPROVED_SIGNAGE_DIR):
            folder_files = [f for f in os.listdir(APPROVED_SIGNAGE_DIR) 
                          if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]
            st.info(f"📂 {APPROVED_SIGNAGE_DIR} folder contains {len(folder_files)} image files")
            
            if st.button("🔄 Regenerate Embeddings from Folder", key="regen_from_folder"):
                with st.spinner("🔄 Regenerating embeddings from existing folder..."):
                    success, output = run_embedding_generation_script()
                    
                    if success:
                        st.success("✅ Embeddings regenerated successfully!")
                        st.success(f"Output: {output}")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ {output}")
        else:
            st.warning(f"⚠️ {APPROVED_SIGNAGE_DIR} folder does not exist")
            if st.button("📁 Create Approved Signage Folder", key="create_folder"):
                try:
                    os.makedirs(APPROVED_SIGNAGE_DIR, exist_ok=True)
                    st.success(f"✅ Created {APPROVED_SIGNAGE_DIR} folder")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create folder: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">💎 SIGNVER - SIGNAGE VERIFICATION</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 18px;">Upload a signage image to verify compliance with LASAA regulations</p>', unsafe_allow_html=True)
    
    # Load model and embeddings
    model = load_model()
    approved_embeddings, approved_image_names = load_approved_embeddings()
    
    # Check if embeddings are available (system check)
    if approved_embeddings is None:
        st.markdown("""
        <div class="system-error">
            <h2>⚠️ SYSTEM ERROR</h2>
            <p><strong>Approved signage database not found.</strong></p>
            <p>Please contact the system administrator.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("Developer Note: Make sure 'approved_embeddings.pkl' and 'approved_names.pkl' files are present in the app directory.")
        st.info("💡 Tip: Use the admin section below to add approved signages and generate embeddings.")
        
        # Still show admin section even if embeddings don't exist
        admin_section()
        return
    
    # Show system status
    st.success(f"✅ System Ready - {len(approved_embeddings)} approved signage references loaded")
    
    # Main upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Signage Image")
    
    # Image input options
    input_method = st.radio(
        "Choose how to provide your image:",
        ["📁 Upload from device", "🌐 Use image URL"],
        horizontal=True
    )
    
    img_pil = None
    image_source = ""
    
    if input_method == "📁 Upload from device":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image of the signage you want to verify for LASAA compliance"
        )
        
        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert('RGB')
            image_source = uploaded_file.name
    
    else:  # URL input
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/your-signage-image.jpg",
            help="Paste the direct link to your signage image"
        )
        
        if image_url:
            with st.spinner("🌐 Loading image from URL..."):
                img_pil, error = load_image_from_url(image_url)
                
                if error:
                    st.error(f"❌ {error}")
                    img_pil = None
                else:
                    image_source = image_url
    
    if img_pil is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Your Signage")
            st.image(img_pil, use_column_width=True)
            if input_method == "🌐 Use image URL":
                st.caption(f"Source: {image_source}")
            else:
                st.caption(f"File: {image_source}")
        
        # Process image
        with st.spinner("🔍 Analyzing signage for compliance..."):
            # Get embedding
            uploaded_embedding = get_embedding(img_pil, model).reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(uploaded_embedding, approved_embeddings)
            distances = cdist(uploaded_embedding, approved_embeddings, metric='euclidean')
            
            # Find best match
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[0][best_match_index]
            best_distance = distances[0][best_match_index]
            
            # Decision
            is_approved = best_similarity >= SIMILARITY_THRESHOLD
        
        with col2:
            st.subheader("📊 Verification Result")
            
            # Display result
            if is_approved:
                st.markdown(f"""
                <div class="result-approved">
                    <h2>✅ SIGNAGE APPROVED</h2>
                    <p><strong>LASAA Compliance:</strong> PASSED</p>
                    <p><strong>Confidence:</strong> {best_similarity:.1%}</p>
                    <p><strong>Status:</strong> Your signage meets regulatory requirements</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <h2>❌ SIGNAGE NOT APPROVED</h2>
                    <p><strong>LASAA Compliance:</strong> FAILED</p>
                    <p><strong>Confidence:</strong> {best_similarity:.1%}</p>
                    <p><strong>Status:</strong> Your signage does not meet regulatory requirements</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.subheader("📈 Analysis Details")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Similarity Score", f"{best_similarity:.4f}", f"Required: {SIMILARITY_THRESHOLD}")
                st.metric("Best Match", approved_image_names[best_match_index])
            
            with col_b:
                st.metric("Distance Score", f"{best_distance:.4f}")
                st.metric("Database Size", f"{len(approved_embeddings)} references")
        
        # Detailed analysis section
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("🔬 Advanced Analysis (Optional)"):
            # Similarity distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Similarity histogram
            ax1.hist(similarities[0], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(x=best_similarity, color='red', linestyle='--', linewidth=2, label=f'Your Result ({best_similarity:.4f})')
            ax1.axvline(x=SIMILARITY_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Approval Threshold ({SIMILARITY_THRESHOLD})')
            ax1.set_title('Similarity Distribution')
            ax1.set_xlabel('Cosine Similarity')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Distance histogram
            ax2.hist(distances[0], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
            ax2.axvline(x=best_distance, color='red', linestyle='--', linewidth=2, label=f'Your Result ({best_distance:.4f})')
            ax2.set_title('Distance Distribution')
            ax2.set_xlabel('Euclidean Distance')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Top 5 matches
            st.subheader("🏆 Top 5 Most Similar Approved Signs")
            top_5_indices = np.argsort(similarities[0])[-5:][::-1]
            
            for i, idx in enumerate(top_5_indices):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"**#{i+1}**")
                with col2:
                    st.write(approved_image_names[idx])
                with col3:
                    st.write(f"{similarities[0][idx]:.4f}")
    
    else:
        st.markdown("</div>", unsafe_allow_html=True)
        if input_method == "📁 Upload from device":
            st.info("👆 Please upload a signage image to begin verification")
        else:
            st.info("👆 Please enter an image URL to begin verification")
            st.markdown("""
            **Tips for image URLs:**
            - Use direct links to images (ending in .jpg, .png, etc.)
            - Right-click on an image → "Copy image address"
            - Make sure the URL is publicly accessible
            """)
            
            # Example URLs for testing
            with st.expander("📝 Example URLs for testing"):
                st.code("https://example.com/signage1.jpg")
                st.code("https://example.com/billboard.png")
                st.write("*Replace with actual image URLs*")
    
    # Admin section
    admin_section()

if __name__ == "__main__":
    main()
