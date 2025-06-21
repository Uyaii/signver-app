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

def main():
    # ! Header
    st.markdown('<h1 class="main-header">💎 SIGNVER - SIGNAGE VERIFICATION</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 18px;">Upload a signage image to verify compliance with LASAA regulations</p>', unsafe_allow_html=True)
    
    # ! Load model and embeddings
    model = load_model()
    approved_embeddings, approved_image_names = load_approved_embeddings()
    
    # ! Check if embeddings are available (system check)
    if approved_embeddings is None:
        st.markdown("""
        <div class="system-error">
            <h2>⚠️ SYSTEM ERROR</h2>
            <p><strong>Approved signage database not found.</strong></p>
            <p>Please contact the system administrator.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("Developer Note: Make sure 'approved_embeddings.pkl' and 'approved_names.pkl' files are present in the app directory.")
        return
    
    #! Show system status
    st.success(f"✅ System Ready - {len(approved_embeddings)} approved signage references loaded")
    
    # ! Main upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Signage Image")
    
    # ! Image input options
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
    
    else:  # ! URL input
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
        # ! Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Your Signage")
            st.image(img_pil, use_container_width=True)
            if input_method == "🌐 Use image URL":
                st.caption(f"Source: {image_source}")
            else:
                st.caption(f"File: {image_source}")
        
        # ! Process image
        with st.spinner("🔍 Analyzing signage for compliance..."):
            # ! Get embedding
            uploaded_embedding = get_embedding(img_pil, model).reshape(1, -1)
            
            # ! Calculate similarities
            similarities = cosine_similarity(uploaded_embedding, approved_embeddings)
            distances = cdist(uploaded_embedding, approved_embeddings, metric='euclidean')
            
            # ! Find best match
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[0][best_match_index]
            best_distance = distances[0][best_match_index]
            
            # ! Decision
            is_approved = best_similarity >= SIMILARITY_THRESHOLD
        
        with col2:
            st.subheader("📊 Verification Result")
            
            # ! Display result
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
            
            # ! Detailed metrics
            st.subheader("📈 Analysis Details")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Similarity Score", f"{best_similarity:.4f}", f"Required: {SIMILARITY_THRESHOLD}")
                st.metric("Best Match", approved_image_names[best_match_index])
            
            with col_b:
                st.metric("Distance Score", f"{best_distance:.4f}")
                st.metric("Database Size", f"{len(approved_embeddings)} references")
        
        # ! Detailed analysis section
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("🔬 Advanced Analysis (Optional)"):
            # ! Similarity distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # ! Similarity histogram
            ax1.hist(similarities[0], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(x=best_similarity, color='red', linestyle='--', linewidth=2, label=f'Your Result ({best_similarity:.4f})')
            ax1.axvline(x=SIMILARITY_THRESHOLD, color='green', linestyle='--', linewidth=2, label=f'Approval Threshold ({SIMILARITY_THRESHOLD})')
            ax1.set_title('Similarity Distribution')
            ax1.set_xlabel('Cosine Similarity')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ! Distance histogram
            ax2.hist(distances[0], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
            ax2.axvline(x=best_distance, color='red', linestyle='--', linewidth=2, label=f'Your Result ({best_distance:.4f})')
            ax2.set_title('Distance Distribution')
            ax2.set_xlabel('Euclidean Distance')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ! Top 5 matches
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
            
            # ! Example URLs for testing
            with st.expander("📝 Example URLs for testing"):
                st.code("https://example.com/signage1.jpg")
                st.code("https://example.com/billboard.png")
                st.write("*Replace with actual image URLs*")

if __name__ == "__main__":
    main()
