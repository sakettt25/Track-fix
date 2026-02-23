import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import io

# Page Configuration
st.set_page_config(
    page_title="Track-fix: Railway Defect Classifier",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Header
st.title("🚂 Track-fix: Railway Defect Classifier")
st.markdown("""
**An Explainable AI System for Railway Surface Defect Detection**

Powered by ResNet18 + Grad-CAM | Real-time Defect Classification & Visualization
""")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("rail_crack_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def setup_gradcam(_model):
    return GradCAM(model=_model, target_layers=[_model.layer4[-1]])

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")

# Sample Image Option
use_sample = st.sidebar.checkbox("📷 Use Sample Image", value=False, help="Load a default railway track image for testing")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_heatmap = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
show_raw_image = st.sidebar.checkbox("Show Raw Image", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Model Information
- **Architecture**: ResNet18
- **Weights**: ImageNet1K
- **Input Size**: 224×224 RGB
- **Classes**: Defective / Non-Defective
- **Device**: """ + str(device).upper() + """
""")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a railway track image",
        type=["jpg", "png", "jpeg", "webp"],
        help="Supported formats: JPG, PNG, JPEG, WebP",
        disabled=use_sample
    )

# Determine image source
if use_sample:
    # Load sample image
    try:
        image = Image.open("test_image.png").convert("RGB")
        img_array = np.array(image)
        uploaded_file_size = None
        image_source = "Sample Image"
    except FileNotFoundError:
        st.error("❌ Sample image 'test_image.png' not found. Please upload an image instead.")
        image = None
elif uploaded_file is not None:
    # Load and preprocess uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    uploaded_file_size = uploaded_file.size
    image_source = "Uploaded Image"
else:
    image = None

if image is not None:
    # Get file size info
    size_info = f" | Size: {uploaded_file_size / 1024:.1f} KB" if uploaded_file_size else ""
    
    # Display Input Image
    if show_raw_image:
        with col1:
            st.image(image, caption=image_source, use_column_width=True)
            st.caption(f"Resolution: {image.size[0]}×{image.size[1]}{size_info}")
    
    # Run Prediction
    st.text("🔄 Processing...")
    
    try:
        model = load_model()
        cam = setup_gradcam(model)
        
        # Prepare input tensor
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, 1).item()
            confidence = probabilities[prediction].item()
        
        # Determine result
        is_defective = prediction == 0
        label = "🔴 DEFECTIVE" if is_defective else "🟢 NON-DEFECTIVE"
        
        # Grad-CAM Heatmap
        if show_heatmap:
            st.subheader("🔍 Grad-CAM Explainability")
            
            with st.spinner("Generating Grad-CAM heatmap..."):
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                rgb_image = np.array(image.resize((224, 224))) / 255.0
                heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
                
                col_hm1, col_hm2 = st.columns([1, 1])
                
                with col_hm1:
                    st.image(rgb_image, caption="Original Image (224×224)", use_column_width=True)
                
                with col_hm2:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
                    st.caption("🔴 Red = High influence | 🔵 Blue = Low influence")
            
            # Save heatmap
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", heatmap_bgr)
            heatmap_bytes = buffer.tobytes()
            
            st.download_button(
                label="⬇️ Download Heatmap",
                data=heatmap_bytes,
                file_name="gradcam_heatmap.jpg",
                mime="image/jpeg"
            )
        
        # Interpretation Guide
        with st.expander("📖 How to Interpret Results"):
            st.markdown("""
            ### Understanding the Classification
            
            **🔴 DEFECTIVE**: Railway track surface shows signs of defects such as:
            - Surface cracks
            - Rust or corrosion
            - Deformation or wear
            - Other structural damage
            
            **🟢 NON-DEFECTIVE**: Railway track surface is in good condition with:
            - No visible cracks
            - Proper surface condition
            - No significant wear
            
            ### Grad-CAM Heatmap
            - **Red regions**: Areas the model focused on (high influence on prediction)
            - **Yellow regions**: Moderate influence areas
            - **Blue regions**: Less influential areas
            
            ### Confidence Score
            - The percentage indicates model certainty
            - Higher confidence = more reliable prediction
            - Use with caution if confidence < 70%
            """)
        
        st.success("✅ Processing complete!")
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

else:
    st.info("👆 Upload an image or use the sample image to get started!")
    st.markdown("""
    ### How to Use
    1. **Option A**: Click "Browse files" to upload a railway track image
    2. **Option B**: Check "📷 Use Sample Image" in the sidebar to test with a default image
    3. View the Grad-CAM heatmap to understand the model's decision
    4. Download the heatmap if needed
    
    ### Supported Image Formats
    - JPG/JPEG
    - PNG
    - WebP
    """)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: gray;">
Track-fix | Built with PyTorch, Streamlit & Grad-CAM
</p>
""", unsafe_allow_html=True)
