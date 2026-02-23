# 🚂 Track-fix - Streamlit Web Application

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Virtual environment activated

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the Streamlit app**:

#### Option 1: Direct Command
```bash
streamlit run app.py
```

#### Option 2: Using Batch Script (Windows CMD)
```bash
run_app.bat
```

#### Option 3: Using PowerShell Script (Windows PowerShell)
```powershell
.\run_app.ps1
```

### Access the Application

Once running, open your browser and navigate to:
```
http://localhost:8501
```

## Features

✅ **Image Upload**: Drag & drop or browse railway track images  
✅ **Real-time Classification**: Defective vs Non-Defective prediction  
✅ **Confidence Scores**: View classification probability  
✅ **Grad-CAM Visualization**: See which regions influenced the decision  
✅ **Download Results**: Export heatmaps for further analysis  
✅ **Interactive UI**: Adjust settings and thresholds in real-time

## Supported Image Formats
- JPG / JPEG
- PNG
- WebP

## Model Architecture
- **Backbone**: ResNet18 (ImageNet1K pretrained)
- **Classes**: 2 (Defective / Non-Defective)
- **Input Resolution**: 224×224 RGB
- **Explainability**: Grad-CAM heatmaps

## Project Structure
```
gradcam-railway-track-defect-main/
├── app.py                          # Streamlit web application
├── main.py                         # Single image inference script
├── test.py                         # Basic test script
├── test_gradcam_batch.py          # Batch processing script
├── train.py                        # Model training script
├── create_model.py                 # Model initialization
├── rail_crack_model.pth           # Trained model weights
├── run_app.bat                    # Windows batch launcher
├── run_app.ps1                    # PowerShell launcher
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## Troubleshooting

### Issue: Module not found error
**Solution**: Ensure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Port 8501 already in use
**Solution**: Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: CUDA/GPU errors
**Solution**: The app automatically uses CPU if GPU is unavailable.

## Notes
- First run may take time as it downloads the ImageNet model weights (~44MB)
- The app caches the model for faster subsequent runs
- GPU acceleration available if CUDA is properly installed

For technical support, refer to the main README.md file.
