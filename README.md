# 🧠 MRI Image Classification

A Streamlit web application that uses a MobileNetV2 deep learning model to classify images as MRI scans or non-MRI images.

## Features

- 🔍 Binary classification (MRI vs Non-MRI)
- 📊 Confidence score display
- 🎯 Real-time image analysis
- 📱 Responsive web interface

## Model Details

- **Architecture:** MobileNetV2
- **Framework:** TensorFlow 2.10.1
- **Input Size:** 224x224 RGB images
- **Task:** Binary Classification

## Deployment

This app is deployed on [Streamlit Cloud](https://streamlit.io/cloud).

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## File Structure
```
.
├── app.py                      # Main Streamlit application
├── MobileNet_Binary_MRI.keras  # Trained model (21.10 MB)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version
├── packages.txt                # System dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── .gitignore                 # Git ignore patterns
└── README.md                  # This file
```

## Environment

- **Python:** 3.10
- **TensorFlow:** 2.10.1 (CPU)
- **Streamlit:** Latest compatible version

## Important Notes

⚠️ **This is for educational purposes only.**

- NOT for medical diagnosis
- Always consult healthcare professionals
- Results should not replace professional medical advice

## License

[Your License Here]

## Author

[Your Name]
```

---

## **Your Complete File Structure Should Be:**
```
your_project/
├── app.py                          ✅ Main application
├── MobileNet_Binary_MRI.keras      ✅ Your trained model
├── requirements.txt                ✅ Dependencies
├── runtime.txt                     ✅ Python version
├── packages.txt                    ✅ System packages
├── .streamlit/
│   └── config.toml                ✅ Streamlit config
├── .gitignore                      ✅ Git ignore
└── README.md                       ✅ Documentation
