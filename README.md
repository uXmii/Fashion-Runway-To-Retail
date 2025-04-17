# Fashion-Runway-To-Retail
 Image-Based Fashion Retrieval Using Deep Visual Similarity and Ethical Filtering

# Project Overview
This project is an AI-driven fashion retrieval system that helps users find affordable alternatives to high-fashion or runway items by uploading a reference image. The system uses a combination of deep learning-based visual embeddings, color and category analysis, and ethically-aware reranking to return similar products scraped from Amazon. Users can further filter results by price, material, sustainability, and ethical attributes like vegan, locally made, and fair wage.

# Core Features
Upload a fashion image (runway, celebrity, or influencer style)
Deep visual similarity with ResNet50, EfficientNetB3, and MobileNetV2
Metadata-aware filtering: material, brand, sustainability tags
Reranking based on color match, category alignment, and ethical filters
Real-time results using KD-Tree and PCA for fast indexing
React frontend + FastAPI backend with clean UI and responsive search

# Technologies Used

Component	Stack
Frontend	React.js, TailwindCSS
Backend	FastAPI, Uvicorn, Pydantic
Deep Learning	TensorFlow, ResNet50, MobileNetV2, EfficientNetB3
Data Processing	NumPy, OpenCV, PIL, Scikit-learn, PCA
Deployment	Docker-ready / Localhost setup

# Dataset
We scraped a curated collection of ~1,000 Amazon fashion items using a custom crawler. Each product entry includes:
Product title and brand
Image (base64 encoded)
Price
Material and sustainability tags (e.g., vegan, fair wage)
Product URL for reference

# How It Works
User uploads an image via the React interface
The image is sent to the backend where:
Visual features are extracted
Similar items are retrieved using KD-Tree
Reranked using filters and heuristics
Filtered, ranked results are returned with metadata and visual cues
