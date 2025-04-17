from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import shutil
import tempfile
import time
import json
import base64
from PIL import Image
import numpy as np
import io

# Import the improved clothing similarity model
from IRProject import ImprovedClothingSimilarity

# Initialize the app
app = FastAPI(
    title="Fashion Runway to Retail API",
    description="AI-Driven Fashion Retrieval System for finding affordable alternatives to high-fashion looks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
similarity_engine = None
dataset = None
features = None
feature_info = None
brands = []
materials = []
colors = []
sustainability_practices = []
price_range = (0, 1000)

# Models for API responses
class FilterOptions(BaseModel):
    brands: List[str]
    materials: List[str]
    colors: List[str]
    sustainability_practices: List[str]
    price_range: Dict[str, float]
    vegan_friendly: List[str] = ["yes", "no"]
    locally_made: List[str] = ["yes", "no"]
    fair_wage: List[str] = ["yes", "no"]

class ProductDetails(BaseModel):
    rank: int
    similarity_score: float
    color_match: bool
    asin: str
    brand: str
    title: str
    price: str
    material: str
    sustainability: str
    vegan_friendly: str
    locally_made: str
    fair_wage: str
    image_url: str
    product_url: str

class SimilarityResponse(BaseModel):
    results: List[ProductDetails]
    query_image: str  # Base64 encoded image
    total_results: int
    processing_time: float

# Load the dataset and initialize the model
@app.on_event("startup")
async def startup_event():
    global similarity_engine, dataset, features, feature_info, brands, materials, colors, sustainability_practices, price_range
    
    # Path to JSON dataset - change this to your dataset path
    json_path = os.environ.get("C:/Users/HP/Downloads/IR_Project", "dataset.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: Dataset not found at {json_path}. API will not work until dataset is loaded.")
        return
    
    # Initialize the similarity engine
    similarity_engine = ImprovedClothingSimilarity()
    
    # Load and prepare the dataset
    print(f"Loading dataset from {json_path}...")
    dataset, features, feature_info = similarity_engine.prepare_dataset(json_path)
    print(f"Dataset loaded with {len(dataset)} items")
    
    # Extract filter options
    brands = sorted(list(set(item.get('brand', '') for item in dataset if item.get('brand'))))
    materials = sorted(list(set(item.get('material', '') for item in dataset if item.get('material'))))
    
    colors = set()
    for info in feature_info:
        for color in info.get('color_names', []):
            colors.add(color)
    colors = sorted(list(colors))
    
    sustainability_practices = sorted(list(set(item.get('sustainability', '') 
                                            for item in dataset if item.get('sustainability'))))
    
    # Calculate price range
    prices = []
    for item in dataset:
        try:
            if 'price' in item and item['price']:
                if isinstance(item['price'], str):
                    price_str = ''.join(c for c in item['price'] if c.isdigit() or c == '.')
                    if price_str:
                        price = float(price_str)
                        prices.append(price)
                elif isinstance(item['price'], (int, float)):
                    prices.append(float(item['price']))
        except:
            continue
    
    if prices:
        price_range = (min(prices), max(prices))
    
    print("API ready to use!")

# API endpoint to get filter options
@app.get("/filters", response_model=FilterOptions)
async def get_filters():
    """
    Get available filter options for the fashion similarity search
    """
    if similarity_engine is None:
        raise HTTPException(status_code=503, detail="Service is initializing. Please try again later.")
    
    return FilterOptions(
        brands=brands,
        materials=materials,
        colors=colors,
        sustainability_practices=sustainability_practices,
        price_range={"min": price_range[0], "max": price_range[1]},
    )

# API endpoint for similarity search
@app.post("/search", response_model=SimilarityResponse)
async def search_by_image(
    image: UploadFile = File(...),
    top_n: int = Query(30, ge=1, le=100),
    brand: Optional[str] = Form(None),
    material: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    sustainability: Optional[str] = Form(None),
    vegan_friendly: Optional[str] = Form(None),
    locally_made: Optional[str] = Form(None),
    fair_wage: Optional[str] = Form(None),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None)
):
    """
    Search for similar fashion items by uploading an image.
    Returns the top N matching products with similarity scores and product details.
    """
    if similarity_engine is None or dataset is None or features is None:
        raise HTTPException(status_code=503, detail="Service is initializing. Please try again later.")
    
    start_time = time.time()
    
    # Process filters
    filters = {}
    if brand:
        filters['brand'] = brand
    if material:
        filters['material'] = material
    if color:
        filters['color'] = color
    if sustainability:
        filters['sustainability'] = sustainability
    if vegan_friendly:
        filters['vegan_friendly'] = vegan_friendly
    if locally_made:
        filters['locally_made'] = locally_made
    if fair_wage:
        filters['fair_wage'] = fair_wage
    if min_price is not None and max_price is not None:
        filters['price_range'] = (min_price, max_price)
    
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Save the upload file temporarily
        shutil.copyfileobj(image.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the image and get results
        query_features, query_feature_info, query_img = similarity_engine.preprocess_query_image(temp_file_path)
        
        # Get similar items
        results = similarity_engine.find_similar_images(
            query_features, query_feature_info, dataset, features, feature_info, top_n=top_n, filters=filters
        )

         # Debug first result (if any)
        if results:
            first_result_keys = list(results[0].keys())
            print(f"First result keys: {first_result_keys}")
        
        
        # Convert query image to base64 for response
        buffered = io.BytesIO()
        query_img.save(buffered, format="JPEG")
        query_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            # Get product URL
            product_url = result.get('product_url', '')
            
            # Format the product details
            product_detail = ProductDetails(
                rank=i+1,
                similarity_score=float(result.get('similarity_score', 0.0)),
                color_match=result.get('color_match', False),
                asin=result.get('asin', ''),
                brand=result.get('brand', 'Unknown'),
                title=result.get('title', 'Unknown'),
                price=result.get('price', 'Unknown'),
                material=result.get('material', 'Unknown'),
                sustainability=result.get('sustainability', 'Unknown'),
                vegan_friendly=result.get('vegan_friendly', 'Unknown'),
                locally_made=result.get('locally_made', 'Unknown'),
                fair_wage=result.get('fair_wage', 'Unknown'),
                image_url=base64.b64encode(result.get('image_data')).decode('utf-8') 
                    if isinstance(result.get('image_data'), bytes) 
                    else result.get('image_data', ''),
                product_url=product_url
            )
            formatted_results.append(product_detail)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the results
        return SimilarityResponse(
            results=formatted_results,
            query_image=query_img_base64,
            total_results=len(formatted_results),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {"status": "healthy", "model_loaded": similarity_engine is not None}

if __name__ == "__main__":
    # Run the FastAPI using Uvicorn
    uvicorn.run("API_IR:app", host="127.0.0.1", port=8000, reload=True)