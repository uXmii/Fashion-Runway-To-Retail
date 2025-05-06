import json
import base64
import io
import numpy as np
import time
import os
import shutil
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB3
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.preprocessing import image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.spatial.distance import mahalanobis
import pickle
from concurrent.futures import ThreadPoolExecutor

# Add this before loading your dataset
cache_dir = "similarity_cache"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    print(f"Cleared cache in {cache_dir}")

# Define clothing categories for classification
CLOTHING_CATEGORIES = [
    "hoodie", "sweatshirt", "jacket", "coat", "t-shirt", "shirt", 
    "sweater", "jeans", "pants", "shorts", "dress", "skirt"
]

# Define common colors for better color matching
COLOR_NAMES = {
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "navy": [0, 0, 128],
    "olive": [128, 128, 0],
    "purple": [128, 0, 128],
    "brown": [165, 42, 42],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203]
}

class ImprovedClothingSimilarity:
    def __init__(self, cache_dir="similarity_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models for feature extraction
        print("Initializing models...")
        self.style_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.efficient_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
        
        # Preprocessing for color analysis
        self.color_bins = 10  # Number of histogram bins per channel
        
        # PCA for dimensionality reduction
        self.pca = None
        
        # KD-Tree for fast nearest neighbor search
        self.kdtree = None
        
        # Covariance matrix for Mahalanobis distance
        self.inv_cov = None
        
        # Store the expected feature dimension
        self.expected_feature_dim = None
    
    def load_json_dataset(self, json_file_path):
        """Load and parse JSON file with error handling for various formats"""
        print(f"Loading dataset from {json_file_path}...")
        
        # Check if cached results exist
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(json_file_path)}_parsed.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached parsed dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        start_time = time.time()
        
        # Try different parsing approaches
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            try:
                # Try reading as JSONL
                data = []
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            data.append(item)
                        except:
                            continue
            except Exception as e:
                # Last resort: manual repair of JSON
                print(f"Attempting manual repair of JSON file: {str(e)}")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Try to fix common JSON issues
                content = content.replace('}\n{', '},\n{')  # Fix JSON Lines format
                content = content.rstrip(",\n ") + "]"  # Ensure proper array closing
                content = "[" + content if not content.startswith("[") else content
                
                try:
                    data = json.loads(content)
                except:
                    raise ValueError("Could not parse JSON file after repair attempts")
        
        # Normalize data structure
        if not isinstance(data, list):
            data = [data]
        
        # Add counters
        total_objects = len(data)
        objects_with_images = 0
        
        # Process each product
        parsed_data = []
        
        for i, product in enumerate(data):
            if 'image_data' in product and product['image_data']:
                try:
                    # Decode base64
                    image_data = base64.b64decode(product['image_data'])
                    
                    # Extract title for category detection
                    title = product.get('title', '').lower()
                    
                    # Extract metadata
                    meta = {
                        'index': i,
                        'asin': product.get('asin', f'product_{i}'),
                        'brand': product.get('brand', 'Unknown'),
                        'price': product.get('price', 'Unknown'),
                        'title': product.get('title', 'Unknown'),
                        'material': product.get('material', 'Unknown'),
                        'sustainability': product.get('sustainability', 'Unknown'),
                        'vegan_friendly': product.get('vegan_friendly', 'Unknown'),
                        'locally_made': product.get('locally_made', 'Unknown'),
                        'fair_wage': product.get('fair_wage', 'Unknown'),
                        'product_url': product.get('product_url', 'Unknown'),
                        'image_data': image_data  # Store binary image data
                    }
                    
                    # Analyze the title to detect category
                    for category in CLOTHING_CATEGORIES:
                        if category in title:
                            meta['category'] = category
                            break
                    else:
                        meta['category'] = 'unknown'
                    
                    parsed_data.append(meta)
                    objects_with_images += 1
                except Exception as e:
                    print(f"Error processing product {i}: {e}")
        
        print(f"Total objects in JSON: {total_objects}")
        print(f"Objects with valid images: {objects_with_images}")
        print(f"Parsed {len(parsed_data)} products with images (took {time.time() - start_time:.2f}s)")
        
        # Cache results
        with open(cache_path, 'wb') as f:
            pickle.dump(parsed_data, f)
        
        return parsed_data
    
    def get_image_from_data(self, image_data):
        """Convert binary image data to PIL Image with error handling"""
        try:
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {e}")
            # Return a small blank image as fallback
            return Image.new('RGB', (224, 224), color='gray')
    
    def detect_primary_colors(self, img, k=3):
        """Detect the primary colors in an image using K-means clustering"""
        # Convert to numpy array and reshape for k-means
        img_array = np.array(img.resize((100, 100)))  # Resize for faster processing
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calculate the percentage of each color
        counts = np.bincount(labels.flatten(), minlength=k)
        percentages = counts / counts.sum()
        
        # Sort colors by percentage
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_centers = centers[sorted_indices].astype(int)
        sorted_percentages = percentages[sorted_indices]
        
        # Get color names for dominant colors
        color_names = []
        for center in sorted_centers:
            # Find closest named color
            min_dist = float('inf')
            closest_color = "unknown"
            for name, rgb in COLOR_NAMES.items():
                dist = np.sqrt(np.sum((center - rgb)**2))
                if dist < min_dist:
                    min_dist = dist
                    closest_color = name
            color_names.append(closest_color)
        
        return sorted_centers, sorted_percentages, color_names
    
    def extract_color_features(self, img):
        """Extract enhanced color features from an image"""
        # RGB histogram features
        r_hist = cv2.calcHist([np.array(img)], [0], None, [self.color_bins], [0, 256]).flatten()
        g_hist = cv2.calcHist([np.array(img)], [1], None, [self.color_bins], [0, 256]).flatten()
        b_hist = cv2.calcHist([np.array(img)], [2], None, [self.color_bins], [0, 256]).flatten()
        
        # Normalize histograms
        r_hist = r_hist / r_hist.sum() if r_hist.sum() > 0 else r_hist
        g_hist = g_hist / g_hist.sum() if g_hist.sum() > 0 else g_hist
        b_hist = b_hist / b_hist.sum() if b_hist.sum() > 0 else b_hist
        
        # HSV color space for better color perception
        hsv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv_img], [0], None, [self.color_bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv_img], [1], None, [self.color_bins], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv_img], [2], None, [self.color_bins], [0, 256]).flatten()
        
        # Normalize HSV histograms
        h_hist = h_hist / h_hist.sum() if h_hist.sum() > 0 else h_hist
        s_hist = s_hist / s_hist.sum() if s_hist.sum() > 0 else s_hist
        v_hist = v_hist / v_hist.sum() if v_hist.sum() > 0 else v_hist
        
        # Detect primary colors
        centers, percentages, color_names = self.detect_primary_colors(img)
        
        # Create features for primary colors (top 3)
        primary_color_features = np.zeros(9)  # 3 colors x 3 channels (RGB)
        for i in range(min(3, len(centers))):
            if i < len(centers):
                primary_color_features[i*3:(i+1)*3] = centers[i] / 255.0
        
        # Combine all color features
        color_features = np.concatenate([
            r_hist * 0.5,               # Red channel (weighted)
            g_hist * 0.3,               # Green channel (weighted)
            b_hist * 0.2,               # Blue channel (weighted)
            h_hist * 0.7,               # Hue (more important for color perception)
            s_hist * 0.2,               # Saturation
            v_hist * 0.1,               # Value/Brightness
            primary_color_features * 3  # Primary colors (heavily weighted)
        ])
        
        return color_features, color_names
    
    def extract_category_features(self, img, title=None):
        """Extract features specific to clothing categories"""
        # Load the image and prepare it for processing
        img_array = np.array(img.resize((224, 224)))
        
        # Edge detection for shape features
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density for different regions (5x5 grid)
        h, w = edges.shape
        regions = 25  # 5x5 grid
        edge_features = np.zeros(regions)
        
        for i in range(5):
            for j in range(5):
                region = edges[i*h//5:(i+1)*h//5, j*w//5:(j+1)*w//5]
                edge_features[i*5+j] = np.sum(region > 0) / (region.shape[0] * region.shape[1])
        
        # Shape detection - aspect ratio, symmetry, and density
        aspect_ratio = h / w if w > 0 else 1.0
        
        # Horizontal symmetry (compare left and right halves)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)  # Flip horizontally
        
        # Adjust sizes if needed
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        symmetry_diff = np.abs(left_half[:, :min_width] - right_half_flipped[:, :min_width])
        horizontal_symmetry = 1 - (np.sum(symmetry_diff) / (255 * symmetry_diff.size))
        
        # Calculate shape density (ratio of non-zero edges to total pixels)
        shape_density = np.sum(edges > 0) / (h * w)
        
        shape_features = np.array([aspect_ratio, horizontal_symmetry, shape_density])
        
        # Texture features using GLCM
        glcm = cv2.normalize(cv2.calcHist([gray], [0], None, [8], [0, 256]), None, 0, 1, cv2.NORM_MINMAX)
        texture_features = glcm.flatten()
        
        # Category detection from title (if available)
        category_vector = np.zeros(len(CLOTHING_CATEGORIES))
        if title:
            title = title.lower()
            for i, category in enumerate(CLOTHING_CATEGORIES):
                if category in title:
                    category_vector[i] = 1.0
                    break
        
        # Combine all category-related features
        combined_features = np.concatenate([
            edge_features,
            shape_features,
            texture_features,
            category_vector
        ])
        
        return combined_features
    
    def extract_multiple_features(self, img, title=None):
        """Extract multiple types of features from an image"""
        # Resize image and convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 1. Deep features from multiple neural networks
        # Style features (MobileNet)
        img_array_mobilenet = preprocess_input_mobilenet(img_array.copy())
        style_features = self.style_model.predict(img_array_mobilenet, verbose=0)
        
        # Content features (ResNet)
        img_array_resnet = preprocess_input_resnet(img_array.copy())
        content_features = self.feature_model.predict(img_array_resnet, verbose=0)
        
        # Additional features (EfficientNet)
        img_array_efficient = preprocess_input_efficientnet(img_array.copy())
        efficient_features = self.efficient_model.predict(img_array_efficient, verbose=0)
        
        # 2. Enhanced color features with color name detection
        color_features, color_names = self.extract_color_features(img_resized)
        
        # 3. Category-specific features
        category_features = self.extract_category_features(img_resized, title)
        
        # Combine all features with adjusted weights
        combined_features = np.concatenate([
            color_features * 3.0,           # Color features (heavily weighted: 30%)
            category_features * 2.5,        # Category features (heavily weighted: 25%)
            content_features.flatten() * 2.0, # Content features (20%)
            style_features.flatten() * 1.5,   # Style features (15%)
            efficient_features.flatten() * 1.0 # Additional deep features (10%)
        ])
        
        # Store feature info
        feature_info = {
            'color_names': color_names,
            'primary_color': color_names[0] if color_names else 'unknown'
        }
        
        # Store the feature dimension if not already set
        if self.expected_feature_dim is None:
            self.expected_feature_dim = combined_features.shape[0]
            
        return combined_features, feature_info
    
    def extract_dataset_features(self, dataset, batch_size=10):
        """Extract features from all images in dataset with parallel processing"""
        cache_path = os.path.join(self.cache_dir, "dataset_features.pkl")
        feature_info_path = os.path.join(self.cache_dir, "feature_info.pkl")
        
        # Check if features are cached
        if os.path.exists(cache_path) and os.path.exists(feature_info_path):
            print("Loading cached feature vectors...")
            with open(cache_path, 'rb') as f:
                features = pickle.load(f)
            with open(feature_info_path, 'rb') as f:
                feature_info = pickle.load(f)
                
            # Store the expected feature dimension
            if features and len(features) > 0:
                self.expected_feature_dim = features[0].shape[0]
                print(f"Expected feature dimension: {self.expected_feature_dim}")
                
            return features, feature_info
                
        print(f"Extracting features from {len(dataset)} images...")
        start_time = time.time()
        
        all_features = []
        all_feature_info = []
        
        # Process in batches with ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                
                # Process each image in parallel
                futures = []
                for item in batch:
                    # Get the image from binary data
                    img = self.get_image_from_data(item['image_data'])
                    title = item.get('title', '')
                    futures.append(executor.submit(self.extract_multiple_features, img, title))
                
                # Collect results
                for future in futures:
                    features, feature_info = future.result()
                    all_features.append(features)
                    all_feature_info.append(feature_info)
                
                print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)} images...")
        
        print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
        
        # Store the expected feature dimension
        if all_features and len(all_features) > 0:
            self.expected_feature_dim = all_features[0].shape[0]
            print(f"Expected feature dimension: {self.expected_feature_dim}")
        
        # Cache the features
        with open(cache_path, 'wb') as f:
            pickle.dump(all_features, f)
        with open(feature_info_path, 'wb') as f:
            pickle.dump(all_feature_info, f)
        
        return all_features, all_feature_info
    
    def reduce_dimensions(self, features, n_components=150):
        """Reduce dimensionality of feature vectors using PCA"""
        if self.pca is None:
            print(f"Fitting PCA to reduce dimensions to {n_components}...")
            self.pca = PCA(n_components=n_components)
            reduced_features = self.pca.fit_transform(features)
            print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            reduced_features = self.pca.transform(features)
            
        return reduced_features
    
    def compute_covariance(self, features):
        """Compute inverse covariance matrix for Mahalanobis distance"""
        print("Computing covariance matrix for Mahalanobis distance...")
        cov_matrix = np.cov(features, rowvar=False)
        self.inv_cov = np.linalg.inv(cov_matrix)
        return self.inv_cov
    
    def build_index(self, features):
        """Build KD-Tree index for fast similarity search"""
        print("Building KD-Tree index for fast similarity search...")
        start_time = time.time()
        self.kdtree = cKDTree(features)
        print(f"Index built in {time.time() - start_time:.2f}s")
    
    def calculate_similarity_weights(self, query_feature_info, dataset_feature_info):
        """Calculate adaptive weights based on query image properties"""
        query_color = query_feature_info.get('primary_color', 'unknown')
        
        color_match_weights = []
        for item_info in dataset_feature_info:
            item_color = item_info.get('primary_color', 'unknown')
            
            # Higher weight if colors match
            if query_color == item_color:
                color_match_weights.append(3.0)  # Triple the weight for color match
            elif query_color != 'unknown' and item_color != 'unknown':
                # Check if any color in top 3 matches
                query_colors = query_feature_info.get('color_names', [])
                item_colors = item_info.get('color_names', [])
                
                if any(color in item_colors for color in query_colors):
                    color_match_weights.append(2.0)  # Double weight for partial color match
                else:
                    color_match_weights.append(1.0)  # Normal weight
            else:
                color_match_weights.append(1.0)  # Default weight
        
        return np.array(color_match_weights)
    
    def color_distance(self, color1, color2):
        """Calculate perceptual distance between colors"""
        # Convert to numpy arrays if they aren't already
        color1 = np.array(color1)
        color2 = np.array(color2)
        
        # Calculate Euclidean distance
        return np.sqrt(np.sum((color1 - color2)**2))
    
    def find_similar_images(self, query_features, query_feature_info, dataset, features, feature_info, top_n=30, filters=None):
        """Find similar images using KD-Tree and advanced ranking with adaptive weighting"""
        if self.kdtree is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Verify dataset and features have same length
        if len(dataset) != len(features):
            print(f"Error: Dataset size ({len(dataset)}) doesn't match feature count ({len(features)})")
            return []
            
        # Safety check - ensure we have data
        if len(dataset) == 0:
            print("Error: Empty dataset")
            return []
            
        # Find k nearest neighbors - get more than we need for filtering
        # Make sure we don't request more neighbors than exist in the dataset
        k = min(top_n * 10, len(dataset) - 1)  # Get 10x more to account for filtering
        if k <= 0:
            k = 1  # Always get at least one neighbor
            
        # Make sure query features is reshaped correctly
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
            
        # Query the KD-tree with appropriate k value
        distances, indices = self.kdtree.query(query_features, k=k)
            
        # Flatten arrays if they're 2D (single query point)
        if len(distances.shape) > 1:
            distances = distances[0]
            indices = indices[0]
            
        # Verify all indices are within bounds
        valid_mask = indices < len(features)
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]
            
        # If no valid indices, return empty list
        if len(valid_indices) == 0:
            print("No valid indices found")
            return []
            
        # Get the corresponding candidates
        candidates = []
        candidate_features = []
        candidate_info = []
        for idx in valid_indices:
            if idx < len(dataset) and idx < len(features) and idx < len(feature_info):
                candidates.append(dataset[idx])
                candidate_features.append(features[idx])
                candidate_info.append(feature_info[idx])
        
        # Calculate color similarity weights
        color_weights = self.calculate_similarity_weights(query_feature_info, candidate_info)
        
        # Calculate comprehensive similarity scores with adaptive weighting
        similarity_scores = []
        for i, (candidate, feature, info) in enumerate(zip(candidates, candidate_features, candidate_info)):
            # Get primary colors
            query_color = query_feature_info.get('primary_color', 'unknown')
            candidate_color = info.get('primary_color', 'unknown')
            
            # Calculate base similarity (inverse of distance)
            base_similarity = 1.0 / (1.0 + valid_distances[i])
            
            # Apply color weighting
            weighted_similarity = base_similarity * color_weights[i]
            
            # Apply title-based category matching
            query_title = query_feature_info.get('title', '').lower() if hasattr(query_feature_info, 'get') else ''
            candidate_title = candidate.get('title', '').lower()
            
            # Check for hoodie-specific matching
            if 'hoodie' in query_title and 'hoodie' in candidate_title:
                weighted_similarity *= 2.0  # Double score for hoodie-to-hoodie matches
            elif 'hoodie' in query_title and 'jacket' in candidate_title:
                weighted_similarity *= 0.5  # Halve score for hoodie-to-jacket matches
            
            # Color-specific boosting for black hoodies
            if 'black' in query_title and 'hoodie' in query_title:
                if 'black' in candidate_title and 'hoodie' in candidate_title:
                    weighted_similarity *= 3.0  # Triple score for black hoodie matches
                elif 'black' not in candidate_title:
                    weighted_similarity *= 0.3  # Severely penalize non-black items
            
            similarity_scores.append(weighted_similarity)
        
        # Sort by similarity scores
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
        
        # Apply filters and create result list
        results = []
        for idx in sorted_indices:
            if len(results) >= top_n:
                break
                
            candidate = candidates[idx]
            candidate_copy = candidate.copy()
            candidate_copy['similarity_score'] = float(similarity_scores[idx])
            candidate_copy['color_match'] = color_weights[idx] > 1.0
            
            # Apply filters
            if filters:
                skip_item = False
                
                # Material filter
                if 'material' in filters and filters['material']:
                    if candidate_copy.get('material') != filters['material']:
                        skip_item = True
                
                # Sustainability filter
                if 'sustainability' in filters and filters['sustainability']:
                    if candidate_copy.get('sustainability') != filters['sustainability']:
                        skip_item = True
                
                # Vegan-friendly filter
                if 'vegan_friendly' in filters and filters['vegan_friendly']:
                    if candidate_copy.get('vegan_friendly') != filters['vegan_friendly']:
                        skip_item = True
                
                # Locally-made filter
                if 'locally_made' in filters and filters['locally_made']:
                    if candidate_copy.get('locally_made') != filters['locally_made']:
                        skip_item = True
                
                # Fair-wage filter
                if 'fair_wage' in filters and filters['fair_wage']:
                    if candidate_copy.get('fair_wage') != filters['fair_wage']:
                        skip_item = True
                    
                # Brand filter
                if 'brand' in filters and filters['brand']:
                    if candidate_copy.get('brand') != filters['brand']:
                        skip_item = True
                    
                # Price range filter
                if 'price_range' in filters and filters['price_range']:
                    min_price, max_price = filters['price_range']
                    try:
                        # Extract numerical price if it's a string
                        if isinstance(candidate_copy['price'], str):
                            price_str = ''.join(c for c in candidate_copy['price'] if c.isdigit() or c == '.')
                            price = float(price_str)
                        else:
                            price = float(candidate_copy['price'])
                            
                        if price < min_price or price > max_price:
                            skip_item = True
                    except:
                        # Skip items with unparseable prices
                        skip_item = True
                        
                # Color filter
                if 'color' in filters and filters['color']:
                    if filters['color'] not in candidate_info[idx].get('color_names', []):
                        skip_item = True
                
                if skip_item:
                    continue
            
            results.append(candidate_copy)
        
        return results
    
    def preprocess_query_image(self, image_path):
        """Process a query image and extract features with dimension matching"""
        try:
            if isinstance(image_path, str):
                # Load image from file
                img = Image.open(image_path).convert('RGB')
                # Try to extract title from filename
                filename = os.path.basename(image_path)
                # Extract title part before extension
                title = os.path.splitext(filename)[0].replace('_', ' ').lower()
            else:
                # Assume image_path is already a PIL Image
                img = image_path.convert('RGB')
                title = ""
                
            # Extract features
            features, feature_info = self.extract_multiple_features(img, title)
            
            # Add title to feature info
            feature_info['title'] = title
            
            # Ensure dimensions match what PCA expects
            if self.expected_feature_dim is not None:
                actual_dim = features.shape[0]
                
                if self.expected_feature_dim != actual_dim:
                    print(f"Feature dimension mismatch. Making dimensions consistent...")
                    print(f"Expected {self.expected_feature_dim}, got {actual_dim}")
                    
                    # Ensure consistent dimensions by padding or truncating
                    if actual_dim > self.expected_feature_dim:
                        print("Truncating features to match expected dimensions")
                        features = features[:self.expected_feature_dim]  # Truncate to match
                    else:
                        print("Padding features to match expected dimensions")
                        # Pad with zeros
                        padded_features = np.zeros(self.expected_feature_dim)
                        padded_features[:actual_dim] = features
                        features = padded_features
            
            # Reduce dimensions (if PCA is fitted)
            if self.pca is not None:
                features = self.pca.transform(features.reshape(1, -1))
                
            return features, feature_info, img
        except Exception as e:
            print(f"Error processing query image: {e}")
            print("Using default image instead...")
            # Create a default gray image as fallback
            query_img = Image.new('RGB', (224, 224), color='gray')
            
            # Extract features for the default image with dimension handling
            features, feature_info = self.extract_multiple_features(query_img)
            
            # Handle dimensions as above
            if self.expected_feature_dim is not None:
                actual_dim = features.shape[0]
                
                if self.expected_feature_dim != actual_dim:
                    if actual_dim > self.expected_feature_dim:
                        features = features[:self.expected_feature_dim]
                    else:
                        padded_features = np.zeros(self.expected_feature_dim)
                        padded_features[:actual_dim] = features
                        features = padded_features
            
            # Reduce dimensions (if PCA is fitted)
            if self.pca is not None:
                features = self.pca.transform(features.reshape(1, -1))
                
            return features, feature_info, query_img
    
    def prepare_dataset(self, json_file_path):
        """Load, parse, and prepare dataset for similarity search"""
        # Load and parse dataset
        dataset = self.load_json_dataset(json_file_path)
        
        # Extract features
        features, feature_info = self.extract_dataset_features(dataset)
        
        # Check dataset and features have the same length
        if len(dataset) != len(features):
            print(f"WARNING: Dataset size ({len(dataset)}) doesn't match feature count ({len(features)})")
            # Adjust to smallest size to avoid index errors
            min_size = min(len(dataset), len(features))
            dataset = dataset[:min_size]
            features = features[:min_size]
            feature_info = feature_info[:min_size]
            print(f"Adjusted to common size: {min_size}")
        
        # Reduce dimensions
        reduced_features = self.reduce_dimensions(features)
        
        # Build index
        self.build_index(reduced_features)
        
        # Compute covariance matrix for Mahalanobis distance
        try:
            self.compute_covariance(reduced_features)
        except Exception as e:
            print(f"Warning: Could not compute covariance matrix: {e}")
            self.inv_cov = None
        
        return dataset, reduced_features, feature_info
    
    def display_results(self, query_img, results, figsize=(20, 16)):
        """Display query image and top 30 similar items with their details and rankings"""
        n = len(results)
        
        if n == 0:
            print("No results to display.")
            return
            
        # Determine grid layout based on number of results
        rows = min(7, (n // 5) + 2)  # +2 for query row
        
        # Create figure with grid layout
        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(rows, 5, figure=fig)
        
        # Query image takes up top left position
        ax_query = fig.add_subplot(grid[0, 0:2])
        ax_query.imshow(query_img)
        ax_query.set_title("Query Image", fontsize=14)
        ax_query.axis('off')
        
        # Show color palette of query image in top right
        ax_palette = fig.add_subplot(grid[0, 2:])
        self._display_color_palette(query_img, ax_palette)
        
        # Display results as a grid
        for i, result in enumerate(results):
            if i >= 30:  # Show max 30 results
                break
                
            row = (i // 5) + 1  # 5 items per row
            col = i % 5
            
            if row < rows:  # Make sure we don't exceed the grid
                ax = fig.add_subplot(grid[row, col])
                
                # Get the image from binary data
                img = self.get_image_from_data(result['image_data'])
                ax.imshow(img)
                
                # Format details for display
                price = result.get('price', 'Unknown')
                material = result.get('material', 'Unknown')
                sustainability = result.get('sustainability', 'Unknown')
                vegan = result.get('vegan_friendly', 'Unknown')
                local = result.get('locally_made', 'Unknown')
                fair = result.get('fair_wage', 'Unknown')
                color_match = result.get('color_match', False)
                product_url = result.get('product_url', 'Unknown')
                image = result.get('image_data', 'Unknown')
                
                # Create title with rank, score and key details
                title = f"#{i+1} Score: {result['similarity_score']:.2f}\n"
                title += f"{result['brand']}\n"
                
                # Add marker for color match
                if color_match:
                    title += "🎨 "  # Color match indicator
                
                title += f"{price}\n"
                title += f"Material: {material[:10]}...\n" if len(material) > 10 else f"Material: {material}\n"
                
                # Add sustainability info using symbols to save space
                eco_info = ""
                if vegan == "yes":
                    eco_info += "V+ "  # Vegan-friendly
                if local == "yes":
                    eco_info += "L+ "  # Locally-made
                if fair == "yes":
                    eco_info += "F+ "  # Fair-wage
                    
                if eco_info:
                    title += eco_info
                
                ax.set_title(title, fontsize=9)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Display sustainability details in a separate table
        if n > 0:
            self._display_sustainability_table(results[:30] if len(results) >= 30 else results)
    
    def _display_color_palette(self, img, ax):
        """Extract and display dominant colors"""
        # Detect primary colors
        centers, percentages, color_names = self.detect_primary_colors(img, k=5)
        
        # Create color bars
        ax.axis('off')
        ax.set_title("Dominant Colors", fontsize=14)
        
        # Create horizontal color bars
        for i, (center, percentage, color_name) in enumerate(zip(centers, percentages, color_names)):
            # Ensure values are clipped to 0-1 range
            color = np.clip(center / 255.0, 0, 1)
            
            # Create rectangle for color
            height = 1.0 / len(centers)
            y = i * height
            rect = plt.Rectangle((0, y), percentage, height, color=color)
            ax.add_patch(rect)
            
            # Add percentage text and color name
            text_x = percentage + 0.02
            text_y = y + height/2
            ax.text(text_x, text_y, f"{color_name} ({percentage*100:.1f}%)", 
                   va='center', ha='left', fontsize=10)
        
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1)
    
    def _display_sustainability_table(self, results):
        """Display a table with detailed sustainability information"""
        if not results:
            return
            
        # Create a new figure for the table
        plt.figure(figsize=(15, len(results) * 0.4 + 1))
        
        # Prepare table data
        table_data = []
        for i, item in enumerate(results):
            sustainability = item.get('sustainability', 'Unknown')
            material = item.get('material', 'Unknown')
            vegan = item.get('vegan_friendly', 'Unknown')
            local = item.get('locally_made', 'Unknown')
            fair = item.get('fair_wage', 'Unknown')
            
            row = [
                f"#{i+1}",
                item.get('brand', 'Unknown'),
                f"{item['similarity_score']:.2f}",
                material,
                sustainability,
                vegan,
                local,
                fair
            ]
            table_data.append(row)
        
        # Create table
        plt.table(
            cellText=table_data,
            colLabels=['Rank', 'Brand', 'Score', 'Material', 'Sustainability', 'Vegan', 'Local', 'Fair Wage'],
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2'] * 8,
            colWidths=[0.05, 0.15, 0.05, 0.15, 0.3, 0.1, 0.1, 0.1]
        )
        
        plt.title('Detailed Sustainability Information', fontsize=16)
        plt.axis('off')  # Hide axis
        plt.tight_layout()
        plt.show()