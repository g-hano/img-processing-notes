# Signature Analysis Pipeline Technical Documentation

## 1. Advanced Preprocessing and Image Normalization

### 1.1 RGB Conversion and Format Standardization
The pipeline starts by ensuring consistent image format through RGB conversion:

```python
def ensure_rgb(self, image):
    """Convert image to RGB if it's not already"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # RGB
            if image.dtype == np.uint8:
                return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif isinstance(image, Image.Image):
        return image.convert('RGB')
    return image
```

### 1.2 Size Normalization
Implements aspect ratio preservation during resizing:

```python
def normalize_size(self, image1, image2):
    """Resize images to same dimensions while preserving aspect ratio"""
    # Convert to PIL images if needed
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1)
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2)
        
    w1, h1 = image1.size
    w2, h2 = image2.size
    
    # Calculate target size based on aspect ratios
    aspect_ratio1 = w1 / h1
    aspect_ratio2 = w2 / h2
    
    if aspect_ratio1 > aspect_ratio2:
        new_width = min(w1, w2)
        new_height1 = int(new_width / aspect_ratio1)
        new_height2 = int(new_width / aspect_ratio2)
        new_width1 = new_width2 = new_width
    else:
        new_height = min(h1, h2)
        new_width1 = int(new_height * aspect_ratio1)
        new_width2 = int(new_height * aspect_ratio2)
        new_height1 = new_height2 = new_height
        
    # Resize using high-quality LANCZOS resampling
    return (
        image1.resize((new_width1, new_height1), Image.Resampling.LANCZOS),
        image2.resize((new_width2, new_height2), Image.Resampling.LANCZOS)
    )
```

### 1.3 Image Enhancement Pipeline
Advanced preprocessing steps for optimal feature extraction:

```python
def preprocess_image(self, image):
    """Enhance image quality for better matching"""
    # Convert to numpy array for OpenCV operations
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Non-Local Means denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove small noise with morphological opening
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Convert back to RGB for CLIP
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cleaned_rgb)
```

## 2. CLIP-based Semantic Similarity Analysis

### 2.1 CLIP Feature Extraction
Implementation of CLIP-based similarity measurement:

```python
def calculate_clip_similarity(self, image1, image2):
    """Calculate CLIP similarity for normalized images"""
    # Normalize sizes and resize to CLIP's target size
    norm_img1, norm_img2 = self.normalize_size(image1, image2)
    img1_resized = norm_img1.resize(self.target_size, Image.Resampling.LANCZOS)
    img2_resized = norm_img2.resize(self.target_size, Image.Resampling.LANCZOS)
    
    # Process images through CLIP
    inputs1 = self.clip_processor(images=img1_resized, return_tensors="pt", padding=True)
    inputs2 = self.clip_processor(images=img2_resized, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Extract features
        emb1 = self.clip_model.get_image_features(**inputs1)
        emb2 = self.clip_model.get_image_features(**inputs2)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(emb1, emb2).item()
    
    # Normalize similarity score
    return max(0, (similarity - self.clip_threshold) / (1 - self.clip_threshold))
```

## 3. Shape and Structural Analysis

### 3.1 HOG and SSIM Analysis
Combined shape-based similarity analysis:

```python
def calculate_shape_similarity(self, img1_array, img2_array):
    """Calculate shape-based similarities using HOG and SSIM"""
    # Convert to grayscale if needed
    if len(img1_array.shape) == 3:
        img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1_array
        img2_gray = img2_array
    
    # Ensure same size for comparison
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    target_size = (min(w1, w2), min(h1, h2))
    
    img1_resized = cv2.resize(img1_gray, target_size)
    img2_resized = cv2.resize(img2_gray, target_size)
    
    # Calculate HOG features
    hog_1 = hog(img1_resized, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=False)
    hog_2 = hog(img2_resized, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=False)
    
    # Calculate HOG similarity using normalized dot product
    hog_similarity = np.dot(hog_1, hog_2) / (np.linalg.norm(hog_1) * np.linalg.norm(hog_2))
    
    # Calculate structural similarity
    ssim_score = ssim(img1_resized, img2_resized)
    
    # Return average of HOG and SSIM similarities
    return (hog_similarity + ssim_score) / 2
```

### 3.2 Combined Similarity Metrics
Integration of different similarity measures:

```python
def calculate_similarity_metrics(self, img1, img2):
    """Calculate combined similarity metrics"""
    # Ensure RGB images
    img1 = self.ensure_rgb(img1)
    img2 = self.ensure_rgb(img2)
    
    # Normalize sizes
    norm_img1, norm_img2 = self.normalize_size(img1, img2)
    
    # Preprocess images
    proc_img1 = self.preprocess_image(norm_img1)
    proc_img2 = self.preprocess_image(norm_img2)
    
    # Calculate similarities
    clip_sim = self.calculate_clip_similarity(proc_img1, proc_img2)
    shape_sim = self.calculate_shape_similarity(
        np.array(proc_img1), 
        np.array(proc_img2)
    )
    
    # Return weighted combination
    return {
        'clip': clip_sim,
        'shape': shape_sim,
        'combined': 0.7 * clip_sim + 0.3 * shape_sim
    }
```

## 4. Usage Example

```python
# Initialize the similarity analyzer
analyzer = SignatureSimilarity(clip_threshold=0.8)

# Load images (can be PIL Images, numpy arrays, or file paths)
image1 = Image.open('signature1.png')
image2 = Image.open('signature2.png')

# Calculate similarity metrics
metrics = analyzer.calculate_similarity_metrics(image1, image2)

# Print results
print(f"CLIP Similarity: {metrics['clip']:.3f}")
print(f"Shape Similarity: {metrics['shape']:.3f}")
print(f"Combined Score: {metrics['combined']:.3f}")
```

The pipeline combines advanced image preprocessing, CLIP-based semantic analysis, and traditional computer vision techniques to provide robust signature similarity assessment. Key features include:

1. Sophisticated preprocessing:
   - Noise reduction
   - Adaptive thresholding
   - Morphological cleaning
   - Size normalization

2. Multi-modal similarity analysis:
   - CLIP embeddings for high-level features
   - HOG features for shape analysis
   - SSIM for structural comparison

3. Weighted combination of metrics:
   - 70% weight to CLIP similarity
   - 30% weight to shape similarity
   - Normalized scores for consistent results