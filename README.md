# Medical Image Retrieval System

## Overview

This project implements an advanced image retrieval system designed for medical imaging applications. It provides efficient and accurate image search capabilities, particularly focusing on Whole Slide Images (WSI) in medical diagnostics. The system employs deep learning techniques for image embedding generation and fast similarity search.

## Features

- **Image-to-Image Retrieval**: Search for similar medical images based on a query image
- **High-performance Vector Search**: Utilizes optimized vector databases for fast similarity search
- **Batch Processing**: Support for processing multiple query images simultaneously
- **Region Detection**: Identifies connected regions within search results
- **REST API**: Comprehensive API for seamless integration with other applications
- **Multi-backend Support**: Compatible with multiple vector databases (FAISS, Milvus, Qdrant)



## Project Structure

- `Retrieval_Server/`: Main server implementation
  - `app.py`: Core REST API endpoints
  - `image_app.py`: Image retrieval service
  - `modules/`: Core modules for retrieval functionality
- `MDI_RAG_Image2Image_Research/`: Research and development codebase
- `Cuda_code/`: Custom CUDA implementations for performance optimization
- `test_*.py/ipynb`: Some test scripts and notebooks for different components

## Setup and Installation

1. Clone the repository:


2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your image database (vectors need to be pre-computed)

4. Start the server:
   ```
   python Retrieval_Server/app.py
   ```

## API Usage

### Image Search

```
POST /search
```

Request body:
```json
{
  "query_img_path": "path/to/image.jpg",
  "top_k": 20,
  "include_metadata": true
}
```

### Batch Image Search

```
POST /batch_search
```

Request body:
```json
{
  "query_img_paths": ["path/to/image1.jpg", "path/to/image2.jpg"],
  "top_k": 20,
  "combine_results": true
}
```

## Performance Considerations

- The system is optimized for GPU acceleration with CUDA support
- Large-scale image databases require substantial memory resources
- Vector database optimization is crucial for production deployments

## Requirements

See requirements.txt

Due to the file size limitation, more dependencies may need to be downloaded manually, such as UNI Encoder and datasets, etc. We will provide the netbook download link after we organize them later. If there are any questions about the code,  please submit Issue or contact anonymous (under review).