# Semantic Clean - Intelligent Data Cleaning Tool

Label consistency validation based on DINOv3 embeddings, providing intelligent cleaning functionality for image datasets.

## 1. Quick Start

### Installation
```bash
python3 -m pip install -v -e .
```

### Basic Workflow

1. **Generate Metadata JSON**
```bash 
# Example for subtype dataset
python scripts/dataset_transforms/subtype.py data/subtype_251204_mideast-train.npy test_output.json --img-root ~/data/subtype/train_images_dir
``` 

2. **Generate Vector Database**
```bash 
sem_clean --generate --file test_output.json --db-path test_db
```

3. **Data Cleaning**
```bash
sem_clean --clean --base ./test_db --target target.json --output clean_result.json
```

## 2. Command Line Arguments

### Database Generation
```bash
sem_clean --generate --file <input_json_file> --db-path <database_path>
```

### Data Cleaning
```bash
sem_clean --clean --base <base_database_path> --target <target_json_file> --output <output_result_file>
```

### Complete Parameter List
- `--generate`: Database generation mode
- `--clean`: Data cleaning mode
- `--file`: Input JSON file path (generation mode)
- `--base`: Base database path (cleaning mode)
- `--target`: Target JSON file path (cleaning mode)
- `--output`: Output result file path (cleaning mode)
- `--db-path`: ChromaDB database path (default: chroma_db)
- `--help`: Display help information

# 2. Generate Metadata and Vector Database

The metadata generation process follows a structured framework:

## Metadata Processing Framework

1. **Data Ingestion**
   - Load structured annotation data from multiple input formats
   - Handle both individual files and batch processing

2. **Semantic Annotation**
   - Convert raw annotations to semantic category labels
   - Support multi-label classification for rich object descriptions

3. **Protobuf Serialization**
   - Use protocol buffers for efficient data serialization
   - Store comprehensive image metadata including:
   - Image identifiers and file paths
   - Multiple semantic categories per image
   - Feature vector storage for similarity search

4. **Vector Database Creation**
   - Extract deep visual features using DINOv3 model
   - Store embeddings alongside metadata in ChromaDB
   - Enable semantic similarity search and category-based filtering

This metadata framework forms the foundation for data cleaning operations.

## Example Protocol Buffer Definition
```proto
syntax = "proto2";
package demoproto;

enum ImageCategory {
    UNKNOWN = 0;      // Default category
    CAT = 1;          // Example: cat
    DOG = 2;          // Example: dog
}

// Image data structure
message ImageData {
    optional string id = 1;                      // Image ID
    optional string path = 2;                    // Image path
    optional ImageCategory category = 3 [default = UNKNOWN];  // Default category
    repeated float features = 4;                 // Image feature vector (786 dimensions)
}
```

# 3. Data Cleaning Algorithm

## Core Principles

Label consistency validation based on DINOv3 embeddings, evaluating image label credibility through three key metrics:

### 3.1 Algorithm Metrics

1. **kNN Label Consistency (p)**
   - Query k nearest neighbors of the image in the database
   - Calculate the proportion of same-label neighbors
   - Evaluate local label consistency

2. **Class Mean Distance (d_μ)**
   - Calculate distance from query image to class center
   - Normalize using intra-class average distance
   - Evaluate global distribution consistency

3. **Nearest Same-Class Distance (d_min)**
   - Query distance to nearest same-label sample
   - Evaluate local density after normalization

### 3.2 Confidence Score

Comprehensive scoring formula:
```
S = w1 × p - w2 × d_min^norm - w3 × d_μ^norm
```

Default weights:
- w1 = 1.0 (kNN consistency weight)
- w2 = 0.5 (nearest distance weight)  
- w3 = 0.5 (class mean distance weight)

### 3.3 Decision Strategy

| Confidence Range | Decision | Description |
|-------------------|----------|-------------|
| S ≥ 0.4          | accept   | High confidence, auto-accept |
| S ≤ -0.4         | reject   | Low confidence, auto-reject |
| -0.4 < S < 0.4   | review   | Medium confidence, manual review |

## 3.4 Output Format

The cleaning result JSON file contains detailed information for each image:

```json
[
  {
    "image_id": "image_001",
    "image_path": "/path/to/image.jpg",
    "status": "accept",
    "score": 0.75,
    "category": "CAR",
    "metrics": {
      "knn_consistency": 0.95,
      "nearest_distance_normalized": 0.12,
      "class_distance_normalized": 0.34
    },
    "error": null
  }
]
```

### Field Descriptions
- `image_id`: Unique image identifier
- `image_path`: Image file path
- `status`: Cleaning decision (accept/reject/review)
- `score`: Confidence score (range approximately -1 to 1)
- `category`: Primary validated category
- `metrics`: Detailed algorithm metrics
- `error`: Error information (if any processing exception)

## 3.5 Statistics Information

Detailed statistics displayed after cleaning completion:

```
=== Cleaning Results Statistics ===
Total: 1000
Accept: 650 (65.00%)
Reject: 200 (20.00%)
Review: 150 (15.00%)
Processing Errors: 0
```

## 3.6 Technical Features

- **Batch Processing Optimization**: Support memory-optimized processing for large datasets
- **Error Recovery**: Comprehensive exception handling and fault tolerance
- **Progress Display**: Real-time processing progress and statistics
- **Component Reuse**: Efficient reuse of existing feature extraction and database components
- **Extensibility**: Modular design facilitates algorithm extension and parameter tuning
