# Semantic Clean - Intelligent Data Cleaning Tool

Label consistency validation based on DINOv3 embeddings, providing intelligent cleaning functionality for image datasets.

## 1. Quick Start

### Installation
```bash
conda env create -f environment.yml
conda activate torch
```

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
$$
S = w_1 × p - w_2 × d_{min}^{norm} - w_3 × d_μ^{norm}
$$

Default weights:
- $w_1$ = 1.0 (kNN consistency weight)
- $w_2$ = 0.5 (nearest distance weight)  
- $w_3$ = 0.5 (class mean distance weight)

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

# 4. Interactive Visual Review System

The Interactive Visual Review System provides a web-based interface for manual review and annotation of edge cases identified during the data cleaning process.

## 4.1 Quick Start

### Installation

```bash
# Navigate to the interactive visual module
cd src/interact_visual

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Or use the one-click startup script:

```bash
./start_server.sh
```

The server will start on `http://localhost:8023` by default.

### Basic Workflow

1. **Load the cleaning result file**
   - Enter the path to your `edge_cases_result.json` file
   - Click "📂 Load File"

2. **Select a category**
   - Choose a category from the dropdown menu
   - Filter by decision status (review/accept/reject)

3. **Review images**
   - Left-click to zoom in on an image
   - Right-click to select/deselect images
   - Ctrl + mouse hover to batch select

4. **Choose selection mode**
   - **Positive mode**: Selected images → accept, Unselected → reject
   - **Negative mode**: Selected images → reject, Unselected → accept

5. **Save changes**
   - Click "💾 Save Changes" to update decisions
   - Add optional comment tags

6. **Download results**
   - Click "📥 Download Results" to get the modified JSON file

## 4.2 Core Features

### Data Management
- **File Loading**: Load `edge_cases_result.json` and create a working copy
- **Category Filtering**: Filter samples by category and decision status
- **Pagination**: Display 100/500/1000 images per page
- **State Synchronization**: Real-time sync between frontend and backend

### User Interaction
- **Image Selection**: Right-click to select, Ctrl+hover for batch selection
- **Image Preview**: Left-click to zoom in with zoom controls
- **Keyboard Shortcuts**: Full keyboard support for efficient navigation
- **Selection Modes**: Positive and negative selection modes

### Performance Optimization
- **Lazy Loading**: Images load only when entering the viewport
- **Memory Management**: Automatic cleanup of off-screen images
- **Preloading**: Preload next page images for smooth navigation
- **Performance Monitoring**: Real-time memory usage display

## 4.3 API Endpoints

### Core Endpoints

#### Load Review Data
```http
POST /api/load_review_data
Content-Type: application/json

{
    "file_path": "/path/to/edge_cases_result.json"
}
```

#### Filter by Category
```http
POST /api/filter_by_category
Content-Type: application/json

{
    "category": "SV_POLICE_CAR",
    "decision": "review",
    "page": 1,
    "per_page": 100
}
```

#### Save Changes
```http
POST /api/save_changes
Content-Type: application/json

{
    "selection_mode": "positive",
    "current_category": "SV_POLICE_CAR",
    "current_decision": "review",
    "selected_images": ["image_id_1", "image_id_2"],
    "comments": ["tag1", "tag2"]
}
```

#### Download Result
```http
GET /api/download_result/<filename>
```

## 4.4 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + Mouse Hover | Batch select images |
| Left Click | Zoom image |
| Right Click | Select/Deselect |
| Space | Select in modal |
| ← → | Navigate images |
| ESC | Close modal |
| F | Show keyboard shortcuts |

## 4.5 Project Structure

```
src/interact_visual/
├── app.py                 # Flask backend application
├── start_server.sh        # One-click startup script
├── requirements.txt       # Python dependencies
├── README.md             # Module documentation
├── templates/
│   └── index.html        # Frontend main page
└── static/
    ├── css/
    │   └── style.css     # Stylesheet
    └── js/
        ├── main.js        # Core JavaScript logic
        └── performance.js # Performance optimization module
```

## 4.6 Technical Details

### Decision Logic

The system supports two-level decision management:

1. **Category-level decisions**: Each category within an image has its own decision status
2. **Overall decision**: The overall image decision is derived from category decisions
   - If any category is `reject` → overall `reject`
   - If any category is `review` (and no `reject`) → overall `review`
   - Otherwise → overall `accept`

### Empty Selection Logic

When no images are selected:
- **Positive mode**: All unselected images are set to `reject`
- **Negative mode**: All unselected images are set to `accept`

This allows for efficient bulk operations.

### Performance Features

- **Lazy Loading**: Uses IntersectionObserver API
- **Memory Optimization**: Freezes off-screen images
- **Preloading**: Fetches next page in background
- **Responsive Design**: Adapts to different screen sizes

## 4.7 Configuration

### Environment Variables

```bash
# Server port (default: 8023)
export SERVER_PORT=8023

# Flask debug mode
export FLASK_DEBUG=1
```

### Application Settings (in app.py)

```python
# Pagination
PER_PAGE = 20  # Images per page

# File upload limit
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Logging level
LOG_LEVEL = logging.INFO
```

## 4.8 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8023
lsof -i :8023

# Kill the process
kill -9 <PID>
```

### Images Not Loading
- Check image paths in the JSON file
- Verify image files exist and have correct permissions
- Ensure the server has access to the image directory

### Virtual Environment Issues
```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4.9 Development

### Running in Development Mode

```bash
# Activate virtual environment
source venv/bin/activate

# Run with debug mode
python app.py
```

### Adding New Features

1. **Backend**: Add new routes in `app.py`
2. **Frontend**: Extend `ImageFilterApp` class in `main.js`
3. **Styles**: Modify `style.css`
4. **Performance**: Add optimizations in `performance.js`
