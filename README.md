# 1. Quick Start 
run intall 
```bash 
python3 -m pip install -v -e .
```

first generate meta info json
```bash 
# for example subtype
python scripts/dataset_transforms/subtype.py data/subtype_251204_mideast-train.npy test_output.json --img-root ~/data/subtype/train_images_dir
``` 

generate database for querying
```bash 
sem_clean --generate --file test_output.json --db-path test_db
```

query database
```bash
sem_clean --clean --target target.json --output clean_result.json
```

# 2. Generate metas and vector database

The generate metas process follows a structured framework:

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

examples:
```proto
syntax = "proto2";
package demoproto;

enum ImageCategory {
    UNKNOWN = 0;      // 默认类别
    CAT = 1;          // 示例：猫
    DOG = 2;          // 示例：狗
}

// 图片数据结构
message ImageData {
    optional string id = 1;                      // 图片ID
    optional string path = 2;                    // 图片路径
    optional ImageCategory category = 3 [default = UNKNOWN];  // 默认类别
    repeated float features = 4;                 // 图片特征向量（786维）
}
```
