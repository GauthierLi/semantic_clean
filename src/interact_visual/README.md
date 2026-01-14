# review
从`sem_clean --clean --base ./test_db --target test_output.json --output edge_cases_result.json` 中获取到状态为"review"的样本
```
  {
    "image_id": "JME2717_23_1763017890_1763017900_4290323_0",
    "image_path": "/home/gauthierli/data/subtype/train_images_dir/JME2717_23_1763017890_1763017900_4290323_0.jpg",
    "decision": "review",
    "score": 0.33538519402385425,
    "category": "CAR",
    "metrics": {
      "knn_consistency": 1.0,
      "nearest_distance_normalized": 0.021955199539661407,
      "class_distance_normalized": 1.30727441241263
    },
    "error": null
  },
```
从中拿到图片的路径以及类别