# NoTouch
A contactless fingerprint recognition react app

Working:
1. Take finger photo and segment using `extract_skin` function in `fingerprint_segmentation.ipynb`
2. Then execute `binary_img.py`
3. Then execute `finegerprint_pipline.py`
4. Use backend saved model to find cosine similarity.
---
Improvement required in encoder to make unique encodings of fingerprint
