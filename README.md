# Vector Building Shape Intelligence Framework

## Project Overview

This project focuses on intelligent processing methods for vector building data, overcoming the limitations of traditional deep learning methods that primarily deal with raster data. It introduces a new Transformer-based framework for intelligent processing of vector building data, with the following main contributions:

1. **New Framework for Vector Building Data**: Abstracts vector building sequence data as spatial language and systematically introduces the Transformer model for direct modeling and processing of vector coordinate sequences. The framework covers building extraction, self-supervised feature representation, few-shot shape recognition, and end-to-end outline simplification.

2. **Polygon Regularization Network (PolyReg)**: Proposes a novel building outline extraction and regularization algorithm based on autoregressive sequence generation, using a Mask-Attention mechanism to output structured, regular coordinate sequences. Achieves significant improvements in IoU metrics on public datasets.

3. **Pre-trained Shape Representation Transformer (PSRT)**: Develops a model for building shape feature extraction and recognition based on self-supervised pre-training, with three novel tasks (CMP, COC, CSR). Enables one-shot building shape recognition with high accuracy and improved IoU metrics.

4. **Transformer-based Polygon Simplification Model (TPSM)**: Proposes an end-to-end method for building outline simplification using a sequence-to-sequence encoder-decoder architecture, outperforming existing algorithms in geometric fidelity and simplification quality.

## Quick Start

To quickly test the main functionalities, run:

```bash
python quick_start.py
```

This script demonstrates:
- Serialization and deserialization of building coordinates
- Polygon regularization (PolyReg)
- Polygon simplification (TPSM)
- Feature encoding (PSRT, ShapeClassifier, ShapeClassifierBCE)
- Shape type prediction

## Main Classes and Modules

### shape_model_hub.py
- **ShapeModelHub**: Unified interface for managing and invoking all building shape-related models. Provides methods to initialize and use PolyReg, TPSM, PSRT, ShapeClassifier, and ShapeClassifierBCE models, as well as utilities for preprocessing, postprocessing, and shape type prediction.

### shape_simply.py
- **PolyReg**: Polygon Regularization Network. Uses an autoregressive sequence generation approach with a Mask-Attention mechanism to regularize building outlines. The `generate` method takes a string of index tokens and outputs a regularized sequence.

### shape_regularization.py
- **TPSM**: Transformer-based Polygon Simplification Model. Uses a seq2seq Transformer to simplify building outlines. The `generate` method takes a vector and outputs a simplified sequence.
- **load_data**: Utility for loading and augmenting vector data samples for testing and demonstration.

### shape_feature_encoder.py
- **ShapeClassifier**: Encodes building shapes and computes similarity between them using a Transformer backbone.
- **ShapeClassifierBCE**: Similar to ShapeClassifier but uses a contrastive learning approach for batch encoding and similarity scoring.
- **PSRT_Model**: Pre-trained Shape Representation Transformer for extracting general shape features from vector data.

## Requirements

The requirements have been cleaned up to include only the packages actually used in the codebase. Install them with:

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- torch
- numpy
- matplotlib
- bert4torch
- transformers
- Pillow
- scikit-learn
- scipy
- opencv-python

## Weights and Data

Model weights and vocabularies are expected in the `weights/` directory. Example data is in `datasets/`.

---
For more details, see the code and comments in each module.

## Notes on bert4torch

- This project uses `bert4torch==0.1.5`. The official repository is: https://github.com/Tongjilibo/bert4torch
- For model training, please refer to the documentation and examples in the bert4torch repository. **Note:** Newer versions of bert4torch may not be compatible with this codebase. Please use version 0.1.5 as specified in requirements.txt.

## Citation

If you find this project helpful, please consider citing one or more of the following papers:

1. Cui L, Li C, Chen X, Wang X, Qian H. PolyReg: Autoregressive Building Outline Regularization via Masked Attention Sequence Generation[J]. Remote Sensing, 2025, 17(9): 1650.
2. Cui L, Qian H, Xu J, Li C, Niu X. Contrastive learning for one-shot building shape recognition using vector polygon transformers[J]. Geocarto International, 2025, 40(1): 2471087.
3. Cui L, Xu J, Jiang L, Qian H. End-to-End Vector Simplification for Building Contours via a Sequence Generation Model[J]. ISPRS International Journal of Geo-Information. 2025; 14(3):124.
4. Cui L, Niu X, Qian H, Wang X, Xu J. A Transformer-Based Approach for Efficient Geometric Feature Extraction from Vector Shape Data[J]. Applied Sciences. 2025; 15(5):2383.
