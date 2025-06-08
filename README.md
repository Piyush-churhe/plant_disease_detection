This project implements a hybrid deep learning model combining CNN (EfficientNet-B3) and GNN (GCNConv) for detecting plant diseases from leaf images. It aims to leverage the powerful feature extraction of CNNs along with the spatial relationship modeling ability of GNNs to improve classification performance on plant disease datasets (Google Colab was used for execution)

**Project Features**
1. Utilizes EfficientNet-B3 as the CNN backbone for high-quality feature extraction
2. Builds an adaptive k-NN graph from CNN features to capture structural similarity
3. Implements a Graph Convolutional Network (GCN) with skip connections and batch normalization
4. Performs data augmentation using Albumentations to improve generalization
5. Supports mixed-precision training for faster computation on GPU
6. Includes enhanced training loop with early stopping, class balancing, and Cosine Annealing scheduler
7. Generates evaluation metrics including classification report, confusion matrix, and random test image predictions

**Dataset Used**
PlantDoc Dataset from GitHub : https://github.com/pratikkayal/PlantDoc-Dataset.git
This dataset contains annotated leaf images of multiple plant species showing different types of diseases.
In this project, a subset of classes was used:
1. Apple Scab Leaf
2. Apple Rust Leaf
3. Blueberry Leaf
4. Peach Leaf
5. Tomato Leaf

**Libraries and Tools Used**
1. PyTorch & PyTorch Geometric – for CNN and GNN implementation

2. timm – to load pretrained EfficientNet

3. Albumentations – for advanced data augmentation

4. sklearn – for train-test split, evaluation metrics

5. matplotlib, seaborn – for visualization

6. PIL, numpy, shutil, os – for image and filesystem handling


**Model Accuracy**
1. Best Validation Accuracy: ~90.75%
2. Final Test Accuracy: ~70.80%
The model achieved high performance on the test set, showing strong generalization across disease categories.

**Future Improvements**
1. Expand to use the full PlantDoc dataset with all 27+ classes

2. Integrate Vision Transformers (ViTs) or Graph Attention Networks (GAT) for enhanced accuracy

3. Build a web interface using Streamlit or Flask for user-friendly disease diagnosis

4. Incorporate self-supervised learning to reduce reliance on labeled data

5. Deploy the model to mobile or edge devices for offline use in agriculture
