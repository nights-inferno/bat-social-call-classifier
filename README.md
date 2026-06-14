# Bat Social Call Classifier

A machine learning pipeline for classifying pipistrelle bat echolocation and social calls using deep learning encoders (Perch2, EfficientNet B0, and NLM-BEATs). This repository contains tools for audio preprocessing, feature extraction, model training, and comprehensive evaluation metrics.

## Project Structure

###  **data/** - Data Management and Visualization
Tools for generating, organizing, and visualizing the bat audio dataset.

- **`csv_generator.py`** - Consolidates bat audio dataset from organized folders into CSV metadata files
  - Parses audio files from multi-label class folders (Type A, B, C, D, Echolocation)
  - Handles multi-label samples (e.g., samples that are both Type A and Type C)
  - Generates metadata CSV with file paths and one-hot encoded labels
  - Creates relative paths for DataLoader integration

- **`csv_ood_generator.py`** - Generates out-of-distribution (OOD) dataset metadata
  - Extracts and organizes OOD audio samples from external sources
  - Creates corresponding CSV files for OOD evaluation

- **`data_visualisation.ipynb`** - Jupyter notebook for exploratory data analysis
  - Visualizes dataset distributions and class imbalances
  - Plots audio spectrograms and waveforms
  - Provides statistical summaries of the dataset

- **`bat_metadata.csv`** - Consolidated metadata for in-distribution training data
- **`bat_metadata_len.csv`** - Training data with additional duration information
- **`ood_metadata.csv`** - Out-of-distribution samples for robustness evaluation
- **`xenocanto-dataset/`** - External audio dataset from Xeno-canto

---

###  **preprocessing/** - Audio Processing Pipeline
Core audio preprocessing and dataset handling for model training.

- **`dataset.py`** - Complete audio preprocessing and dataset management
  - **`BatAudioPipeline`** - Main preprocessing class handling:
    - Audio loading with soundfile/torchaudio
    - Butterworth bandpass filtering (high-pass at 15 kHz, optional bandreject for echoes)
    - Time expansion (10x or 5x depending on encoder) for ultrasonic call analysis
    - Sliding window segmentation with configurable overlap
    - Time-shift augmentation
  
  - **`AugmentationPipeline`** - Data augmentation strategies:
    - Iterative oversampling for multi-label class imbalance handling
    - Online augmentations: time shifting, noise addition, pitch shifting
    - Resample-time augmentations for duplicated minority samples
    - Imbalance Ratio per Label (IRLBL) calculation
  
  - **`PipistrelleDataset`** - Custom PyTorch Dataset class:
    - Loads and preprocesses pipistrelle bat recordings
    - Supports encoder-specific pipeline configurations (Perch2, EfficientNet B0, NLM-BEATs)
    - Applies multi-level data augmentation strategies
    - Integrates class imbalance handling and resampling

---

### **models/** - Model Architectures, Feature Extraction, and Training
Machine learning models and feature extraction utilities for bat call classification.

- **`feature_generation.py`** - Feature extraction from audio encoders
  - **`extract_feature()`** - Extracts embeddings from individual audio windows
    - Supports TensorFlow (Perch2) and PyTorch (EfficientNet B0, NLM-BEATs) models
    - Cross-framework compatibility between TensorFlow and PyTorch
  
  - **`build_feature_bank()`** - Batch feature extraction for entire dataset
    - Processes all recordings and extracts their embeddings
    - Returns structured feature bank and labels for downstream classifiers
  
  - **`extract_encoder()`** - Loads and initializes encoder models
    - Supports Perch2 (TensorFlow Hub), EfficientNet B0, and NLM-BEATs
    - Handles CPU/GPU device management
  
  - **`pool_features()`** - Aggregates features across spatial and temporal dimensions
    - Mean/max pooling across window dimensions
    - Encoder-specific pooling strategies (different axes for different models)

- **`classifier_models.py`** - Classification models for multi-label prediction
  - **`BalancedMLP`** - Multi-layer perceptron with class imbalance handling
    - Configurable hidden dimensions and dropout rates
    - Integrated focal loss support for long-tailed distributions
    - Automatic positive weight balancing for BCEWithLogitsLoss
    - Optional batch normalization
    - Validation set support during training
  
  - **`MultilabelPrevalenceBaseline`** - Baseline classifier for comparison
    - Supports stochastic, majority, and prevalence-based prediction strategies
    - Useful for evaluating classifier performance against baselines

- **`focal_loss.py`** - Focal Loss implementation for imbalanced classification
  - Reduces loss for well-classified samples, focusing on hard negatives
  - Configurable gamma and alpha hyperparameters

- **`linear_probes.py`** - Linear probing utilities
  - Trains shallow linear models on top of frozen encoder features
  - Useful for understanding encoder feature quality

- **`MLP_balancing.py`** - Enhanced MLP with advanced balancing strategies
  - Class weighted loss and Focal loss options

- **`abmil_model.py`** - Attention-Based Multiple Instance Learning (ABMIL)
  - Aggregates bag-level predictions from instance-level embeddings
  - Attention weights highlight important audio segments for classification
  - Suitable for weakly supervised multi-label learning
  - Ultra-minority sample delegation to Logistic Regressor

---

### 📊 **evaluation/** - Model Evaluation and Results Analysis
Comprehensive evaluation metrics, statistical testing, and visualization tools.

- **`metrics.py`** - Extensive classification metrics and performance evaluation
  - Multi-label classification metrics (precision, recall, F1-score per label)
  - Aggregation strategies (micro, macro, weighted averaging)
  - Confusion matrix computation for multi-label tasks
  - ROC-AUC curves and threshold optimization
  - Additional custom metrics for specialized use cases

- **`statistical_tests.py`** - Statistical hypothesis testing
  - Compares classifier performance across models and conditions
  - Namanyi and Friedman tests
  - Effect size calculations
  - Multiple comparison corrections 

- **`attention_visual.py`** - Attention mechanism visualization
  - Visualizes attention weights from ABMIL models
  - Highlights important time regions in audio spectrograms
  - Helps interpret which audio segments drive predictions

- **`tables.py`** - Results table generation and formatting
  - Compiles metrics across multiple models and conditions
  - Formats results for reporting
  - Supports LaTeX table output 

---

## Usage Example

```python
# 1. Prepare dataset
from preprocessing.dataset import PipistrelleDataset

dataset = PipistrelleDataset(
    data_input="data/bat_metadata.csv",
    root_dir="path/to/audio/files",
    is_training=True,
    resample=True,
    encoder="perch2"
)

# 2. Extract features
from models.feature_generation import extract_encoder, build_feature_bank

encoder = extract_encoder("perch2", device='cuda')
features, labels = build_feature_bank(dataset, encoder, "perch2")

# 3. Train classifier
from models.classifier_models import BalancedMLP

clf = BalancedMLP(
    input_dim=1536,
    hidden_dim=128,
    focal_loss=True,
    balanced=True
)
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# 4. Evaluate
predictions = clf.predict(X_test)