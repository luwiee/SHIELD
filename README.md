# Rice Leaf Disease Prediction

This application predicts rice leaf diseases using various machine learning models including pre-trained Keras models, models trained from scratch, and SVM models using LBP and HOG features. The application is built using Streamlit for the web interface.

## Models Used

1. **Keras Model Pre-trained**: A pre-trained Keras model for rice leaf disease prediction.
2. **Keras Model From Scratch**: A Keras model trained from scratch for rice leaf disease prediction.
3. **SVM Model (LBP)**: An SVM model trained on Local Binary Patterns (LBP) features.
4. **SVM Model (HOG)**: An SVM model trained on Histogram of Oriented Gradients (HOG) features.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- Streamlit
- TensorFlow
- Keras
- OpenCV
- scikit-image
- joblib
- Pillow

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/rice-leaf-disease-prediction.git
    cd rice-leaf-disease-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Place your trained model files (`svm_rice_leaf_model_lbp.joblib`, `rice_leaf_model_pretrained.h5`, `rice_leaf_model_scratch.h5`, `rice_leaf_model_svm_hog.joblib`) in the project directory.

### Running the Application

Run the Streamlit app:
```sh
streamlit run app.py
