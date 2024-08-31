# RNN Model Experiments

This repository contains experiments with Recurrent Neural Networks (RNNs) for sequence tasks, including sentiment analysis. The experiments include:

1. **Implementing a Basic RNN Model**
2. **Stacking RNN Layers and Bi-Directional RNNs**
3. **Exploring Hybrid Architectures**

## 1. Implementing a Basic RNN Model

This script implements a basic RNN model for sentiment analysis using the IMDB dataset.

### **Script: `Implementing a Basic RNN Model.py`**

- **Description:** Trains a basic RNN model on the IMDB dataset.
- **Data:** Assumes the dataset is a CSV file with columns `review` and `sentiment`.
- **Tasks:**
  - Load and preprocess the dataset.
  - Tokenize and pad text sequences.
  - Encode sentiment labels.
  - Build and train a basic RNN model.
  - Evaluate the model's performance.

### **How to Run:**

1. Ensure you have the required libraries installed: `numpy`, `pandas`, `tensorflow`, `scikit-learn`.
2. Update the dataset path in the script.
3. Run the script:
   ```bash
   python basic_rnn.py 
    ```

## Stacking RNN Layers and Bi-Directional RNNs
This script modifies the basic RNN model by adding stacked RNN layers and converting it into a bi-directional RNN.
### Script: `Stacking RNN Layers and Bi-Directional RNNs.py`**

- **Description:** Implements and trains stacked RNN layers and a bi-directional RNN model.
- **Tasks:**
  - Load and preprocess the dataset.
  - Tokenize and pad text sequences.
  - Encode sentiment labels.
  - Build and train models with stacked RNN layers and bi-directional RNN layers.
  - Compare performance with the basic RNN model.
### **How to Run:**

1. Ensure you have the required libraries installed: numpy, pandas, tensorflow, scikit-learn.
2. Update the dataset path in the script.
3. Run the script:
    ```bash
   Stacking RNN Layers and Bi-Directional RNNs.py
    ```
## Exploring Hybrid Architectures
This script explores hybrid architectures by combining RNNs with other models like CNNs or Attention mechanisms.

### Script: hybrid_rnn_cnn.py
- **Description:** Implements a hybrid RNN-CNN model for sentiment analysis.
- **Tasks:**
  - Load and preprocess the dataset.
  - Tokenize and pad text sequences.
  - Encode sentiment labels.
  - Build and train a hybrid model combining RNN and CNN layers.
  - Compare performance with previous models.

### **How to Run:**
1. Ensure you have the required libraries installed: numpy, pandas, tensorflow, scikit-learn.
2. Update the dataset path in the script.
3. Run the script:
```bash
Exploring Hybrid Architectures.py
```
## Results and Analysis
- Each script will output the model’s training and validation performance.
- The performance metrics include accuracy and loss.
- A detailed report discussing the results, challenges faced, and the benefits of hybrid approaches will be included in the project.

## Dependencies
- Python 3.7 or later
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn

