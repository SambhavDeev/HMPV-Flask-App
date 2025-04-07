<div align="center">
  <img src="https://raw.githubusercontent.com/SambhavDev/HMPV-Flask-App/main/static/pneumonia_banner.png" alt= width="800">
  <h1><samp>⚕️</samp> HMPV-Flask-App: Pneumonia Detection from Chest X-Rays <samp>🩻</samp></h1>
  <p>
    A deep learning web application for classifying chest X-ray images as Pneumonia or Normal using a fine-tuned ResNet50 model.
  </p>
</div>

<br>

## <samp>📌</samp> Project Overview

This project leverages the power of deep convolutional neural networks to accurately classify chest X-ray images into two critical categories: **Pneumonia** and **Normal**. Building upon initial experimentation with VGG19, we transitioned to the more robust **ResNet50** architecture, which demonstrated significant improvements in model performance and its ability to generalize to unseen data.

## <samp>🧠</samp> Model: ResNet50 (Transfer Learning)

We harnessed the power of **ResNet50**, a 50-layer deep Convolutional Neural Network (CNN) architecture pre-trained on the extensive ImageNet dataset. Its innovative residual connections and deep structure make it exceptionally well-suited for transfer learning tasks, particularly in the domain of medical imaging.

**Model Architecture Details:**

* **Input Image Size:** `224 x 224` pixels
* **Base Model:** Pre-trained ResNet50 (feature extraction layers were **frozen** during initial training)
* **Custom Classification Head:**
    ```
    Flatten → Dense(512, activation='relu') → Dropout(0.5)
            → Dense(128, activation='relu') → Dropout(0.5)
            → Dense(2, activation='softmax')  # Binary classification (Pneumonia/Normal)
    ```

## <samp>🔁</samp> Data Augmentation

To mitigate the risk of overfitting and enhance the model's ability to generalize to diverse unseen data, we employed a comprehensive suite of aggressive data augmentation techniques during the training process:

* **Rotations:** Randomly rotated images by ±30 degrees.
* **Shifts:** Randomly shifted image width and height by up to 20%.
* **Zooming:** Randomly zoomed in/out on images within the range of [0.8, 1.2].
* **Shearing:** Applied random shear transformations.
* **Brightness Adjustments:** Randomly altered image brightness.
* **Flipping:** Randomly flipped images horizontally.
* **Preprocessing:** Utilized the `preprocess_input` function specifically designed for ResNet50 to normalize input pixel values.

## <samp>📊</samp> Performance Comparison: ResNet50 vs VGG19

| Metric            | VGG19   | ResNet50 (This Model) | Improvement        |
| ----------------- | ------- | --------------------- | ------------------ |
| **Train Accuracy** | ~88%    | **~93.3%** | <font color="green">↑ 5.3%</font> |
| **Validation Accuracy** | ~56%    | **~93.75%** | <font color="green">↑ 37.75%</font>|
| **Best Val Loss** | ~0.66   | **~0.212** | <font color="green">↓ 0.448</font>|
| **ROC-AUC Score** | -       | **0.97** | <font color="green">Excellent</font> |
| **Test Accuracy** | -       | **0.89** | -                  |
| **Weighted F1 Score**| -       | **0.89** | -                  |
| **Parameters** | ~20M    | 75M (frozen base)     | -                  |

<br>

✅ **Key Takeaway:** ResNet50 demonstrated a significantly superior performance compared to VGG19, particularly in terms of validation accuracy and overall generalization capability.

## <samp>🧪</samp> Evaluation and Callbacks

We implemented several key strategies during training to ensure robust model development and prevent overfitting:

* **EarlyStopping:** Monitored the validation loss and stopped training when it ceased to improve, preventing overfitting.
* **ModelCheckpoint:** Saved the model weights that achieved the best performance on the validation set (`resnet50_best_model.h5`).
* **ReduceLROnPlateau:** Dynamically reduced the learning rate when the validation loss plateaued, helping the model to fine-tune and potentially escape local minima.

**Evaluation Metrics:**

* Accuracy
* Loss
* Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)

## <samp>📂</samp> Dataset

The dataset utilized for this project is the widely recognized **Chest X-ray Pneumonia dataset** sourced from Kaggle. The dataset is meticulously organized into the following directory structure:


chest_xray/
├── train/    # Training images (Pneumonia and Normal subdirectories)
├── val/      # Validation images (Pneumonia and Normal subdirectories)
└── test/     # Testing images (Pneumonia and Normal subdirectories)

## <samp>📈</samp> ROC Curve & Insights

The Receiver Operating Characteristic (ROC) curve generated for the ResNet50 model visually confirms its strong discriminatory power. The curve demonstrates a high **True Positive Rate (TPR)** across various thresholds while maintaining a low **False Positive Rate (FPR)**, resulting in an impressive **AUC score of 0.97**. This indicates the model's excellent ability to distinguish between Pneumonia and Normal chest X-ray images.

## <samp>💡</samp> Conclusion

This project unequivocally highlights the superiority of the **ResNet50** architecture over VGG19 for the task of pneumonia detection using chest X-ray images. The deeper network structure and the incorporation of residual connections in ResNet50 enable more effective feature learning, leading to substantial improvements in accuracy, training stability, and the crucial ability to generalize well to unseen medical data.

## <samp>🚀</samp> Future Work

Moving forward, we plan to explore several avenues to further enhance the performance and applicability of this project:

* **Fine-tuning Deeper Layers:** Experiment with unfreezing and fine-tuning more layers of the pre-trained ResNet50 model to potentially extract more task-specific features.
* **Visual Explainability with Grad-CAM:** Implement Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual explanations highlighting the regions in the chest X-ray images that the model focuses on when making its predictions. This can improve trust and interpretability.
* **Exploring Advanced Architectures:** Investigate the potential of more recent and efficient deep learning architectures such as **EfficientNet** or **Vision Transformers (ViTs)** for this task.

<br>

Feel free to **clone**, **modify**, and **contribute** to this project! Your feedback and contributions are highly valued.

For any queries or potential collaborations, please don't hesitate to contact:

**Sambhav Dev**
