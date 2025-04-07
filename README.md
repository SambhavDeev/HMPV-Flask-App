# HMPV-Flask-App
Pneumonia Detection using ResNet50 on Chest X-Ray Images
📌 Project Overview
This project aims to classify chest X-ray images into two categories: Pneumonia and Normal, leveraging the power of deep convolutional neural networks. After initial trials with VGG19, we adopted a more robust architecture, ResNet50, which provided significant improvements in model performance and generalization.

🧠 Model: ResNet50 (Transfer Learning)
We used ResNet50, a 50-layer deep CNN architecture pretrained on ImageNet. Its residual connections and deep structure make it highly effective for transfer learning in medical imaging tasks.

Input image size: 224 x 224

Custom classification head added on top of the ResNet50 base:

Flatten → Dense(512, relu) → Dropout(0.5)

Dense(128, relu) → Dropout(0.5)

Dense(2, softmax) for binary classification

🔁 Data Augmentation
To reduce overfitting and increase robustness, we applied aggressive data augmentation during training:

Rotation: ±30°

Width/Height Shift: 20%

Zoom: [0.8, 1.2]

Shear, Brightness, Flip

Preprocessing: preprocess_input from ResNet50

📊 Performance Comparison: ResNet50 vs VGG19
Metric	       VGG19	 ResNet50 (This Model)
Train Accuracy	~88%	 ~93.3%
Val Accuracy	~87%	   93.75%
Best Val Loss	~0.33	    0.212
ROC-AUC Score	~0.90	    0.97
Test accuracy           0.89
Weighted F1 Score       0.89
Params	~20M	75M (but frozen base)
✅ ResNet50 significantly outperformed VGG19 in validation accuracy and generalization.

🧪 Evaluation and Callbacks
Used EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau

Saved the best model: resnet50_best_model.h5

Evaluation metrics included: accuracy, loss, ROC curve

📂 Dataset
The dataset used is the Chest X-ray Pneumonia dataset from Kaggle, organized in the following structure:

bash
Copy
Edit
chest_xray/
├── train/
├── val/
└── test/
📈 ROC Curve & Insights
The ROC curve plotted for the ResNet50 model indicates a high true positive rate with low false positive rate, validating the model's excellent discriminative power.

💡 Conclusion
This project demonstrated the superiority of ResNet50 over VGG19 for pneumonia detection from chest X-rays. The deeper architecture and residual connections allow for better feature learning, leading to improved accuracy, stability, and generalization on unseen data.

🚀 Future Work
Fine-tuning deeper layers of ResNet50

Applying Grad-CAM for visual explainability

Exploring EfficientNet or Vision Transformers (ViTs)

Feel free to clone, modify, and contribute to this project.
For queries or contributions, contact: Sambhav Dev
