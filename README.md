# Rice-Leaf-Disease


## 🎯 Project Goal & Domain Analysis

* Objective: Build a Convolutional Neural Network (CNN) model to detect diseases in rice leaves using image 	data.

* Domain Expertise: Focused on classifying leaf diseases using image classification techniques.

## 📚 Data Preparation

* Dataset: Rice leaf images categorized into 3 classes.

* Data Splitting:

  * Split into Training, Validation, and Testing sets using splitfolders.
    
  * Ratio: 80% (Train), 10% (Validation), 10% (Test)

## 🖼️ Data Augmentation & Preprocessing
	
  * Data Augmentation Techniques:
		
    * Rotation Range: 40°
		
    * Width & Height Shift: 0.2
		
    * Shear Range: 0.2
		
    * Zoom Range: 0.2
		
    * Rescaling to [0,1] using ImageDataGenerator
	
  * Batch Size: 16 images per batch with RGB channels.

## 🧠 CNN Model Creation
	
  * Model Architecture:
		
    * Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation
		
    * MaxPooling Layer 1: 2x2 pooling
		
    * Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation
		
    * MaxPooling Layer 2: 2x2 pooling
		
    * Flatten Layer
		
    * Dense Layer: 128 neurons, ReLU activation
		
    * Output Layer: 3 neurons (Softmax for 3-class classification)
	
  * Model Summary:
		
    * Total Parameters: ~1.5 million
		
    * Compiled with Adam Optimizer and Categorical Cross-Entropy Loss

## 🏋️ Model Training & Evaluation
	
  * Training:
		
    * Epochs: 50
		
    * Training Accuracy: ~96% by final epoch
		
    * Validation Accuracy: ~91% by final epoch
	
  * Model Saved As: my_model.keras

## 📊 Model Performance Analysis
	
  * Accuracy Curve:
		
    * Training and validation accuracy plotted to ensure no overfitting.
		
    * Model showed consistent performance over epochs.
	
  * Evaluation on Test Data:
		
    * Test Accuracy: ~96.2%
		
    * Loss: Very low, indicating a well-trained model.

## 🎨 Visualization of Model Predictions
	
  * Visualized test images with predictions to verify model performance.
	
  * Predictions aligned with actual labels, indicating strong classification capability.

## ⚡ Challenges Faced
	
  * Dataset Limitation: Limited images per class.
	
  * Overfitting Risk: Handled through augmentation and dropout layers.
	
  * Computational Time: High number of epochs increased training time.

## ✅ Conclusion
	
  * The CNN model successfully detects rice leaf diseases with high accuracy (~96%).
	
  * The model can be further optimized by:
		
    * Increasing dataset size

    * Exploring more advanced CNN architectures
		
    * Fine-tuning hyperparameters to minimize overfitting.
