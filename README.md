🌱 Plant Growth Prediction using CNN

 📌 Project Overview

This project focuses on leveraging **Convolutional Neural Networks (CNN)** to predict plant growth stages using image data. It also integrates **fertilizer recommendation** and **plant disease detection**, aiming to support smart and sustainable agriculture.

Developed as part of the final-year project at **Sandip Institute of Technology and Research Centre**, this solution automates growth monitoring and enhances agricultural decision-making through machine learning and image processing techniques.

 🎯 Objectives

1. **Plant Growth Stage Prediction**

   * Classify plant images into stages: *early*, *mid*, *mature* using CNN.
   * Real-time prediction through a simple web interface.

2. **Fertilizer Recommendation System**

   * Suggest optimal fertilizers based on crop type, soil conditions, and growth stage.

3. **Plant Disease Detection**

   * Detect plant diseases from leaf images using image classification models.



 🧠 Technologies Used

* **Programming:** Python
* **Deep Learning:** TensorFlow, Keras
* **Image Processing:** OpenCV, Pillow
* **Web Development:** Flask, HTML, CSS, JavaScript
* **Database:** SQLite / MySQL
* **Visualization:** Matplotlib, Seaborn
* **Development Platforms:** Google Colab, GitHub

---

 🏗️ System Architecture

 🔁 Workflow

1. **Image Preprocessing**

   * Resize to 224x224 pixels
   * Normalize pixel values
   * Apply augmentation (flip, rotate, zoom)

2. **Model Design (CNN)**

   * Conv → ReLU → Pooling → Dropout → Dense → Softmax
   * Compiled using *Adam Optimizer* and *categorical crossentropy*
   * Trained with early stopping and monitored validation performance

3. **Evaluation**

   * Metrics: Accuracy, Precision, Recall
   * Tools: Confusion Matrix, Classification Report

4. **Deployment**

   * Real-time prediction using Flask web interface
   * Upload image → Predict stage → Get fertilizer/disease insights

---

 💻 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/nileshsonawanes/Plant-Growth-Prediction-Using-CNN
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

---

 📊 Model Performance

* **Accuracy:** \~95% on validation data
* **Stability:** Minor overfitting observed but overall stable
* **Classification:** High precision and recall across all growth stages

---

 ✅ Test Cases

* 📈 Accurate classification into early, mid, mature growth stages
* 📤 Real-time image upload and instant prediction
* 💡 Fertilizer and disease suggestions based on prediction

---

 📌 Conclusion

This project provides an **AI-based smart agriculture solution**. It automates plant growth stage prediction, offers fertilizer recommendations, and detects plant diseases using image data—empowering farmers to make data-driven, sustainable decisions.

