# Task: Object Detection with YOLO-based Fine-Tuning

## 1) Problem Statement

Build a real-time object detection system capable of identifying and classifying **custom objects** — including bottles, flowers, and tools — from live webcam input. The system should:

* Work in real-time
* Accurately classify across 9 defined object categories
* Be lightweight enough for edge deployment
* Offer an intuitive web interface for end-users

---

## 2) Project Architecture

1. **Dataset Collection & Augmentation**
   Images collected for 9 object classes; data augmented using Albumentations.

2. **Annotation**
   Used Roboflow to annotate and export in YOLOv5 format.

3. **Model Training**
   YOLOv5s was fine-tuned using custom data over 100 epochs.

4. **Model Evaluation**
   Precision, Recall, mAP50, and misclassifications tracked.

5. **Deployment**
   Live webcam inference using OpenCV + Flask + HTML frontend.

---

## 3) Classes

**1) Water Bottles**

Borosil Bottle

Plastic Bottle

Tupperware Bottle

**2) Flowers**

Daisy Flower

Hibiscus Flower

Rose Flower

**3) Tools**

Screwdriver

Hammer

Spanner

---

## 4) Tools & Technologies Used

* **YOLOv5**: For training and detection
* **Python & OpenCV**: Real-time video processing
* **Flask**: Web interface for deployment
* **Albumentations**: Image augmentation
* **Roboflow**: Dataset annotation and export
* **Google Colab / Local**: For training environment

---

## 5) Methodology

1. Collect & annotate a balanced dataset
2. Apply data augmentation to boost performance
3. Train YOLOv5m model with custom data
4. Evaluate using mAP, Precision, Recall
5. Deploy using Flask with webcam support

---

## 6) Component Breakdown

* `train.py`: Used to train YOLOv5m model
* `val.py`: Used for evaluating model performance
* `detect.py`: Runs inference on test images or webcam
* `app.py`: Flask backend for live detection
* `templates/index.html`: Frontend interface
* `/runs/train/`: Stores training outputs
* `/runs/detect/`: Stores detection results

---

## 7) Input and Output Specifications

### Input

* Image or webcam stream
* Format: JPG, PNG or live webcam frames

### Output

* Detected object with bounding boxes
* Real-time class label and confidence score overlay
* Detection saved or shown via web interface

---

## 8) Result

Achieved significant improvement in mAP and real-time detection accuracy after augmentation and retraining with:

* **Total images used**: 978 (balanced across 9 classes)
* **Final model**: YOLOv5s, trained for 100 epochs

Successfully deployed via Flask with live webcam stream.

---
## 9) Output

## 10) Future Improvements

* Fine-tune using YOLOv8 or other transformers
* Deploy on mobile/Jetson Nano for edge AI
* Add object count and audio feedback
