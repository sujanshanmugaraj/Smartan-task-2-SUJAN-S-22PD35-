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
   YOLOv5m was fine-tuned using custom data over 100 epochs.

   ![Screenshot 2025-06-25 133432](https://github.com/user-attachments/assets/76fec44d-0ae8-461d-a945-89291cf416eb)


5. **Model Evaluation**
   Precision, Recall, mAP50, and misclassifications tracked.

6. **Deployment**
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

![Screenshot 2025-06-24 220725](https://github.com/user-attachments/assets/b8c7e5d1-bb36-46dc-89cd-598dad451f44)

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
## 10) Limitations & Challenges

**1) Small Dataset for Some Classes**


Initially had fewer images (under 50) for tupperware_bottle, borosil_bottle, etc., which impacted early model performance.

**2) Class Confusion**


Visually similar items like hammer, spanner, and screwdriver were often misclassified due to overlapping features.

**3) Long Training Time on CPU**


Local training took over 9 hours for 100 epochs, delaying experiments—partially resolved using Colab GPU.


## 11) Future Improvements

* Fine-tune using YOLOv8 or other transformers
* Deploy on mobile/Jetson Nano for edge AI
* Add object count and audio feedback
