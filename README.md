# Task: Object Detection with YOLO-based Fine-Tuning

> **Name**: Sujan S  
> **Roll No**: 22PD35  
> **Course**: MSc Data Science

## Video Explaination

https://drive.google.com/file/d/1GT3UNzcg7BkTRr5SejtTpRpy5XNP0Q9C/view?usp=sharing

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

**Results got after testing on data**

!python detect.py \
--weights /content/yolov5/runs/train/smartan_yolo_v5m_finalrun2/weights/best.pt \
--img 640 \
--conf 0.25 \
--source /content/drive/MyDrive/dataset/train/images/bottle08_aug4_jpg.rf.ba097f3be8145bea5aa969f6281c291a.jpg \
--name smartan_test

![WhatsApp Image 2025-06-25 at 17 04 02_0a5aff64](https://github.com/user-attachments/assets/32be4267-30d3-4597-8158-12793ca1e457)

![WhatsApp Image 2025-06-25 at 17 07 18_f6294710](https://github.com/user-attachments/assets/ed37217e-94be-4fd0-a399-59d975b31192)

![WhatsApp Image 2025-06-25 at 17 10 59_3d057225](https://github.com/user-attachments/assets/0e6bac51-e2db-4576-8afc-6f1f8155a05b)

![WhatsApp Image 2025-06-25 at 17 14 37_c6953667](https://github.com/user-attachments/assets/242f2677-1dd9-46f8-8b10-b848161b4f0c)

![WhatsApp Image 2025-06-25 at 17 15 26_b2effae7](https://github.com/user-attachments/assets/dcf68177-6e5f-4b32-a975-1b1ae409663c)

![WhatsApp Image 2025-06-25 at 17 16 11_8f4b30c3](https://github.com/user-attachments/assets/09ff6d60-fe26-4d93-8963-bea6b4bf1720)

![WhatsApp Image 2025-06-25 at 17 16 55_e35951c7](https://github.com/user-attachments/assets/9976e86c-eb41-4b8c-962f-5c2ad2d2875f)

![WhatsApp Image 2025-06-25 at 16 36 47_20839afa](https://github.com/user-attachments/assets/1773784e-eca2-4347-9072-412f61fbc8a2)

![WhatsApp Image 2025-06-25 at 15 10 01_3e8e4e83](https://github.com/user-attachments/assets/8004fb50-99d0-4e2e-88c0-8886133cc908)

**Frontend**

![Screenshot 2025-06-25 212948](https://github.com/user-attachments/assets/b9b28050-a1c8-4dda-a992-9c68b9e6dee3)

***Spanner***

![Screenshot (298)](https://github.com/user-attachments/assets/59d7b77c-306d-41f3-a35e-6fb975ef61c4)

***Hammer***

![Screenshot (306)](https://github.com/user-attachments/assets/7de575e9-5be6-4f20-803d-f1efa192cc37)

***Skrewdriver**

![Screenshot (309)](https://github.com/user-attachments/assets/f962a05a-e302-4b17-b296-ff2d7aa68b45)

***Tupperware Bottle***

![Screenshot (315)](https://github.com/user-attachments/assets/0accda48-7004-4fcf-9200-03bb256e1072)

***Plastic Bottle***

![Screenshot (320)](https://github.com/user-attachments/assets/8a19f3a2-5edf-4fe3-aaa9-f46e52ead317)

***Borosil Bottle***

![Screenshot (325)](https://github.com/user-attachments/assets/d21f9a3a-a6b8-45a9-b04a-72ee284bb676)

***Rose**

![Screenshot (330)](https://github.com/user-attachments/assets/780b53a9-a15d-4e46-b793-4dfedc1a0546)

***Hibiscus***

![Screenshot (334)](https://github.com/user-attachments/assets/5516eba3-af97-4811-83d2-f769f4aac5ea)

***Daisy***

![Screenshot (345)](https://github.com/user-attachments/assets/72683ed2-fc38-46f7-a50a-d61201c41e7f)




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
