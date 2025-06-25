# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import base64
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from pathlib import Path
# import pathlib
# import time

# # Fix for Windows path issues with torch
# pathlib.PosixPath = pathlib.WindowsPath

# # === YOLOv5 Imports ===
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import non_max_suppression
# from yolov5.utils.augmentations import letterbox
# from yolov5.utils.torch_utils import select_device

# # === scale_coords function ===
# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     if ratio_pad is None:
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]
#     coords[:, [0, 2]] -= pad[0]
#     coords[:, [1, 3]] -= pad[1]
#     coords[:, :4] /= gain
#     coords[:, :4] = coords[:, :4].clamp(min=0)
#     return coords

# # === Flask App Setup ===
# app = Flask(__name__, template_folder='template')
# CORS(app)
# device = select_device('')

# # === Load Model ===
# yolov5_path = Path(__file__).resolve().parent / "yolov5"
# weights_path = yolov5_path / "runs/train/smartan_yolo_run_final/weights/best.pt"
# model = DetectMultiBackend(str(weights_path), device=device)
# names = model.names
# stride = model.stride

# # === Custom class mapping (change as per your classes) ===
# # CLASS_MAPPING = {
# #     'borosil_bottle': 'Borosil Bottle',
# #     'plastic_bottle': 'Plastic Bottle',
# #     'tupperware_bottle': 'Tupperware Bottle',
# #     'daisy_flower': 'Daisy Flower',
# #     'hibiscus_flower': 'Hibiscus Flower',
# #     'rose_flower': 'Rose Flower',
# #     'screwdriver': 'Screwdriver',
# #     'hammer': 'Hammer',
# #     'spanner': 'Spanner'
# # }

# CLASS_MAPPING = {
    
#     'hibiscus_flower': 'Hibiscus Flower',
   
# }
# # === Serve HTML ===
# @app.route('/', methods=['GET'])
# def index():
#     try:
#         return open('template/index.html', encoding='utf-8').read()
#     except FileNotFoundError:
#         return "<h3>🚀 Please save frontend as template/index.html</h3>"

# # === Detection Endpoint ===
# @app.route('/detect', methods=['POST'])
# def detect():
#     try:
#         t_start = time.time()
#         data = request.get_json()
#         img_data = data['image'].split(',')[1]
#         image = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
#         img = np.array(image)

#         img0 = img.copy()
#         img = letterbox(img, 640, stride=stride, auto=True)[0]
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#         img_tensor = torch.from_numpy(img).to(device).float() / 255.0
#         img_tensor = img_tensor.unsqueeze(0)

#         pred = model(img_tensor)
#         pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

#         detections = []
#         for det in pred:
#             if len(det):
#                 det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
#                 for *xyxy, conf, cls in det:
#                     class_name = names[int(cls)]
#                     if class_name in CLASS_MAPPING:
#                         x1, y1, x2, y2 = map(int, xyxy)
#                         detections.append({
#                             "class": class_name,
#                             "display_name": CLASS_MAPPING[class_name],
#                             "confidence": round(float(conf) * 100, 1),
#                             "bbox": [x1, y1, x2, y2]
#                         })

#         fps = round(1.0 / (time.time() - t_start), 1)
#         return jsonify({
#             "success": True,
#             "detections": detections,
#             "stats": {
#                 "fps": fps,
#                 "total_detections": len(detections)
#             }
#         })

#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500

# if __name__ == '__main__':
#     print("🚀 Flask YOLOv5 Detection Server running at http://localhost:5000")
#     app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)




from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import cv2
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path
import pathlib
import time

# === PATCH: Fix PosixPath error on Windows ===
pathlib.PosixPath = pathlib.WindowsPath

# === YOLOv5 Imports ===
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        # Calculate gain and padding
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # scaling factor
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,  # width padding
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # height padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # De-pad and rescale to original coordinates
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip coordinates to image size
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2

    return coords

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication
device = select_device('')

# === Load YOLOv5 model ===
yolov5_path = Path(__file__).resolve().parent / "yolov5"
weights_path = yolov5_path / "runs/train/smartan_yolo_run_final/weights/best.pt"
model = DetectMultiBackend(str(weights_path), device=device)
names = model.names
stride = model.stride

# Class mapping for your 9 custom classes
CLASS_MAPPING = {
    'borosil_bottle': 'Borosil Bottle',
    'plastic_bottle': 'Plastic Bottle', 
    'tupperware_bottle': 'Tupperware Bottle',
    'daisy_flower': 'Daisy Flower',
    'hibiscus_flower': 'Hibiscus Flower',
    'rose_flower': 'Rose Flower',
    'screwdriver': 'Screwdriver',
    'hammer': 'Hammer',
    'spanner': 'Spanner'
}


# Stats tracking
detection_stats = {
    'total_detections': 0,
    'fps': 0,
    'last_frame_time': time.time()
}

# === Home route - serve the HTML interface ===
@app.route('/', methods=['GET'])
def home():
    # Read the HTML file content (you'll save the HTML as a separate file)
    try:
        with open('template/index1.html', 'r', encoding='utf-8') as f:
            html_content = f.read()

        return html_content
    except FileNotFoundError:
        return """
        <h1>🚀 YOLOv5 Flask Server Running!</h1>
        <p>Please save your HTML frontend as 'templates/index.html'</p>
        <p>API Endpoint: POST to /detect with base64 image data</p>
        """

# === Enhanced detection route ===
@app.route('/detect', methods=['POST'])
def detect():
    try:
        start_time = time.time()
        
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = np.array(img)

        # Preprocess
        img0 = img.copy()
        # img = letterbox(img, 640, stride=stride, auto=True)[0]
        img = letterbox(img, 416, stride=stride, auto=True)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Inference
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # Process detections
        detections = []
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    #class_id = int(cls)
                    class_name = names[cls]
                    if class_name.lower() != "hammer":
                        continue 
                    confidence = float(conf)
                    
                    # Map to your display names
                    #display_name = CLASS_MAPPING.get(class_name, class_name)
                    
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # detections.append({
                    #     'class': class_name,
                    #     'display_name': display_name,
                    #     'confidence': round(confidence * 100, 1),
                    #     'bbox': [x1, y1, x2, y2]
                    # })
                    detections.append({
                        'class': class_name,
                        'display_name': 'Hammer',
                        'confidence': round(confidence * 100, 1),
                        'bbox': [x1, y1, x2, y2]
})
                


        # Update stats
        processing_time = time.time() - start_time
        detection_stats['fps'] = round(1.0 / processing_time, 1)
        detection_stats['total_detections'] += len(detections)
        detection_stats['last_frame_time'] = time.time()

        return jsonify({
            'success': True,
            'detections': detections,
            'stats': {
                'fps': detection_stats['fps'],
                'total_detections': detection_stats['total_detections'],
                'processing_time': round(processing_time * 1000, 1)  # ms
            }
        })
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# === Stats endpoint ===
@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(detection_stats)

# === Health check ===
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': list(CLASS_MAPPING.values())
    })

if __name__ == '__main__':
    print("🚀 Starting YOLOv5 Flask Server...")
    print(f"📱 Model loaded on device: {device}")
    print(f"🎯 Detecting classes: {list(CLASS_MAPPING.values())}")
    print("🌐 Frontend will be available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)