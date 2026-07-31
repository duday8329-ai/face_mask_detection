# Face Mask Detection with Live Alert System

Real-time face mask detection using a webcam. Detects faces with OpenCV Haar
Cascades, classifies each face as **Mask** / **No Mask** with a MobileNetV2
CNN trained via transfer learning, and raises an on-screen alert when someone
is detected without a mask.

## Objective

Detect whether people are wearing face masks in real time using a webcam
feed, and alert when a person without a mask is detected.

## Tech Stack

- **Python 3.9+**
- **OpenCV** — face detection (Haar Cascades) and video I/O
- **TensorFlow / Keras** — MobileNetV2 transfer-learning CNN for mask classification
- **Flask** — optional web UI for single-image detection
- **scikit-learn** — train/test split, evaluation metrics
- **Matplotlib** — training curve visualization

## Project Structure

```
face-mask-detection/
├── dataset/
│   ├── with_mask/          # place masked-face images here
│   └── without_mask/       # place unmasked-face images here
├── model/                  # trained model + training plot saved here
├── static/
├── templates/
│   └── index.html          # Flask upload UI
├── train_mask_detector.py  # trains the CNN
├── detect_mask_video.py    # real-time webcam detection + alert
├── app.py                  # optional Flask app for image upload
└── requirements.txt
```

## Setup

1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get a dataset**
   Download a masked/unmasked face dataset from Kaggle, e.g. the
   ["Face Mask Detection ~12K Images Dataset"](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset),
   and place the images into:
   ```
   dataset/with_mask/
   dataset/without_mask/
   ```

3. **Train the model**
   ```bash
   python train_mask_detector.py --dataset dataset --model model/mask_detector.model
   ```
   This trains a MobileNetV2-based classifier, prints a classification
   report, and saves:
   - `model/mask_detector.model` — the trained model
   - `model/training_plot.png` — loss/accuracy curves

4. **Run real-time detection**
   ```bash
   python detect_mask_video.py --model model/mask_detector.model
   ```
   A window opens showing your webcam feed with bounding boxes and
   Mask/No Mask labels. If a face without a mask is detected continuously
   for more than 2 seconds, an on-screen **ALERT** banner appears and a
   message is logged to the console. Press `q` to quit.

5. **(Optional) Run the Flask web app**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000`, upload a photo, and view the annotated
   result in the browser.

## How It Works

1. **Face detection** — each video frame is converted to grayscale and
   passed through OpenCV's Haar Cascade classifier to locate face regions.
2. **Preprocessing** — each detected face is cropped, resized to 224×224,
   converted to RGB, and normalized with MobileNetV2's `preprocess_input`.
3. **Classification** — the preprocessed face is passed to the trained CNN,
   which outputs a `(mask, no_mask)` probability pair.
4. **Visualization** — a colored bounding box (green = mask, red = no mask)
   and confidence percentage are drawn on the frame.
5. **Alerting** — if a "No Mask" face persists for more than a short
   threshold, an alert banner is shown and logged.

## Results

After training, check `model/training_plot.png` for accuracy/loss curves
and the console classification report (precision, recall, F1-score) to
gauge model performance on the held-out test split.

## Possible Extensions

- Swap the Haar Cascade for a more accurate DNN face detector (e.g. SSD/ResNet).
- Add email/SMS alerting instead of an on-screen banner.
- Log detection events with timestamps for a compliance dashboard.
- Package with Docker for easier deployment.

## Dataset & Interview Prep

- Any Kaggle masked/unmasked face dataset works — just keep the two-folder
  structure (`with_mask/`, `without_mask/`).
- Review core CNN, transfer learning, and OpenCV concepts before interviews:
  what transfer learning is and why MobileNetV2 is a good fit for real-time
  inference, how Haar Cascades detect faces, and how to evaluate a
  classifier with precision/recall/F1.
