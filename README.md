# 🛡️ DeepGuard AI

DeepGuard AI is an AI-powered Deepfake Detection System that identifies whether an uploaded image or video is **Real** or **Fake** using Deep Learning. The system provides a modern web interface with real-time predictions powered by a FastAPI backend and EfficientNet-B0 model.

---

## 📌 Features

- ✅ Image Deepfake Detection
- ✅ Video Deepfake Detection
- ✅ Drag & Drop Upload
- ✅ Image & Video Preview
- ✅ Confidence Score
- ✅ Frame-wise Video Analysis
- ✅ Responsive Modern UI
- 🚧 Audio Deepfake Detection (Coming Soon)
- 🚧 Live Camera Detection (Coming Soon)

---

## 🖥️ Tech Stack

### Frontend

- React.js
- CSS3
- Framer Motion
- Axios
- React Icons

### Backend

- FastAPI
- Python
- PyTorch
- TorchVision
- OpenCV
- Pillow

### AI Model

- EfficientNet-B0
- Transfer Learning
- Binary Classification

---

## 📂 Project Structure

```
DeepGuard/

│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app.py
│   ├── predict_video.py
│   ├── video_utils.py
│   └── requirements.txt
│
├── models/
│   └── deepguard_v1.pth
│
├── datasets/
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/deepguard-ai.git
cd deepguard-ai
```

---

## Backend Setup

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn app:app --reload
```

Backend runs at

```
http://127.0.0.1:8000
```

---

## Frontend Setup

```bash
cd frontend
```

Install packages

```bash
npm install
```

Run

```bash
npm run dev
```

Frontend runs at

```
http://localhost:5173
```

---

## API Endpoints

### Home

```
GET /
```

---

### Image Prediction

```
POST /predict
```

Upload

```
Image
```

Response

```json
{
  "prediction": "real",
  "confidence": 98.74
}
```

---

### Video Prediction

```
POST /predict-video
```

Upload

```
Video
```

Response

```json
{
  "prediction": "fake",
  "confidence": 96.35,
  "fake_frames": 42,
  "real_frames": 11,
  "total_frames": 53
}
```

---

## Dataset

Current training uses:

- FaceForensics++
- Original Videos
- DeepFake Videos

Future datasets

- ASVspoof 2021
- WaveFake
- DFDC
- Celeb-DF v2

---

## Model Performance

| Metric | Value |
|---------|--------|
| Model | EfficientNet-B0 |
| Classes | Real / Fake |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Output | Binary Classification |

---

## Future Improvements

- Audio Deepfake Detection
- Live Webcam Detection
- Explainable AI (Grad-CAM)
- Mobile Application
- Cloud Deployment
- User Authentication
- Detection History
- PDF Report Generation

---

## Screenshots

Add screenshots here.

```
Home Page

Upload Page

Prediction Result

Video Detection
```

---

## Author

**Abhishek Hurakyagol**

Artificial Intelligence & Data Science Engineering Student

---

## License

This project is developed for educational and research purposes.