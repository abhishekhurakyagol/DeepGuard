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

DeepGuard Training/
│
├── backend/
│   ├── app.py
│   ├── auth.py
│   ├── database.py
│   ├── config.py
│   ├── security.py
│   ├── audit.py
│   ├── predict_video.py
│   ├── predict_audio.py
│   ├── video_utils.py
│   ├── audio_utils.py
│   ├── gradcam.py
│   ├── requirements.txt
│   └── routers/
│       ├── auth_router.py
│       ├── scans_router.py
│       └── audit_router.py
│
├── frontend/
│
├── models/
├── checkpoints/
├── configs/
├── dataset/
├── datasets/
├── scripts/
│   ├── 01_download_dataset.py
│   ├── 02_extract_frames.py
│   ├── 03_prepare_dataset.py
│   ├── 04_train_model.py
│   ├── 05_evaluate_model.py
│   └── 06_predict.py
│
├── training/
├── notebooks/
├── tests/
│   ├── test_stage1.py
│   └── test_stage_all.py
│
├── uploads/
├── outputs/
├── results/
├── logs/
├── test_images/
├── test_predict_video.py
└── test_video_utils.py

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

### Environment Configuration

Copy the template to `.env` and configure your settings:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `HOST` | API server host | `127.0.0.1` |
| `PORT` | API server port | `8000` |
| `ENVIRONMENT` | Deployment environment (`development` / `production`) | `development` |
| `CORS_ORIGINS` | Allowed frontend origins (comma-separated) | `http://localhost:5173,http://127.0.0.1:5173` |
| `MONGODB_URI` | MongoDB Atlas Connection String | `mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority` |
| `MONGODB_DB_NAME` | Database name | `deepguard` |
| `MONGODB_TIMEOUT_MS` | Server connection timeout (ms) | `3000` |
| `MAX_UPLOAD_SIZE_MB` | Maximum allowed upload size | `100` |

Run API

```bash
cd backend
uvicorn app:app --reload
```

Run Tests

```bash
pytest tests/ -v
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

### System Health Check

```
GET /health
```

Response

```json
{
  "status": "healthy",
  "api": "online",
  "database": {
    "status": "connected",
    "connected": true,
    "message": "Connected to MongoDB Atlas successfully.",
    "database_name": "deepguard"
  },
  "models": {
    "image_model": "loaded",
    "audio_model": "available",
    "device": "cuda"
  },
  "environment": "development",
  "version": "1.0.0"
}
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