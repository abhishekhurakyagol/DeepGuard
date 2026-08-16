import { useState } from "react";
import axios from "axios";
import {
  FaShieldAlt,
  FaCloudUploadAlt,
  FaImage,
  FaVideo,
  FaCheckCircle,
} from "react-icons/fa";

import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);

    if (selectedFile.type.startsWith("image")) {
      setPreview(URL.createObjectURL(selectedFile));
    } else if (selectedFile.type.startsWith("video")) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview(null);
    }
  };

  const onInputChange = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      let endpoint = "";

      if (file.type.startsWith("image")) {
        endpoint = "http://127.0.0.1:8000/predict";
      } else {
        endpoint = "http://127.0.0.1:8000/predict-video";
      }

      const response = await axios.post(endpoint, formData);

      setResult(response.data);
    } catch (err) {
      alert("Prediction Failed");
      console.log(err);
    }

    setLoading(false);
  };

  return (
    <div className="container">

      <div className="card">

        <FaShieldAlt className="logo" />

        <h1>DeepGuard AI</h1>

        <p className="subtitle">
          AI Powered Deepfake Image & Video Detection
        </p>

        <div
          className="upload-box"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <FaCloudUploadAlt className="upload-icon" />

          <h2>Drag & Drop Files</h2>

          <p>or choose a file below</p>

          <input
            type="file"
            accept="image/*,video/*"
            onChange={onInputChange}
          />
        </div>

        {file && (
          <div className="selected">

            <h3>Selected File</h3>

            <p>{file.name}</p>

            {preview &&
              file.type.startsWith("image") && (
                <img
                  src={preview}
                  alt="preview"
                  className="preview"
                />
              )}

            {preview &&
              file.type.startsWith("video") && (
                <video
                  src={preview}
                  controls
                  className="preview"
                />
              )}

            <button
              className="analyze-btn"
              onClick={analyze}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>

          </div>
        )}

        {result && (
          <div className="result">

            <h2>
              <FaCheckCircle /> Prediction Result
            </h2>

            <div
              className={
                result.prediction === "real"
                  ? "badge real"
                  : "badge fake"
              }
            >
              {result.prediction.toUpperCase()}
            </div>

            <div className="confidence">

              <p>
                Confidence : {result.confidence}%
              </p>

              <div className="progress">

                <div
                  className="progress-fill"
                  style={{
                    width: `${result.confidence}%`,
                  }}
                ></div>

              </div>

            </div>

            {"total_frames" in result && (
              <div className="stats">

                <div className="stat">
                  <FaVideo />
                  <span>Total</span>
                  <strong>{result.total_frames}</strong>
                </div>

                <div className="stat">
                  <FaImage />
                  <span>Fake</span>
                  <strong>{result.fake_frames}</strong>
                </div>

                <div className="stat">
                  <FaCheckCircle />
                  <span>Real</span>
                  <strong>{result.real_frames}</strong>
                </div>

              </div>
            )}

          </div>
        )}

      </div>

    </div>
  );
}

export default App;