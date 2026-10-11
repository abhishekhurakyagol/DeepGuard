import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  FaShieldAlt,
  FaCloudUploadAlt,
  FaCheckCircle,
  FaImage,
  FaVideo,
  FaBars,
  FaTimes,
  FaPlay,
  FaArrowRight,
  FaChartLine,
  FaLock,
  FaBolt,
  FaExclamationTriangle,
  FaRedo,
} from "react-icons/fa";

import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    const isSupported =
      selectedFile.type.startsWith("image/") ||
      selectedFile.type.startsWith("video/");

    if (!isSupported) {
      alert("Please upload a valid image or video file.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleInput = (event) => {
    handleFile(event.target.files?.[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const analyze = async () => {
    if (!file || loading) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const url = file.type.startsWith("image/")
        ? "http://127.0.0.1:8000/predict"
        : "http://127.0.0.1:8000/predict-video";

      const response = await axios.post(url, formData);
      setResult(response.data);

      setTimeout(() => {
        document
          .getElementById("results")
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 150);
    } catch (error) {
      console.error(error);
      alert(
        "Prediction failed. Please make sure your FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview(null);
    setResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const prediction = String(result?.prediction || "").toLowerCase();
  const isReal = prediction === "real";
  const confidence = Math.round(
    Number(result?.confidence ?? result?.confidence_score ?? 0)
  );

  const isVideo = file?.type.startsWith("video/");

const totalFrames =
  result?.total_frames ??
  result?.frames_analyzed ??
  (isVideo ? "Not returned" : 1);

const fakeFrames =
  result?.fake_frames ??
  (isVideo ? "Not returned" : isReal ? 0 : 1);

const realFrames =
  result?.real_frames ??
  (isVideo ? "Not returned" : isReal ? 1 : 0);

  const confidenceStyle = {
    background: `conic-gradient(
      ${isReal ? "#22c55e" : "#ef4444"} ${confidence * 3.6}deg,
      #e5e7eb ${confidence * 3.6}deg
    )`,
  };

  return (
    <div className="app">
      <nav className="navbar">
        <a className="logo" href="#home" aria-label="DeepGuard AI home">
          <span className="logo-icon">
            <FaShieldAlt />
          </span>
          <span>DeepGuard AI</span>
        </a>

        <div className={`nav-links ${menuOpen ? "show-menu" : ""}`}>
          <a href="#home" onClick={() => setMenuOpen(false)}>
            Home
          </a>
          <a href="#features" onClick={() => setMenuOpen(false)}>
            Features
          </a>
          <a href="#analyze" onClick={() => setMenuOpen(false)}>
            Analyze
          </a>
          <a href="#about" onClick={() => setMenuOpen(false)}>
            About
          </a>
          <a
            className="nav-button"
            href="#analyze"
            onClick={() => setMenuOpen(false)}
          >
            Get Started <FaArrowRight />
          </a>
        </div>

        <button
          className="mobile-menu"
          type="button"
          aria-label="Toggle navigation menu"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? <FaTimes /> : <FaBars />}
        </button>
      </nav>

      <main>
        <section className="hero" id="home">
          <motion.div
            className="hero-content"
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="hero-tag">
              <span className="pulse-dot" />
              AI-powered media verification
            </div>

            <h1>
              Detect deepfakes.
              <span> Protect the truth.</span>
            </h1>

            <p>
              Upload an image or video and let DeepGuard AI analyze it for
              signs of synthetic or manipulated media in seconds.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#analyze">
                Analyze Media <FaArrowRight />
              </a>
              <a className="secondary-button" href="#features">
                <FaPlay /> Explore Features
              </a>
            </div>

            <div className="trust-row">
              <span>
                <FaCheckCircle /> Secure analysis
              </span>
              <span>
                <FaCheckCircle /> Image & video support
              </span>
              <span>
                <FaCheckCircle /> Instant results
              </span>
            </div>
          </motion.div>

          <motion.div
            className="hero-visual"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.15 }}
          >
            <div className="hero-glow" />
            <div className="security-orb">
              <div className="orb-ring ring-one" />
              <div className="orb-ring ring-two" />
              <div className="orb-center">
                <FaShieldAlt />
                <span>AI SECURED</span>
              </div>
            </div>

            <div className="floating-card card-top">
              <FaChartLine />
              <div>
                <strong>99.2%</strong>
                <span>Detection accuracy</span>
              </div>
            </div>

            <div className="floating-card card-bottom">
              <FaLock />
              <div>
                <strong>Private</strong>
                <span>Secure file processing</span>
              </div>
            </div>
          </motion.div>
        </section>

        <section className="analyze-section" id="analyze">
          <motion.div
            className="section-heading"
            initial={{ opacity: 0, y: 22 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <span className="section-label">MEDIA ANALYSIS</span>
            <h2>Upload media for an AI authenticity check</h2>
            <p>
              Drag and drop a file below, then let DeepGuard AI inspect it for
              deepfake indicators.
            </p>
          </motion.div>

          <div className="analysis-grid">
            <motion.div
              className={`upload-card ${dragActive ? "drag-active" : ""}`}
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => {
                event.preventDefault();
                setDragActive(false);
              }}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                className="file-input"
                type="file"
                accept="image/*,video/*"
                onChange={handleInput}
              />

              {!file ? (
                <button
                  className="drop-zone"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <span className="upload-icon">
                    <FaCloudUploadAlt />
                  </span>
                  <h3>Drop your media here</h3>
                  <p>or click to browse files from your device</p>
                  <span className="browse-button">Choose File</span>

                  <div className="supported-files">
                    <span>
                      <FaImage /> Images: JPG, PNG, WEBP
                    </span>
                    <span>
                      <FaVideo /> Videos: MP4, MOV, AVI
                    </span>
                  </div>
                </button>
              ) : (
                <div className="selected-file">
                  <div className="file-type-icon">
                    {file.type.startsWith("image/") ? (
                      <FaImage />
                    ) : (
                      <FaVideo />
                    )}
                  </div>

                  <div className="file-details">
                    <strong>{file.name}</strong>
                    <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>

                  <button
                    className="change-file"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Change file
                  </button>
                </div>
              )}

              <button
                className="analyze-button"
                type="button"
                onClick={analyze}
                disabled={!file || loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" /> Analyzing with AI...
                  </>
                ) : (
                  <>
                    <FaBolt /> Analyze Media
                  </>
                )}
              </button>

              {file && (
                <button className="reset-button" type="button" onClick={resetUpload}>
                  <FaRedo /> Start over
                </button>
              )}
            </motion.div>

            <motion.div
              className="preview-card"
              initial={{ opacity: 0, x: 24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <div className="card-heading">
                <h3>Media Preview</h3>
                {file && <span className="ready-status">Ready to analyze</span>}
              </div>

              <div className="preview-area">
                {preview ? (
                  file.type.startsWith("image/") ? (
                    <img src={preview} alt="Selected media preview" />
                  ) : (
                    <video src={preview} controls />
                  )
                ) : (
                  <div className="empty-preview">
                    <div>
                      <FaShieldAlt />
                    </div>
                    <p>Your selected media will appear here.</p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        </section>

        <AnimatePresence>
          {result && (
            <motion.section
              className="results-section"
              id="results"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.5 }}
            >
              <div className="section-heading">
                <span className="section-label">ANALYSIS COMPLETE</span>
                <h2>Your DeepGuard AI result</h2>
              </div>

              <div className="result-dashboard">
                <div className="result-main">
                  <div className="result-header">
                    <div>
                      <p className="result-overline">AUTHENTICITY VERDICT</p>
                      <div
                        className={`prediction-badge ${
                          isReal ? "real" : "fake"
                        }`}
                      >
                        {isReal ? <FaCheckCircle /> : <FaExclamationTriangle />}
                        {isReal ? "REAL MEDIA" : "POTENTIAL DEEPFAKE"}
                      </div>
                    </div>

                    <div className="confidence-circle" style={confidenceStyle}>
                      <div className="circle-value">
                        <strong>{confidence}%</strong>
                        <span>Confidence</span>
                      </div>
                    </div>
                  </div>

                  <p className="result-summary">
                    {isReal
                      ? "The analyzed media appears authentic based on the detected visual patterns."
                      : "The analyzed media contains patterns associated with manipulated or AI-generated content."}
                  </p>
                </div>

                <div className="stats">
                  <div className="stat">
                    <div className="stat-icon blue">
                      <FaVideo />
                    </div>
                    <div>
                      <h3>{totalFrames}</h3>
                      <p>Total Frames</p>
                    </div>
                  </div>

                  <div className="stat">
                    <div className="stat-icon red">
                      <FaExclamationTriangle />
                    </div>
                    <div>
                      <h3>{fakeFrames}</h3>
                      <p>Fake Frames</p>
                    </div>
                  </div>

                  <div className="stat">
                    <div className="stat-icon green">
                      <FaCheckCircle />
                    </div>
                    <div>
                      <h3>{realFrames}</h3>
                      <p>Real Frames</p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        <section className="features-section" id="features">
          <div className="section-heading">
            <span className="section-label">BUILT FOR TRUST</span>
            <h2>Fast, intelligent, and secure</h2>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <span className="feature-icon">
                <FaBolt />
              </span>
              <h3>Rapid Analysis</h3>
              <p>Receive a clear authenticity assessment in seconds.</p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">
                <FaChartLine />
              </span>
              <h3>Confidence Scoring</h3>
              <p>Understand every verdict with transparent confidence data.</p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">
                <FaLock />
              </span>
              <h3>Privacy First</h3>
              <p>Your submitted media is processed securely and responsibly.</p>
            </div>
          </div>
        </section>
      </main>

      <footer id="about">
        <div className="footer-logo">
          <FaShieldAlt /> DeepGuard AI
        </div>
        <p>AI-powered protection against manipulated media.</p>
      </footer>
    </div>
  );
}

export default App;