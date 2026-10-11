
import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { jsPDF } from "jspdf";
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
  FaFilePdf,
  FaMusic,
  FaFileAlt,
  FaDownload,
} from "react-icons/fa";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const MAX_FILE_SIZE_MB = 100;

const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes)) return "Unknown size";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

const getMediaType = (file) => {
  if (!file) return null;

  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";

  // Some browsers do not provide a MIME type.
  const extension = file.name.split(".").pop()?.toLowerCase();

  if (["jpg", "jpeg", "png", "webp", "bmp"].includes(extension)) {
    return "image";
  }

  if (["mp4", "mov", "avi", "mkv", "webm"].includes(extension)) {
    return "video";
  }

  if (["wav", "mp3", "flac", "ogg", "m4a"].includes(extension)) {
    return "audio";
  }

  return null;
};

const getErrorMessage = (error) => {
  if (error.response?.data?.detail) {
    const detail = error.response.data.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }

  if (error.code === "ECONNABORTED") {
    return "The analysis took too long. Try a smaller file or try again.";
  }

  if (!error.response) {
    return "Cannot connect to the DeepGuard API. Make sure your FastAPI backend is running at http://127.0.0.1:8000.";
  }

  if (error.response.status === 413) {
    return "The uploaded file is too large for the server.";
  }

  if (error.response.status === 415) {
    return "This file format is not supported by the backend.";
  }

  if (error.response.status === 503) {
    return "The required model is currently unavailable. Check your backend model configuration.";
  }

  return `Analysis failed (${error.response.status}). ${error.message || "Please try again."}`;
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  const fileInputRef = useRef(null);
  const previewRef = useRef(null);

  const mediaType = getMediaType(file);

  useEffect(() => {
    let active = true;

    axios
      .get(`${API_BASE}/`, { timeout: 5000 })
      .then(() => {
        if (active) setApiStatus("online");
      })
      .catch(() => {
        if (active) setApiStatus("offline");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    if (result) {
      document.getElementById("results")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [result]);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    const type = getMediaType(selectedFile);

    if (!type) {
      setErrorMessage(
        "Unsupported file. Choose an image, video, or audio file."
      );
      return;
    }

    if (selectedFile.size === 0) {
      setErrorMessage("The selected file is empty.");
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setErrorMessage(
        `The maximum frontend file size is ${MAX_FILE_SIZE_MB} MB. Choose a smaller file.`
      );
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setErrorMessage("");
  };

  const handleInput = (event) => {
    handleFile(event.target.files?.[0]);

    // Allow selecting the same file again.
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const analyze = async () => {
    if (!file || loading) return;

    const type = getMediaType(file);

    if (!type) {
      setErrorMessage("Please select a supported media file.");
      return;
    }

    setLoading(true);
    setErrorMessage("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpointByType = {
      image: "/predict",
      video: "/predict-video",
      audio: "/predict-audio",
    };

    try {
      const response = await axios.post(
        `${API_BASE}${endpointByType[type]}`,
        formData,
        {
          timeout: 180000,
        }
      );

      if (
        !response.data ||
        typeof response.data.prediction !== "string"
      ) {
        throw new Error(
          "The backend response does not contain a valid prediction."
        );
      }

      setResult({
        ...response.data,
        media_type: response.data.media_type || type,
        filename: file.name,
        file_size: file.size,
        analyzed_at: new Date().toISOString(),
      });
    } catch (error) {
      console.error("DeepGuard analysis error:", error);
      setErrorMessage(
        error.message ===
          "The backend response does not contain a valid prediction."
          ? error.message
          : getErrorMessage(error)
      );
    } finally {
      setLoading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResult(null);
    setErrorMessage("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const exportReport = () => {
    if (!result) return;

    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const margin = 18;
    const contentWidth = pageWidth - margin * 2;
    let y = 22;

    const addWrappedText = (text, fontSize = 10) => {
      pdf.setFontSize(fontSize);
      const lines = pdf.splitTextToSize(
        String(text ?? "Not provided"),
        contentWidth
      );

      pdf.text(lines, margin, y);
      y += lines.length * (fontSize * 0.48) + 5;
    };

    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(21);
    pdf.text("DeepGuard AI", margin, y);
    y += 9;

    pdf.setFont("helvetica", "normal");
    addWrappedText("Media Authenticity Analysis Report", 13);

    pdf.setDrawColor(100, 116, 139);
    pdf.line(margin, y, pageWidth - margin, y);
    y += 10;

    const prediction = String(result.prediction || "Unknown").toUpperCase();
    const confidence = Number(
      result.confidence ?? result.confidence_score
    );

    pdf.setFont("helvetica", "bold");
    addWrappedText("ANALYSIS RESULT", 12);

    pdf.setFont("helvetica", "normal");
    addWrappedText(`Verdict: ${prediction}`);
    addWrappedText(
      `Confidence: ${
        Number.isFinite(confidence)
          ? `${Math.round(confidence * 100) / 100}%`
          : "Not provided"
      }`
    );
    addWrappedText(`Media type: ${result.media_type || mediaType}`);
    addWrappedText(`Filename: ${result.filename || file?.name}`);
    addWrappedText(`File size: ${formatBytes(result.file_size)}`);
    addWrappedText(
      `Analysis date: ${new Date(result.analyzed_at || Date.now()).toLocaleString()}`
    );

    y += 3;
    pdf.setFont("helvetica", "bold");
    addWrappedText("MODEL DETAILS", 12);

    pdf.setFont("helvetica", "normal");
    addWrappedText(
      `Frames analyzed: ${result.total_frames ?? result.frames_analyzed ?? "Not returned"}`
    );
    addWrappedText(
      `Fake frames: ${result.fake_frames ?? "Not returned"}`
    );
    addWrappedText(
      `Real frames: ${result.real_frames ?? "Not returned"}`
    );

    if (result.message) {
      addWrappedText(`Backend message: ${result.message}`);
    }

    y += 3;
    pdf.setFont("helvetica", "bold");
    addWrappedText("IMPORTANT NOTE", 12);

    pdf.setFont("helvetica", "normal");
    addWrappedText(
      "This report reflects the output of the configured AI model. A prediction or confidence score is not proof of authenticity. Results may be incorrect and should be reviewed alongside other evidence."
    );

    pdf.save("DeepGuard-Analysis-Report.pdf");
  };

  const prediction = String(result?.prediction || "").toLowerCase();
  const isReal = prediction === "real";
  const isFake = ["fake", "deepfake", "manipulated"].some((label) =>
    prediction.includes(label)
  );

  const confidenceValue = Number(
    result?.confidence ?? result?.confidence_score
  );

  const confidence = Number.isFinite(confidenceValue)
    ? Math.max(0, Math.min(100, Math.round(confidenceValue)))
    : null;

  const verdictLabel = isReal
    ? "REAL CLASSIFICATION"
    : isFake
      ? "FAKE CLASSIFICATION"
      : prediction
        ? prediction.toUpperCase()
        : "UNKNOWN";

  const verdictClass = isReal ? "real" : isFake ? "fake" : "unknown";

  const confidenceStyle =
    confidence === null
      ? {}
      : {
          background: `conic-gradient(
            ${isReal ? "#22c55e" : isFake ? "#ef4444" : "#2563eb"}
            ${confidence * 3.6}deg,
            #e5e7eb ${confidence * 3.6}deg
          )`,
        };

  const totalFrames =
    result?.total_frames ?? result?.frames_analyzed ?? null;

  const fakeFrames = result?.fake_frames ?? null;
  const realFrames = result?.real_frames ?? null;

  
const heatmap = result?.heatmap || result?.gradcam;
const spectrogram = result?.spectrogram;

// Use only observations returned by the backend.
// Do not invent forensic findings from the classification score alone.
const forensicSummary =
  result?.executive_summary ||
  (isFake
    ? "The model classified this media as potentially manipulated or synthetic. This is an automated screening result, not independent proof of manipulation."
    : isReal
      ? "The model classified this media as real. This classification does not rule out subtle manipulation or establish the media's provenance."
      : "The model returned an inconclusive or unfamiliar classification. Further review is required.");

const forensicRecommendation =
  result?.recommendation ||
  "Verify the original source, preserve the original file, and compare it with independently verified reference material. Escalate for specialist review if the media will be used as evidence.";

const forensicObservations = Array.isArray(result?.forensic_observations)
  ? result.forensic_observations
  : [];

const featureWeights = Array.isArray(result?.feature_weights)
  ? result.feature_weights
  : [];

  

  const mediaIcon =
    mediaType === "image" ? (
      <FaImage />
    ) : mediaType === "video" ? (
      <FaVideo />
    ) : (
      <FaMusic />
    );

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
              Analyze images, videos, and audio using your configured
              DeepGuard AI models to identify potential signs of manipulated
              media.
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
                <FaCheckCircle /> Three media types
              </span>
              <span>
                <FaCheckCircle /> Model-based results
              </span>
              <span>
                <FaLock /> Local API processing
              </span>
            </div>

            <p className="api-status">
              API status:{" "}
              <strong>
                {apiStatus === "checking"
                  ? "Checking..."
                  : apiStatus === "online"
                    ? "Connected"
                    : "Not connected"}
              </strong>
            </p>
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
                <span>AI ANALYSIS</span>
              </div>
            </div>

            <div className="floating-card card-top">
              <FaChartLine />
              <div>
                <strong>AI Models</strong>
                <span>Image · Video · Audio</span>
              </div>
            </div>

            <div className="floating-card card-bottom">
              <FaLock />
              <div>
                <strong>Privacy aware</strong>
                <span>Processed by your API</span>
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
              Select an image, video, or audio recording. DeepGuard will send
              it to the corresponding backend endpoint.
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
                accept="image/*,video/*,audio/*,.wav,.mp3,.flac,.ogg,.m4a,.mp4,.mov,.avi,.mkv,.webm"
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
                    <span>
                      <FaMusic /> Audio: WAV, MP3, FLAC
                    </span>
                  </div>
                  <p className="file-size-note">
                    Maximum frontend file size: {MAX_FILE_SIZE_MB} MB
                  </p>
                </button>
              ) : (
                <div className="selected-file">
                  <div className="file-type-icon">{mediaIcon}</div>
                  <div className="file-details">
                    <strong title={file.name}>{file.name}</strong>
                    <span>{formatBytes(file.size)}</span>
                    <span>
                      {mediaType?.toUpperCase()} file selected
                    </span>
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
                    <span className="spinner" /> Analyzing media...
                  </>
                ) : (
                  <>
                    <FaBolt /> Analyze Media
                  </>
                )}
              </button>

              {file && (
                <button
                  className="reset-button"
                  type="button"
                  onClick={resetUpload}
                  disabled={loading}
                >
                  <FaRedo /> Start over
                </button>
              )}

              {errorMessage && (
                <div className="error-message" role="alert">
                  <FaExclamationTriangle />
                  <span>{errorMessage}</span>
                </div>
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

              <div className="preview-area" ref={previewRef}>
                {preview && mediaType === "image" ? (
                  <img src={preview} alt="Selected image preview" />
                ) : preview && mediaType === "video" ? (
                  <video src={preview} controls />
                ) : preview && mediaType === "audio" ? (
                  <div className="audio-preview">
                    <FaMusic />
                    <p>{file.name}</p>
                    <audio src={preview} controls />
                  </div>
                ) : (
                  <div className="empty-preview">
                    <div>
                      <FaShieldAlt />
                    </div>
                    <p>Your selected media will appear here.</p>
                  </div>
                )}
              </div>

              {file && (
                <div className="preview-meta">
                  <span>{mediaType?.toUpperCase()}</span>
                  <span>{formatBytes(file.size)}</span>
                </div>
              )}
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
                <p>
                  This report reflects the prediction returned by your
                  configured backend model.
                </p>
              </div>

              <div className="result-dashboard">
                
<div className="forensic-report">
  <div className="forensic-report-header">
    <div>
      <span className="section-label">DEEPGUARD AI · FORENSIC REPORT</span>
      <h2>Media Authenticity Assessment</h2>
      <p>Automated analysis and evidence review</p>
    </div>
    <div className={`forensic-status ${verdictClass}`}>
      {isReal ? <FaCheckCircle /> : <FaExclamationTriangle />}
      {verdictLabel}
    </div>
  </div>

  <div className="forensic-metrics">
    <div className="forensic-metric">
      <span>Media type</span>
      <strong>{String(result.media_type || mediaType).toUpperCase()}</strong>
    </div>
    <div className="forensic-metric">
      <span>Model confidence</span>
      <strong>{confidence === null ? "Not provided" : `${confidence}%`}</strong>
    </div>
    <div className="forensic-metric">
      <span>Frames analyzed</span>
      <strong>{totalFrames ?? "Not applicable / not returned"}</strong>
    </div>
    <div className="forensic-metric">
      <span>Report status</span>
      <strong>Analysis complete</strong>
    </div>
  </div>

  <section className="forensic-panel">
    <div className="forensic-section-title">
      <span className="forensic-number">01</span>
      <div>
        <h3>Executive Forensic Summary</h3>
        <p>Interpretation of the model output</p>
      </div>
    </div>
    <p className="forensic-body">{forensicSummary}</p>
    <div className="recommendation-box">
      <strong><FaShieldAlt /> RECOMMENDATION</strong>
      <p>{forensicRecommendation}</p>
    </div>
  </section>

  <section className="forensic-panel">
    <div className="forensic-section-title">
      <span className="forensic-number">02</span>
      <div>
        <h3>Forensic Observations &amp; Anomaly Indicators</h3>
        <p>Findings supplied by the analysis pipeline</p>
      </div>
    </div>

    {forensicObservations.length > 0 ? (
      <div className="observation-list">
        {forensicObservations.map((item, index) => (
          <div className="observation-item" key={`${index}-${item.title || "finding"}`}>
            <span className="observation-marker">
              <FaExclamationTriangle />
            </span>
            <div>
              <h4>{item.title || item.category || `Observation ${index + 1}`}</h4>
              <p>{item.description || item.detail || "No additional detail was provided."}</p>
              {item.confidence != null && (
                <span className="observation-confidence">
                  Model-reported score: {item.confidence}%
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    ) : (
      <div className="no-evidence">
        <FaFileAlt />
        <div>
          <strong>No feature-level forensic findings returned</strong>
          <p>
            The current API response provides a classification and confidence
            score, but no verified eye, skin-texture, lighting, or edge-blending
            observations. These require a dedicated analysis method and should
            not be inferred from the overall prediction.
          </p>
        </div>
      </div>
    )}
  </section>

  <section className="forensic-panel">
    <div className="forensic-section-title">
      <span className="forensic-number">03</span>
      <div>
        <h3>Top Contributing Feature Weights</h3>
        <p>Grad-CAM and feature-level evidence</p>
      </div>
    </div>

    {featureWeights.length > 0 ? (
      <div className="feature-weight-list">
        {featureWeights.map((item, index) => {
          const weight = Math.max(
            0,
            Math.min(100, Number(item.weight ?? item.percentage ?? 0))
          );

          return (
            <div className="feature-weight" key={`${index}-${item.name || "feature"}`}>
              <div className="feature-weight-heading">
                <strong>{item.name || item.label || `Feature ${index + 1}`}</strong>
                <span>{weight}%</span>
              </div>
              <div
                className="feature-weight-track"
                role="progressbar"
                aria-valuenow={weight}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={item.name || "Feature weight"}
              >
                <div style={{ width: `${weight}%` }} />
              </div>
              {item.description && <p>{item.description}</p>}
            </div>
          );
        })}
      </div>
    ) : (
      <div className="no-evidence">
        <FaChartLine />
        <div>
          <strong>Feature weights unavailable</strong>
          <p>
            The current model response does not contain numerical
            feature-contribution weights. The overall confidence score is not
            a substitute for individual feature weights.
          </p>
        </div>
      </div>
    )}

    {heatmap && mediaType === "image" && (
      <div className="explainability-block">
        <h4>Grad-CAM Visual Explanation</h4>
        <p>
          Model-generated visualization. Highlighted regions indicate areas
          influencing the model, not necessarily confirmed manipulation.
        </p>
        <img
          className="explanation-image"
          src={
            heatmap.startsWith("data:")
              ? heatmap
              : `data:image/png;base64,${heatmap}`
          }
          alt="Model-generated Grad-CAM explanation"
        />
      </div>
    )}

    {spectrogram && mediaType === "audio" && (
      <div className="explainability-block">
        <h4>Audio Spectrogram</h4>
        <img
          className="explanation-image"
          src={
            spectrogram.startsWith("data:")
              ? spectrogram
              : `data:image/png;base64,${spectrogram}`
          }
          alt="Backend-generated audio spectrogram"
        />
      </div>
    )}
  </section>

  <section className="forensic-panel">
    <div className="forensic-section-title">
      <span className="forensic-number">04</span>
      <div>
        <h3>Evidence Record</h3>
        <p>Metadata returned or available during this session</p>
      </div>
    </div>

    <div className="evidence-table-wrap">
      <table className="evidence-table">
        <tbody>
          <tr><th>File name</th><td>{result.filename || file?.name || "—"}</td></tr>
          <tr><th>Media type</th><td>{String(result.media_type || mediaType).toUpperCase()}</td></tr>
          <tr><th>Model prediction</th><td>{result.prediction || "—"}</td></tr>
          <tr><th>Confidence</th><td>{confidence === null ? "Not returned" : `${confidence}%`}</td></tr>
          <tr><th>Total frames</th><td>{totalFrames ?? "Not returned"}</td></tr>
          <tr><th>Fake frames</th><td>{fakeFrames ?? "Not returned"}</td></tr>
          <tr><th>Real frames</th><td>{realFrames ?? "Not returned"}</td></tr>
          <tr><th>File size</th><td>{formatBytes(result.file_size ?? file?.size)}</td></tr>
          <tr>
            <th>Analysis date</th>
            <td>{new Date(result.analyzed_at || Date.now()).toLocaleString()}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>

  <div className="forensic-footer">
    <div>
      <FaLock />
      <span>
        <strong>Interpret with care</strong>
        <p>
          AI predictions are probabilistic and can be incorrect. Preserve the
          original media and independently verify important findings.
        </p>
      </span>
    </div>
    <button
      className="analyze-button export-button"
      type="button"
      onClick={exportReport}
    >
      <FaFilePdf /> Export PDF Report <FaDownload />
    </button>
  </div>
</div>

                <div className="stats">
                  {mediaType === "video" ? (
                    <>
                      <div className="stat">
                        <div className="stat-icon blue">
                          <FaVideo />
                        </div>
                        <div>
                          <h3>{totalFrames ?? "—"}</h3>
                          <p>Frames Analyzed</p>
                        </div>
                      </div>

                      <div className="stat">
                        <div className="stat-icon red">
                          <FaExclamationTriangle />
                        </div>
                        <div>
                          <h3>{fakeFrames ?? "—"}</h3>
                          <p>Fake Frames</p>
                        </div>
                      </div>

                      <div className="stat">
                        <div className="stat-icon green">
                          <FaCheckCircle />
                        </div>
                        <div>
                          <h3>{realFrames ?? "—"}</h3>
                          <p>Real Frames</p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="stat">
                        <div className="stat-icon blue">{mediaIcon}</div>
                        <div>
                          <h3>{mediaType?.toUpperCase() || "MEDIA"}</h3>
                          <p>Media Type</p>
                        </div>
                      </div>

                      <div className="stat">
                        <div className="stat-icon green">
                          <FaCheckCircle />
                        </div>
                        <div>
                          <h3>
                            {confidence === null ? "—" : `${confidence}%`}
                          </h3>
                          <p>Model Confidence</p>
                        </div>
                      </div>

                      <div className="stat">
                        <div className="stat-icon blue">
                          <FaLock />
                        </div>
                        <div>
                          <h3>Complete</h3>
                          <p>Request Status</p>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="evidence-section">
                <div className="card-heading">
                  <h3>Evidence Summary</h3>
                </div>

                <div className="evidence-table-wrap">
                  <table className="evidence-table">
                    <tbody>
                      <tr>
                        <th>File name</th>
                        <td>{result.filename || file?.name || "—"}</td>
                      </tr>
                      <tr>
                        <th>Media type</th>
                        <td>
                          {String(result.media_type || mediaType).toUpperCase()}
                        </td>
                      </tr>
                      <tr>
                        <th>Prediction</th>
                        <td>{result.prediction}</td>
                      </tr>
                      <tr>
                        <th>Confidence score</th>
                        <td>
                          {confidence === null ? "Not returned" : `${confidence}%`}
                        </td>
                      </tr>
                      <tr>
                        <th>Frames analyzed</th>
                        <td>{totalFrames ?? "Not returned"}</td>
                      </tr>
                      <tr>
                        <th>Fake frames</th>
                        <td>{fakeFrames ?? "Not returned"}</td>
                      </tr>
                      <tr>
                        <th>Real frames</th>
                        <td>{realFrames ?? "Not returned"}</td>
                      </tr>
                      <tr>
                        <th>Analysis time</th>
                        <td>
                          {new Date(
                            result.analyzed_at || Date.now()
                          ).toLocaleString()}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {heatmap && mediaType === "image" && (
                <div className="evidence-section">
                  <div className="card-heading">
                    <h3>Image Explainability</h3>
                  </div>
                  <p>
                    Model explanation image returned by the backend. Its
                    meaning depends on the explanation method used.
                  </p>
                  <img
                    className="explanation-image"
                    src={
                      heatmap.startsWith("data:")
                        ? heatmap
                        : `data:image/png;base64,${heatmap}`
                    }
                    alt="Model-generated image explanation"
                  />
                </div>
              )}

              {spectrogram && mediaType === "audio" && (
                <div className="evidence-section">
                  <div className="card-heading">
                    <h3>Audio Spectrogram</h3>
                  </div>
                  <p>
                    Spectrogram returned by the backend for this audio
                    analysis.
                  </p>
                  <img
                    className="explanation-image"
                    src={
                      spectrogram.startsWith("data:")
                        ? spectrogram
                        : `data:image/png;base64,${spectrogram}`
                    }
                    alt="Audio spectrogram"
                  />
                </div>
              )}

              <p className="disclaimer">
                <FaLock /> AI results are not definitive proof of authenticity.
                Confidence scores can be wrong, and performance depends on the
                trained model, input quality, and data distribution.
              </p>
            </motion.section>
          )}
        </AnimatePresence>

        <section className="features-section" id="features">
          <div className="section-heading">
            <span className="section-label">BUILT FOR MEDIA ANALYSIS</span>
            <h2>One platform, three media types</h2>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <span className="feature-icon">
                <FaImage />
              </span>
              <h3>Image Analysis</h3>
              <p>
                Submit images to the configured image classification model.
              </p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">
                <FaVideo />
              </span>
              <h3>Video Analysis</h3>
              <p>
                Submit video files to your backend video analysis pipeline.
              </p>
            </div>

            <div className="feature-card">
              <span className="feature-icon">
                <FaMusic />
              </span>
              <h3>Audio Analysis</h3>
              <p>
                Submit supported audio recordings to the trained audio model.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer id="about">
        <div className="footer-logo">
          <FaShieldAlt /> DeepGuard AI
        </div>
        <p>AI-assisted detection of potentially manipulated media.</p>
        <p className="footer-note">
          For research and informational use. Always verify important findings
          independently.
        </p>
      </footer>
    </div>
  );
}

export default App;
