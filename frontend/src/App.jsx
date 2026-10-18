
import { useCallback, useEffect, useRef, useState } from "react";
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
  FaUser,
  FaHistory,
  FaTrash,
  FaSignInAlt,
  FaSignOutAlt,
  FaUserPlus,
  FaChevronLeft,
  FaChevronRight,
  FaClock,
} from "react-icons/fa";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const MAX_FILE_SIZE_MB = 100;

// ──────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────────────────────────────────────────────
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
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (["jpg", "jpeg", "png", "webp", "bmp"].includes(ext)) return "image";
  if (["mp4", "mov", "avi", "mkv", "webm"].includes(ext)) return "video";
  if (["wav", "mp3", "flac", "ogg", "m4a"].includes(ext)) return "audio";
  return null;
};

const getErrorMessage = (error) => {
  if (error.response?.data?.detail) {
    const d = error.response.data.detail;
    return typeof d === "string" ? d : JSON.stringify(d);
  }
  if (error.code === "ECONNABORTED") return "The analysis took too long. Try a smaller file or try again.";
  if (!error.response) return "Cannot connect to the DeepGuard API. Make sure your FastAPI backend is running at http://127.0.0.1:8000.";
  if (error.response.status === 413) return "The uploaded file is too large for the server.";
  if (error.response.status === 415) return "This file format is not supported by the backend.";
  if (error.response.status === 503) return "The required model is currently unavailable.";
  if (error.response.status === 401) return "Session expired. Please log in again.";
  return `Analysis failed (${error.response.status}). ${error.message || "Please try again."}`;
};

// ──────────────────────────────────────────────────────────────────────────────
// Auth helpers
// ──────────────────────────────────────────────────────────────────────────────
const getStoredToken = () => {
  try { return localStorage.getItem("dg_token"); } catch { return null; }
};
const setStoredToken = (t) => {
  try { if (t) localStorage.setItem("dg_token", t); else localStorage.removeItem("dg_token"); } catch { /* noop */ }
};

const authHeaders = (token) =>
  token ? { Authorization: `Bearer ${token}` } : {};

// ──────────────────────────────────────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────────────────────────────────────

/** Waveform visualizer – 100-point bar chart */
function WaveformVisualizer({ waveform }) {
  if (!Array.isArray(waveform) || waveform.length === 0) return null;
  const max = Math.max(...waveform.map(Math.abs), 1e-6);
  return (
    <div className="waveform-wrap">
      <h4 className="waveform-title">Waveform Envelope</h4>
      <div className="waveform-bars" aria-label="Audio waveform visualization">
        {waveform.map((v, i) => {
          const pct = Math.round((Math.abs(v) / max) * 100);
          return (
            <div key={i} className="waveform-bar-col">
              <div
                className="waveform-bar"
                style={{ height: `${Math.max(pct, 4)}%` }}
                title={`Sample ${i + 1}: ${v.toFixed(4)}`}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Video timeline – per-frame row with timestamp + verdict */
function VideoTimeline({ timeline }) {
  if (!Array.isArray(timeline) || timeline.length === 0) return null;
  return (
    <div className="video-timeline-wrap">
      <h4 className="waveform-title">Frame-by-Frame Timeline</h4>
      <div className="video-timeline">
        {timeline.map((entry, i) => {
          const isFakeFrame = String(entry.prediction || "").toLowerCase().includes("fake");
          const confidence = Number.isFinite(Number(entry.confidence))
            ? Math.round(Number(entry.confidence))
            : null;
          return (
            <div key={i} className={`timeline-row ${isFakeFrame ? "frame-fake" : "frame-real"}`}>
              <span className="timeline-ts">
                <FaClock />
                {Number.isFinite(Number(entry.timestamp_sec))
                  ? `${Number(entry.timestamp_sec).toFixed(2)}s`
                  : `F${entry.frame_index ?? i}`}
              </span>
              <span className={`timeline-verdict ${isFakeFrame ? "verdict-fake" : "verdict-real"}`}>
                {isFakeFrame ? <FaExclamationTriangle /> : <FaCheckCircle />}
                {String(entry.prediction || "").toUpperCase()}
              </span>
              {confidence !== null && (
                <span className="timeline-confidence">{confidence}%</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Auth Modal */
function AuthModal({ mode, onSuccess, onClose }) {
  const [form, setForm] = useState({ username: "", email: "", password: "", confirm_password: "", full_name: "" });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const modalRef = useRef(null);
  const onCloseRef = useRef(onClose);
  const isRegister = mode === "register";

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    modalRef.current?.querySelector("input")?.focus();

    const handleKeyDown = (event) => {
      if (event.key === "Escape") onCloseRef.current();
      if (event.key === "Tab" && modalRef.current) {
        const focusable = modalRef.current.querySelectorAll('button, input:not([disabled]), a[href]');
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const updateField = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const submit = async (e) => {
    e.preventDefault();
    if (isRegister && form.password !== form.confirm_password) {
      setErr("Passwords do not match.");
      return;
    }
    setBusy(true);
    setErr("");
    try {
      if (isRegister) {
        const { confirm_password: _confirmPassword, ...registrationData } = form;
        const res = await axios.post(`${API_BASE}/api/auth/register`, registrationData);
        onSuccess(res.data.token, res.data.user);
      } else {
        const res = await axios.post(`${API_BASE}/api/auth/login`, {
          identifier: form.username,
          password: form.password,
        });
        onSuccess(res.data.token, res.data.user);
      }
    } catch (error) {
      setErr(
        error.response?.data?.detail ||
          (isRegister ? "Registration failed." : "Login failed. Check your credentials.")
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <motion.div
        ref={modalRef}
        className="modal-box"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
        initial={{ opacity: 0, scale: 0.93 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.93 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <span className="section-label">DEEPGUARD AI</span>
            <h3 id="auth-modal-title">{isRegister ? "Create your account" : "Welcome back"}</h3>
            <p>{isRegister ? "Save and review your media analysis history." : "Sign in to access your saved analyses."}</p>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <FaTimes />
          </button>
        </div>
        <form className="auth-form" onSubmit={submit}>
          {isRegister && (
            <>
              <label className="auth-field">
                <span>Full name <span className="field-optional">(optional)</span></span>
                <input className="auth-input" type="text" autoComplete="name" placeholder="Your full name" value={form.full_name} onChange={updateField("full_name")} />
              </label>
              <label className="auth-field">
                <span>Username</span>
                <input className="auth-input" type="text" autoComplete="username" placeholder="Choose a username" minLength={3} maxLength={50} required value={form.username} onChange={updateField("username")} />
              </label>
            </>
          )}
          {isRegister && (
            <label className="auth-field">
              <span>Email address</span>
              <input className="auth-input" type="email" autoComplete="email" placeholder="you@example.com" required value={form.email} onChange={updateField("email")} />
            </label>
          )}
          {!isRegister && (
            <label className="auth-field">
              <span>Email or username</span>
              <input className="auth-input" type="text" autoComplete="username" placeholder="Enter your email or username" required value={form.username} onChange={updateField("username")} />
            </label>
          )}
          <div className="auth-field">
            <label htmlFor="auth-password">Password</label>
            <span className="password-control">
              <input id="auth-password" className="auth-input" type={showPassword ? "text" : "password"} autoComplete={isRegister ? "new-password" : "current-password"} placeholder="Enter your password" minLength={isRegister ? 6 : undefined} maxLength={128} required value={form.password} onChange={updateField("password")} />
              <button className="password-toggle" type="button" onClick={() => setShowPassword((visible) => !visible)} aria-label={showPassword ? "Hide password" : "Show password"} aria-pressed={showPassword}>{showPassword ? "Hide" : "Show"}</button>
            </span>
          </div>
          {isRegister && (
            <div className="auth-field">
              <label htmlFor="auth-confirm-password">Confirm password</label>
              <span className="password-control">
                <input id="auth-confirm-password" className="auth-input" type={showConfirmPassword ? "text" : "password"} autoComplete="new-password" placeholder="Re-enter your password" minLength={6} maxLength={128} required value={form.confirm_password} onChange={updateField("confirm_password")} />
                <button className="password-toggle" type="button" onClick={() => setShowConfirmPassword((visible) => !visible)} aria-label={showConfirmPassword ? "Hide confirmation password" : "Show confirmation password"} aria-pressed={showConfirmPassword}>{showConfirmPassword ? "Hide" : "Show"}</button>
              </span>
            </div>
          )}
          {err && <p className="auth-error" role="alert"><FaExclamationTriangle /> {err}</p>}
          <button className="auth-submit" type="submit" disabled={busy}>
            {busy ? <><span className="spinner" /> Please wait…</> : isRegister ? <><FaUserPlus /> Register</> : <><FaSignInAlt /> Sign In</>}
          </button>
        </form>
      </motion.div>
    </div>
  );
}

/** Scan History Tab */
function ScanHistoryPanel({ token }) {
  const [scans, setScans] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [mediaFilter, setMediaFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const PAGE_SIZE = 10;

  const fetchScans = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setErr("");
    try {
      const params = new URLSearchParams({ page, limit: PAGE_SIZE });
      if (mediaFilter) params.set("media_type", mediaFilter);
      const res = await axios.get(`${API_BASE}/api/scans?${params}`, {
        headers: authHeaders(token),
      });
      setScans(res.data.scans || []);
      setTotal(res.data.total || 0);
    } catch (error) {
      setErr(error.response?.data?.detail || "Failed to load scan history.");
    } finally {
      setLoading(false);
    }
  }, [token, page, mediaFilter]);

  useEffect(() => { fetchScans(); }, [fetchScans]);

  const deleteScan = async (id) => {
    if (!window.confirm("Delete this scan?")) return;
    try {
      await axios.delete(`${API_BASE}/api/scans/${id}`, { headers: authHeaders(token) });
      fetchScans();
    } catch {
      alert("Failed to delete scan.");
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <section className="history-section" id="history">
      <div className="section-heading">
        <span className="section-label">SCAN HISTORY</span>
        <h2>Your Analysis Records</h2>
        <p>Browse, filter, and manage all scans saved to your account.</p>
      </div>

      <div className="history-controls">
        <select
          className="history-filter"
          value={mediaFilter}
          onChange={(e) => { setMediaFilter(e.target.value); setPage(1); }}
        >
          <option value="">All media types</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
          <option value="audio">Audio</option>
        </select>
        <button className="refresh-button" type="button" onClick={fetchScans} disabled={loading}>
          <FaRedo /> Refresh
        </button>
      </div>

      {err && <p className="auth-error" style={{ marginBottom: 16 }}><FaExclamationTriangle /> {err}</p>}

      {loading ? (
        <div className="history-loading"><span className="spinner" /> Loading scans…</div>
      ) : scans.length === 0 ? (
        <div className="no-evidence">
          <FaHistory />
          <div>
            <strong>No scans found</strong>
            <p>Scans are saved automatically when you're logged in and run an analysis.</p>
          </div>
        </div>
      ) : (
        <>
          <div className="scan-list">
            {scans.map((scan) => {
              const isFake = String(scan.prediction || "").toLowerCase().includes("fake");
              const conf = Number.isFinite(Number(scan.confidence))
                ? `${Math.round(Number(scan.confidence))}%`
                : "—";
              return (
                <div key={scan.id} className="scan-row">
                  <div className={`scan-verdict-dot ${isFake ? "dot-fake" : "dot-real"}`} />
                  <div className="scan-info">
                    <strong className="scan-filename">{scan.filename || "—"}</strong>
                    <span className="scan-meta">
                      {String(scan.media_type || "").toUpperCase()} · {String(scan.prediction || "—").toUpperCase()} · {conf}
                    </span>
                    <span className="scan-date">
                      <FaClock /> {new Date(scan.analyzed_at || scan.created_at || Date.now()).toLocaleString()}
                    </span>
                  </div>
                  <button
                    className="scan-delete"
                    type="button"
                    title="Delete scan"
                    onClick={() => deleteScan(scan.id)}
                  >
                    <FaTrash />
                  </button>
                </div>
              );
            })}
          </div>

          <div className="pagination">
            <button
              className="page-btn"
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              <FaChevronLeft />
            </button>
            <span className="page-info">Page {page} of {totalPages}</span>
            <button
              className="page-btn"
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              <FaChevronRight />
            </button>
          </div>
        </>
      )}
    </section>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main App
// ──────────────────────────────────────────────────────────────────────────────
function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [showHeatmap, setShowHeatmap] = useState(true);

  // Auth state
  const [token, setToken] = useState(getStoredToken);
  const [currentUser, setCurrentUser] = useState(null);
  const [authModal, setAuthModal] = useState(null); // "login" | "register" | null

  const fileInputRef = useRef(null);
  const mediaType = getMediaType(file);

  // ── Fetch profile when token changes ─────────────────────────────────────
  useEffect(() => {
    if (!token) { setCurrentUser(null); return; }
    axios
      .get(`${API_BASE}/api/auth/me`, { headers: authHeaders(token) })
      .then((r) => setCurrentUser(r.data))
      .catch(() => {
        setToken(null);
        setStoredToken(null);
        setCurrentUser(null);
      });
  }, [token]);

  // ── API health ────────────────────────────────────────────────────────────
  useEffect(() => {
    let active = true;
    axios.get(`${API_BASE}/`, { timeout: 5000 })
      .then(() => { if (active) setApiStatus("online"); })
      .catch(() => { if (active) setApiStatus("offline"); });
    return () => { active = false; };
  }, []);

  // ── Object URL for preview ────────────────────────────────────────────────
  useEffect(() => {
    if (!file) { setPreview(null); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // ── Auto-scroll to results ────────────────────────────────────────────────
  useEffect(() => {
    if (result) {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;
    const type = getMediaType(selectedFile);
    if (!type) { setErrorMessage("Unsupported file. Choose an image, video, or audio file."); return; }
    if (selectedFile.size === 0) { setErrorMessage("The selected file is empty."); return; }
    if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setErrorMessage(`Maximum frontend file size is ${MAX_FILE_SIZE_MB} MB.`); return;
    }
    setFile(selectedFile);
    setResult(null);
    setErrorMessage("");
  };

  const handleInput = (event) => {
    handleFile(event.target.files?.[0]);
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
    if (!type) { setErrorMessage("Please select a supported media file."); return; }
    setLoading(true);
    setErrorMessage("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpointByType = { image: "/predict", video: "/predict-video", audio: "/predict-audio" };

    try {
      const response = await axios.post(
        `${API_BASE}${endpointByType[type]}`,
        formData,
        { timeout: 180000, headers: authHeaders(token) }
      );
      if (!response.data || typeof response.data.prediction !== "string") {
        throw new Error("The backend response does not contain a valid prediction.");
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
        error.message === "The backend response does not contain a valid prediction."
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
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAuthSuccess = (newToken, user) => {
    setToken(newToken);
    setStoredToken(newToken);
    setCurrentUser(user);
    setAuthModal(null);
  };

  const logout = async () => {
    if (token) {
      try { await axios.post(`${API_BASE}/api/auth/logout`, {}, { headers: authHeaders(token) }); }
      catch { /* best effort */ }
    }
    setToken(null);
    setStoredToken(null);
    setCurrentUser(null);
  };

  // ── Derived display values ────────────────────────────────────────────────
  const prediction = String(result?.prediction || "").toLowerCase();
  const isReal = prediction === "real";
  const isFake = ["fake", "deepfake", "manipulated"].some((l) => prediction.includes(l));

  const confidenceValue = Number(result?.confidence ?? result?.confidence_score);
  const confidence = Number.isFinite(confidenceValue)
    ? Math.max(0, Math.min(100, confidenceValue))
    : null;
  const confidenceLabel = confidence === null ? null : `${Number(confidence.toFixed(2))}%`;

  const verdictLabel = isReal ? "REAL" : isFake ? "DEEPFAKE" : prediction ? prediction.toUpperCase() : "UNKNOWN";
  const verdictClass = isReal ? "real" : isFake ? "fake" : "unknown";

  const totalFrames = result?.total_frames ?? result?.frames_analyzed ?? null;
  const fakeFrames = result?.fake_frames ?? null;
  const realFrames = result?.real_frames ?? null;

  const heatmap = result?.heatmap || result?.gradcam;
  const spectrogram = result?.spectrogram;
  const waveform = result?.waveform;
  const timeline = result?.timeline;

  const forensicSummary =
    result?.executive_summary ||
    `The model returned “${result?.prediction || "inconclusive"}”${confidenceLabel ? ` with ${confidenceLabel} confidence` : ""}. ${isFake
      ? "This is an automated screening result, not independent proof of manipulation."
      : isReal
        ? "This classification does not rule out subtle manipulation."
        : "The result is inconclusive or unfamiliar; further review is required."}`;

  const forensicRecommendation =
    result?.recommendation ||
    "Verify the original source, preserve the original file, and compare it with independently verified reference material.";

  const forensicObservations = Array.isArray(result?.forensic_observations) ? result.forensic_observations : [];
  const classScores = result?.class_scores || result?.prediction_scores || result?.scores;
  const classScoreEntries = classScores && typeof classScores === "object" && !Array.isArray(classScores)
    ? Object.entries(classScores)
    : [];
  const processingTime = result?.processing_time_ms ?? result?.processing_time;
  const modelName = result?.model_name || result?.model_version || "Not returned by the prediction endpoint";

  const mediaIcon =
    mediaType === "image" ? <FaImage /> : mediaType === "video" ? <FaVideo /> : <FaMusic />;

  // ── PDF Export ────────────────────────────────────────────────────────────
  const exportReport = () => {
    if (!result) return;
    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const margin = 18;
    const contentWidth = pageWidth - margin * 2;
    let y = 22;

    const addWrappedText = (text, fontSize = 10) => {
      pdf.setFontSize(fontSize);
      const lines = pdf.splitTextToSize(String(text ?? "Not provided"), contentWidth);
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
    pdf.setFont("helvetica", "bold");
    addWrappedText("ANALYSIS RESULT", 12);
    pdf.setFont("helvetica", "normal");
    addWrappedText(`Verdict: ${String(result.prediction || "Unknown").toUpperCase()}`);
    addWrappedText(`Confidence: ${confidenceLabel || "Not provided"}`);
    addWrappedText(`Media type: ${result.media_type || mediaType}`);
    addWrappedText(`Filename: ${result.filename || file?.name}`);
    addWrappedText(`File size: ${formatBytes(result.file_size)}`);
    addWrappedText(`Analysis date: ${new Date(result.analyzed_at || Date.now()).toLocaleString()}`);
    if (currentUser) addWrappedText(`Analyst: ${currentUser.username}`);
    y += 3;
    pdf.setFont("helvetica", "bold");
    addWrappedText("FORENSIC SUMMARY", 12);
    pdf.setFont("helvetica", "normal");
    addWrappedText(forensicSummary);
    y += 3;
    pdf.setFont("helvetica", "bold");
    addWrappedText("RECOMMENDATION", 12);
    pdf.setFont("helvetica", "normal");
    addWrappedText(forensicRecommendation);
    if (forensicObservations.length > 0) {
      y += 3;
      pdf.setFont("helvetica", "bold");
      addWrappedText("FORENSIC OBSERVATIONS", 12);
      pdf.setFont("helvetica", "normal");
      forensicObservations.forEach((obs, i) => {
        addWrappedText(`${i + 1}. ${obs.title || obs.category || `Observation ${i + 1}`}: ${obs.description || obs.detail || ""}`);
      });
    }
    y += 3;
    pdf.setFont("helvetica", "bold");
    addWrappedText("IMPORTANT NOTE", 12);
    pdf.setFont("helvetica", "normal");
    addWrappedText(
      "This report reflects the output of the configured AI model. A prediction or confidence score is not proof of authenticity."
    );
    pdf.save("DeepGuard-Analysis-Report.pdf");
  };

  // ══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <div className="app">
      {/* ── Navbar ───────────────────────────────────────────────────────── */}
      <nav className="navbar" aria-label="Main navigation">
        <a className="logo" href="#home" aria-label="DeepGuard AI home">
          <span className="logo-icon"><FaShieldAlt /></span>
          <span>DeepGuard AI</span>
        </a>

        <div className={`nav-links ${menuOpen ? "show-menu" : ""}`}>
          <a href="#home" onClick={() => setMenuOpen(false)}>Home</a>
          <a href="#features" onClick={() => setMenuOpen(false)}>Features</a>
          <a href="#analyze" onClick={() => setMenuOpen(false)}>Analyze</a>
          {currentUser && (
            <button
              className="nav-tab-btn"
              type="button"
              onClick={() => { setMenuOpen(false); document.getElementById("history")?.scrollIntoView({ behavior: "smooth" }); }}
            >
              <FaHistory /> History
            </button>
          )}
          <a href="#about" onClick={() => setMenuOpen(false)}>About</a>

          {currentUser ? (
            <div className="nav-user">
              <span className="nav-username"><FaUser /> {currentUser.username}</span>
              <button className="nav-button nav-logout" type="button" onClick={logout}>
                <FaSignOutAlt /> Sign Out
              </button>
            </div>
          ) : (
            <div className="nav-auth-btns">
              <button className="nav-button-outline" type="button" onClick={() => setAuthModal("login")}>
                <FaSignInAlt /> Sign In
              </button>
              <button className="nav-button" type="button" onClick={() => { setMenuOpen(false); setAuthModal("register"); }}>
                <FaUserPlus /> Register
              </button>
            </div>
          )}
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

      {/* ── Auth Modal ───────────────────────────────────────────────────── */}
      <AnimatePresence>
        {authModal && (
          <AuthModal
            key={authModal}
            mode={authModal}
            onSuccess={handleAuthSuccess}
            onClose={() => setAuthModal(null)}
          />
        )}
      </AnimatePresence>

      <main>
        {/* ── Hero ─────────────────────────────────────────────────────── */}
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
            <h1>Detect deepfakes.<span> Protect the truth.</span></h1>
            <p>Analyze images, videos, and audio using your configured DeepGuard AI models to identify potential signs of manipulated media.</p>
            <div className="hero-actions">
              <a className="primary-button" href="#analyze">Analyze Media <FaArrowRight /></a>
              <a className="secondary-button" href="#features"><FaPlay /> Explore Features</a>
            </div>
            <div className="trust-row">
              <span><FaCheckCircle /> Three media types</span>
              <span><FaCheckCircle /> Model-based results</span>
              <span><FaLock /> Local API processing</span>
            </div>
            <p className="api-status">
              API status:{" "}
              <strong>
                {apiStatus === "checking" ? "Checking…" : apiStatus === "online" ? "Connected" : "Not connected"}
              </strong>
            </p>
            {currentUser && (
              <p className="api-status">
                Signed in as: <strong>{currentUser.username}</strong>
                {" · "}
                <button type="button" className="inline-link" onClick={() => { document.getElementById("history")?.scrollIntoView({ behavior: "smooth" }); }}>
                  View history
                </button>
              </p>
            )}
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

        {/* ── Analyze Section ───────────────────────────────────────────── */}
        <section className="analyze-section" id="analyze">
          <motion.div
            className="section-heading"
            initial={{ opacity: 0, y: 22 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <span className="section-label">MEDIA ANALYSIS</span>
            <h2>Upload media for an AI authenticity check</h2>
            <p>Select an image, video, or audio recording. DeepGuard will send it to the corresponding backend endpoint.</p>
            {!currentUser && (
              <p className="auth-nudge">
                <FaLock />{" "}
                <button type="button" className="inline-link" onClick={() => setAuthModal("login")}>Sign in</button>
                {" "}to save scan history.{" "}
                <button type="button" className="inline-link" onClick={() => setAuthModal("register")}>Register</button>
                {" "}for free.
              </p>
            )}
          </motion.div>

          <div className="analysis-grid">
            <motion.div
              className={`upload-card ${dragActive ? "drag-active" : ""}`}
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
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
                <button className="drop-zone" type="button" onClick={() => fileInputRef.current?.click()}>
                  <span className="upload-icon"><FaCloudUploadAlt /></span>
                  <h3>Drop your media here</h3>
                  <p>or click to browse files from your device</p>
                  <span className="browse-button">Choose File</span>
                  <div className="supported-files">
                    <span><FaImage /> Images: JPG, PNG, WEBP</span>
                    <span><FaVideo /> Videos: MP4, MOV, AVI</span>
                    <span><FaMusic /> Audio: WAV, MP3, FLAC</span>
                  </div>
                  <p className="file-size-note">Maximum frontend file size: {MAX_FILE_SIZE_MB} MB</p>
                </button>
              ) : (
                <div className="selected-file">
                  <div className="file-type-icon">{mediaIcon}</div>
                  <div className="file-details">
                    <strong title={file.name}>{file.name}</strong>
                    <span>{formatBytes(file.size)}</span>
                    <span>{mediaType?.toUpperCase()} file selected</span>
                  </div>
                  <button className="change-file" type="button" onClick={() => fileInputRef.current?.click()}>
                    Change file
                  </button>
                </div>
              )}

              <button className="analyze-button" type="button" onClick={analyze} disabled={!file || loading}>
                {loading ? <><span className="spinner" /> Analyzing media…</> : <><FaBolt /> Analyze Media</>}
              </button>

              {file && (
                <button className="reset-button" type="button" onClick={resetUpload} disabled={loading}>
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
              <div className="preview-area">
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
                    <div><FaShieldAlt /></div>
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

        {/* ── Results ───────────────────────────────────────────────────── */}
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
                <p>This report reflects the prediction returned by your configured backend model.</p>
                {result.scan_id && (
                  <p className="scan-saved-note"><FaCheckCircle /> Scan saved to history (ID: {result.scan_id})</p>
                )}
              </div>

              <div className="result-dashboard" id="analysis-report">
                {/* Forensic Report */}
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
                      <strong>{confidenceLabel || "Not provided"}</strong>
                    </div>
                    <div className="forensic-metric">
                      <span>Frames analyzed</span>
                      <strong>{totalFrames ?? "Not applicable"}</strong>
                    </div>
                    <div className="forensic-metric">
                      <span>Analysis status</span>
                      <strong>{result.analysis_status || "Complete"}</strong>
                    </div>
                  </div>

                  {/* Section 01: Executive Summary */}
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

                  {/* Section 02: Forensic Observations */}
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
                            <span className="observation-marker"><FaExclamationTriangle /></span>
                            <div>
                              <h4>{item.title || item.category || `Observation ${index + 1}`}</h4>
                              <p><strong>Backend-reported measurement:</strong> {item.measured_value ?? item.value ?? item.description ?? item.detail ?? "Measurement unavailable."}</p>
                              {item.interpretation && <p className="observation-interpretation"><strong>Interpretation:</strong> {item.interpretation}</p>}
                              <p className="observation-limit"><strong>Limitation:</strong> {item.limitation || "This signal is descriptive and does not independently establish whether media is authentic or manipulated."}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="no-evidence">
                        <FaFileAlt />
                        <div>
                          <strong>No feature-level forensic findings returned</strong>
                          <p>The current API response provides a classification and confidence score, but no individual feature observations.</p>
                        </div>
                      </div>
                    )}
                  </section>

                  {/* Section 03: Feature Weights + Visualizations */}
                  <section className="forensic-panel">
                    <div className="forensic-section-title">
                      <span className="forensic-number">03</span>
                      <div>
                        <h3>Explainable AI &amp; Media Signals</h3>
                        <p>Model visualization and media-derived signals, when provided</p>
                      </div>
                    </div>

                    {/* Grad-CAM */}
                    {heatmap && (mediaType === "image" || mediaType === "video") && preview && (
                      <div className="explainability-block">
                        <div className="explainability-heading">
                          <div>
                            <h4>Grad-CAM visual explanation</h4>
                            <p>Highlighted regions indicate areas that influenced the model, not proof of forgery.{mediaType === "video" ? " The backend returned a representative-frame heatmap; the corresponding source frame is not separately available." : ""}</p>
                          </div>
                          <button className="toggle-button" type="button" role="switch" aria-checked={showHeatmap} onClick={() => setShowHeatmap((visible) => !visible)}>
                            {showHeatmap ? "Hide overlay" : "Show overlay"}
                          </button>
                        </div>
                        <div className={`explainability-images ${showHeatmap ? "with-overlay" : ""}`}>
                          <figure>
                            {mediaType === "image" ? <img className="explanation-image" src={preview} alt="Original uploaded image" /> : <video className="explanation-image" src={preview} controls aria-label="Original uploaded video" />}
                            <figcaption>{mediaType === "image" ? "Original media" : "Original video — not frame-matched"}</figcaption>
                          </figure>
                          {showHeatmap && <figure>
                            <img className="explanation-image" src={heatmap.startsWith("data:") ? heatmap : `data:image/png;base64,${heatmap}`} alt="Backend-generated Grad-CAM overlay" />
                            <figcaption>Grad-CAM overlay</figcaption>
                          </figure>}
                        </div>
                        {showHeatmap && <div className="heatmap-legend"><span /> Low relative activation <span className="legend-high" /> Higher relative activation <small>Relative visualization only; not a probability scale.</small></div>}
                      </div>
                    )}

                    {!heatmap && mediaType !== "audio" && (
                      <div className="no-evidence explainability-unavailable">
                        <FaChartLine />
                        <div>
                          <strong>Explainability unavailable</strong>
                          <p>The backend did not return a heatmap or equivalent visual explanation for this analysis.</p>
                        </div>
                      </div>
                    )}

                    {/* Audio Spectrogram */}
                    {spectrogram && mediaType === "audio" && (
                      <div className="explainability-block">
                        <h4>Audio Spectrogram</h4>
                        <p>This spectrogram is a backend-generated visualization of the audio input, not a model explainability heatmap.</p>
                        <img
                          className="explanation-image"
                          src={spectrogram.startsWith("data:") ? spectrogram : `data:image/png;base64,${spectrogram}`}
                          alt="Backend-generated audio spectrogram"
                        />
                      </div>
                    )}

                    {mediaType === "audio" && !heatmap && (
                      <div className="no-evidence explainability-unavailable">
                        <FaChartLine />
                        <div>
                          <strong>Explainability unavailable</strong>
                          <p>The backend returned no audio explainability map. The spectrogram, if present, visualizes input audio and is not a heatmap of model attention.</p>
                        </div>
                      </div>
                    )}

                    {/* Waveform */}
                    {waveform && mediaType === "audio" && (
                      <WaveformVisualizer waveform={waveform} />
                    )}

                    {/* Video Timeline */}
                    {timeline && mediaType === "video" && (
                      <VideoTimeline timeline={timeline} />
                    )}
                  </section>

                  {/* Section 04: Evidence Record */}
                  <section className="forensic-panel">
                    <div className="forensic-section-title">
                      <span className="forensic-number">04</span>
                      <div>
                        <h3>Evidence Record</h3>
                        <p>Metadata returned or available during this session</p>
                      </div>
                    </div>
                    <div className="evidence-table-wrap" id="technical-details">
                      <table className="evidence-table">
                        <tbody>
                          <tr><th>File name</th><td>{result.filename || file?.name || "—"}</td></tr>
                          <tr><th>Media type</th><td>{String(result.media_type || mediaType).toUpperCase()}</td></tr>
                          <tr><th>Model prediction</th><td>{result.prediction || "—"}</td></tr>
                          <tr><th>Confidence</th><td>{confidenceLabel || "Not returned"}</td></tr>
                          <tr><th>Model</th><td>{modelName}</td></tr>
                          <tr><th>Prediction scores by class</th><td>{classScoreEntries.length ? classScoreEntries.map(([label, score]) => `${label}: ${score}`).join(" · ") : "Not returned by the prediction endpoint"}</td></tr>
                          <tr><th>Processing time</th><td>{processingTime == null ? "Not returned by the prediction endpoint" : `${processingTime}${result.processing_time_ms != null ? " ms" : ""}`}</td></tr>
                          <tr><th>Total frames</th><td>{totalFrames ?? "Not returned"}</td></tr>
                          <tr><th>Fake frames</th><td>{fakeFrames ?? "Not returned"}</td></tr>
                          <tr><th>Real frames</th><td>{realFrames ?? "Not returned"}</td></tr>
                          <tr><th>File size</th><td>{formatBytes(result.file_size ?? file?.size)}</td></tr>
                          {result.duration_sec != null && (
                            <tr><th>{mediaType === "audio" ? "Audio duration" : "Media duration"}</th><td>{Number(result.duration_sec).toFixed(2)}s</td></tr>
                          )}
                          {result.fps != null && <tr><th>Source video frame rate</th><td>{result.fps}</td></tr>}
                          {result.transcription?.status && <tr><th>Transcription</th><td>{result.transcription.status}{result.transcription.message ? ` — ${result.transcription.message}` : ""}</td></tr>}
                          {Array.isArray(result.warnings) && <tr><th>Warnings</th><td>{result.warnings.length ? result.warnings.join("; ") : "None returned"}</td></tr>}
                          <tr><th>Limitations</th><td>Model predictions are probabilistic. Results can be incorrect and should not be treated as definitive evidence.</td></tr>
                          <tr><th>Analysis date</th><td>{new Date(result.analyzed_at || Date.now()).toLocaleString()}</td></tr>
                          {currentUser && <tr><th>Analyst</th><td>{currentUser.username}</td></tr>}
                          {result.scan_id && <tr><th>Scan ID</th><td>{result.scan_id}</td></tr>}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <div className="forensic-footer">
                    <div>
                      <FaLock />
                      <span>
                        <strong>Interpret with care</strong>
                        <p>AI predictions are probabilistic and can be incorrect. Preserve the original media and independently verify important findings.</p>
                      </span>
                    </div>
                    <div className="report-actions">
                      <button className="analyze-button export-button" type="button" onClick={exportReport}>
                        <FaFilePdf /> Download Report <FaDownload />
                      </button>
                      <button className="secondary-button" type="button" onClick={() => document.getElementById("media-preview")?.scrollIntoView({ behavior: "smooth", block: "center" })}>
                        <FaImage /> View Original Media
                      </button>
                      <button className="secondary-button" type="button" onClick={() => document.getElementById("technical-details")?.scrollIntoView({ behavior: "smooth", block: "center" })}>
                        <FaFileAlt /> View Analysis Details
                      </button>
                      <button className="secondary-button" type="button" onClick={() => { resetUpload(); document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth" }); }}>
                        <FaRedo /> Run Another Analysis
                      </button>
                    </div>
                  </div>
                </div>

                {/* Stats row */}
                <div className="stats">
                  {mediaType === "video" ? (
                    <>
                      <div className="stat"><div className="stat-icon blue"><FaVideo /></div><div><h3>{totalFrames ?? "—"}</h3><p>Frames Analyzed</p></div></div>
                      <div className="stat"><div className="stat-icon red"><FaExclamationTriangle /></div><div><h3>{fakeFrames ?? "—"}</h3><p>Fake Frames</p></div></div>
                      <div className="stat"><div className="stat-icon green"><FaCheckCircle /></div><div><h3>{realFrames ?? "—"}</h3><p>Real Frames</p></div></div>
                    </>
                  ) : (
                    <>
                      <div className="stat"><div className="stat-icon blue">{mediaIcon}</div><div><h3>{mediaType?.toUpperCase() || "MEDIA"}</h3><p>Media Type</p></div></div>
                      <div className="stat"><div className="stat-icon green"><FaCheckCircle /></div><div><h3>{confidenceLabel || "—"}</h3><p>Model Confidence</p></div></div>
                      <div className="stat"><div className="stat-icon blue"><FaLock /></div><div><h3>Complete</h3><p>Request Status</p></div></div>
                    </>
                  )}
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* ── Scan History Tab ─────────────────────────────────────────── */}
        {currentUser && (
          <ScanHistoryPanel token={token} />
        )}

        {/* ── Features Section ─────────────────────────────────────────── */}
        <section className="features-section" id="features">
          <div className="section-heading">
            <span className="section-label">BUILT FOR MEDIA ANALYSIS</span>
            <h2>One platform, three media types</h2>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <span className="feature-icon"><FaImage /></span>
              <h3>Image Analysis</h3>
              <p>Submit images to the configured image classification model with Grad-CAM explainability.</p>
            </div>
            <div className="feature-card">
              <span className="feature-icon"><FaVideo /></span>
              <h3>Video Analysis</h3>
              <p>Frame-by-frame pipeline with per-frame timestamps and fake-frame detection timeline.</p>
            </div>
            <div className="feature-card">
              <span className="feature-icon"><FaMusic /></span>
              <h3>Audio Analysis</h3>
              <p>Mel-spectrogram inference with waveform visualization and acoustic forensic findings.</p>
            </div>
          </div>
        </section>
      </main>

      <footer id="about">
        <div className="footer-logo"><FaShieldAlt /> DeepGuard AI</div>
        <p>AI-assisted detection of potentially manipulated media.</p>
        <p className="footer-note">For research and informational use. Always verify important findings independently.</p>
      </footer>
    </div>
  );
}

export default App;
