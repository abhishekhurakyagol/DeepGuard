import { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");

  const [loading, setLoading] = useState(false);

  const handleImage = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));

    setPrediction("");
    setConfidence("");
  };

  const detectDeepfake = async () => {
    if (!image) {
      alert("Please choose an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);

    } catch (err) {
      alert("Prediction Failed");
      console.log(err);
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f172a",
        color: "white",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Arial",
      }}
    >
      <div
        style={{
          width: 500,
          padding: 30,
          borderRadius: 20,
          background: "#1e293b",
          textAlign: "center",
        }}
      >
        <h1>🛡️ DeepGuard AI</h1>

        <p>Detect AI Generated Images</p>

        <input
          type="file"
          accept="image/*"
          onChange={handleImage}
        />

        <br />
        <br />

        {preview && (
          <img
            src={preview}
            alt="preview"
            width="250"
            style={{
              borderRadius: 10
            }}
          />
        )}

        <br />
        <br />

        <button
          onClick={detectDeepfake}
          style={{
            padding: "12px 25px",
            cursor: "pointer",
            fontSize: 16
          }}
        >
          {loading ? "Detecting..." : "Detect Deepfake"}
        </button>

        <br />
        <br />

        {prediction && (
          <>
            <h2>
              Prediction : {prediction.toUpperCase()}
            </h2>

            <h3>
              Confidence : {confidence}%
            </h3>
          </>
        )}
      </div>
    </div>
  );
}

export default App;