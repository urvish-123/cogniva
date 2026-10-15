import { useEffect, useRef, useState } from "react";

const MESSAGES = {
  unknown: {
    title: "Not recognized",
    text: "This image does not match any celebrity or object in our database.",
  },
  multiple_faces: {
    title: "Multiple faces detected",
    text: "Please upload a photo with only one person.",
  },
  database_empty: {
    title: "Database is empty",
    text: "Build the celebrity and object databases before running recognition.",
  },
};

function unknownMessage(result) {
  const faceStatus = result.celebrity?.status;
  if (faceStatus === "multiple_faces") return MESSAGES.multiple_faces;
  if (faceStatus === "database_empty" && result.object?.status === "database_empty") {
    return MESSAGES.database_empty;
  }
  return MESSAGES.unknown;
}

// Models never need more than this (face detector runs at 640, CLIP at 224),
// so larger photos are downscaled in the browser before upload.
const MAX_SIDE = 1280;

async function shrinkImage(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = MAX_SIDE / Math.max(bitmap.width, bitmap.height);
    if (scale >= 1) {
      bitmap.close();
      return file;
    }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; // transparent PNG areas -> white, not black
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.92)
    );
    if (!blob) return file;
    const name = file.name.replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg" });
  } catch {
    return file; // browser can't decode it; let the server validate it
  }
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(2)} s`;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  // Free the preview URL when the image changes.
  useEffect(() => {
    return () => preview && URL.revokeObjectURL(preview);
  }, [preview]);

  function selectFile(selected) {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please choose an image file (JPG, PNG or WEBP).");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }

  function reset() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function recognize() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const upload = await shrinkImage(file);
      const form = new FormData();
      form.append("file", upload, upload.name);
      const response = await fetch("/recognize", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Recognition failed.");
      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot reach the server. Is the backend running on port 8000?"
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  return (
    <div className="page">
      <header className="navbar">
        <span className="brand-text">CognivaEyewear</span>
        <span className="nav-note">Celebrity &amp; Object Recognition</span>
      </header>

      <main className="container">
        <section className="hero">
          <h1>Celebrity &amp; Object Recognition</h1>
          <p>
            Upload a photo to find out which celebrity or object it shows,
            and how long it took to recognize.
          </p>
        </section>

        <section className="card">
          {!preview ? (
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <p className="drop-title">Drop a celebrity or object photo here</p>
              <p className="drop-sub">or <u>browse your files</u> · JPG, PNG, WEBP</p>
            </div>
          ) : (
            <div className="workspace">
              <div className="preview">
                <img src={preview} alt="Uploaded" />
              </div>

              <div className="side">
                <p className="file-name" title={file.name}>{file.name}</p>

                {!result && (
                  <button className="btn primary" onClick={recognize}
                    disabled={loading}>
                    {loading ? "Recognizing..." : "Recognize"}
                  </button>
                )}

                {!result && !loading && (
                  <p className="hint">
                    Tip: use a clear photo of one person or one object.
                  </p>
                )}

                {result && <ResultCard result={result} />}

                <button className="btn ghost" onClick={reset}
                  disabled={loading}>
                  Upload another photo
                </button>
              </div>
            </div>
          )}

          {result?.type === "celebrity" && <DetailsPanel result={result.celebrity} />}

          {error && <div className="error">{error}</div>}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => selectFile(e.target.files[0])}
          />
        </section>

      </main>

      <footer className="footer">
        © {new Date().getFullYear()} CognivaEyewear Celebrity &amp; Object Recognition
      </footer>
    </div>
  );
}

// Every field from the celebrity's Excel row, except ones already shown.
function DetailsPanel({ result }) {
  const shown = new Set(["category", "description"]);
  const entries = Object.entries(result.details || {}).filter(
    ([label, value]) => value && !shown.has(label.toLowerCase())
  );
  if (!entries.length) return null;

  return (
    <div className="details">
      <h3 className="details-title">Celebrity Profile</h3>
      <dl className="details-grid">
        {entries.map(([label, value]) => (
          <div className="detail" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function ResultCard({ result }) {
  if (result.type === "celebrity" || result.type === "object") {
    const isCelebrity = result.type === "celebrity";
    const match = isCelebrity ? result.celebrity : result.object;

    return (
      <div className="result match">
        <span className="result-label">
          {isCelebrity ? "Celebrity identified" : "Object identified"}
        </span>
        <h2>{match.name}</h2>
        <div className="result-meta">
          <span className="chip">{isCelebrity ? "Celebrity" : "Object"}</span>
          {match.category && <span className="chip">{match.category}</span>}
          {match.description && <span>{match.description}</span>}
          <span className="score">
            Similarity {(match.score * 100).toFixed(1)}%
          </span>
        </div>
        <ProcessingTime time={result.processing_time} />
      </div>
    );
  }

  const message = unknownMessage(result);

  return (
    <div className="result warn">
      <span className="result-label">Result</span>
      <h2>{message.title}</h2>
      <p className="result-text">{message.text}</p>
      <ProcessingTime time={result.processing_time} />
    </div>
  );
}

function ProcessingTime({ time }) {
  if (!time) return null;
  return (
    <div className="timing">
      <span>
        Total processing time: <strong>{seconds(time.total_ms)}</strong>
      </span>
    </div>
  );
}
