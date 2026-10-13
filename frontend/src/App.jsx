import { useEffect, useRef, useState } from "react";

const MESSAGES = {
  unknown: {
    title: "Celebrity not recognized",
    text: "This face does not match anyone in our celebrity database.",
  },
  no_face: {
    title: "No face detected",
    text: "Please upload a clear photo where the face is visible.",
  },
  multiple_faces: {
    title: "Multiple faces detected",
    text: "Please upload a photo with only one person.",
  },
  database_empty: {
    title: "Database is empty",
    text: "Build the celebrity database before running recognition.",
  },
};

const STEPS = [
  { title: "Detect", text: "Locates and aligns the face in your photo." },
  { title: "Embed", text: "ArcFace turns the face into a 512-number signature." },
  { title: "Match", text: "Compares it with every celebrity in the database." },
];

function GlassesLogo({ size = 22 }) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} fill="none"
      stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
      <circle cx="9" cy="17" r="5.5" />
      <circle cx="23" cy="17" r="5.5" />
      <path d="M14.5 16.5c1-1 2-1 3 0" />
      <path d="M3.5 15 2 11M28.5 15 30 11" />
    </svg>
  );
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
      const form = new FormData();
      form.append("file", file);
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

  const matched = result?.status === "match";

  return (
    <div className="page">
      <div className="aurora" aria-hidden="true">
        <span className="blob blob-1" />
        <span className="blob blob-2" />
        <span className="blob blob-3" />
      </div>

      <header className="navbar">
        <div className="brand">
          <span className="brand-mark"><GlassesLogo /></span>
          <span className="brand-text">
            Cogniva<span className="gradient-text">Eyewear</span>
          </span>
        </div>
        <span className="nav-pill">
          <span className="live-dot" /> AI Face Recognition
        </span>
      </header>

      <main className="container">
        <section className="hero">
          <span className="badge">Powered by InsightFace · ArcFace</span>
          <h1>
            CognivaEyewear
            <br />
            <span className="gradient-text">Celebrity Recognition</span>
          </h1>
          <p>
            Upload a photo and identify the celebrity in seconds using
            deep-learning face embeddings.
          </p>
        </section>

        <section className={`card ${matched ? "card-success" : ""}`}>
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
              <div className="drop-icon">
                <svg viewBox="0 0 24 24" width="30" height="30" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round"
                  strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="drop-title">Drop a celebrity photo here</p>
              <p className="drop-sub">or <u>browse your files</u> · JPG, PNG, WEBP</p>
            </div>
          ) : (
            <div className="workspace">
              <div className={`preview ${loading ? "is-scanning" : ""}`}>
                <img src={preview} alt="Uploaded" />
                <span className="corner tl" />
                <span className="corner tr" />
                <span className="corner bl" />
                <span className="corner br" />
                {loading && <div className="scan-line" />}
              </div>

              <div className="side">
                <div className="file-row">
                  <span className="file-icon">IMG</span>
                  <p className="file-name" title={file.name}>{file.name}</p>
                </div>

                {!result && (
                  <button className="btn primary" onClick={recognize}
                    disabled={loading}>
                    {loading ? (
                      <>
                        <span className="spinner" /> Analyzing face...
                      </>
                    ) : (
                      "Recognize Celebrity"
                    )}
                  </button>
                )}

                {!result && !loading && (
                  <p className="hint">
                    Tip: a clear, front-facing photo of one person works best.
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

          {matched && <DetailsPanel result={result} />}

          {error && <div className="error">{error}</div>}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => selectFile(e.target.files[0])}
          />
        </section>

        <section className="steps">
          {STEPS.map((step, i) => (
            <div className="step" key={step.title}>
              <span className="step-num">0{i + 1}</span>
              <h3>{step.title}</h3>
              <p>{step.text}</p>
            </div>
          ))}
        </section>
      </main>

      <footer className="footer">
        <GlassesLogo size={16} />
        © {new Date().getFullYear()} CognivaEyewear Celebrity Recognition
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
  if (result.status === "match") {
    const initials = result.name
      .split(" ")
      .map((part) => part[0])
      .slice(0, 2)
      .join("");

    return (
      <div className="result match">
        <div className="result-head">
          <span className="avatar">{initials}</span>
          <div>
            <span className="result-label">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
                stroke="currentColor" strokeWidth="3" strokeLinecap="round"
                strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Celebrity identified
            </span>
            <h2>{result.name}</h2>
          </div>
        </div>
        {(result.category || result.description) && (
          <div className="result-meta">
            {result.category && <span className="chip">{result.category}</span>}
            {result.description && <span>{result.description}</span>}
          </div>
        )}
      </div>
    );
  }

  const message = MESSAGES[result.status] || {
    title: "Something went wrong",
    text: result.message || "Please try again.",
  };

  return (
    <div className="result warn">
      <span className="result-label">Result</span>
      <h2>{message.title}</h2>
      <p className="result-text">{message.text}</p>
    </div>
  );
}
