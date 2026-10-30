import { useEffect, useMemo, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000";
const HISTORY_DB = "verdant-analysis-reports";
const HISTORY_STORE = "reports";

function openHistoryDb() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("This browser does not support saved analysis details."));
      return;
    }
    const request = window.indexedDB.open(HISTORY_DB, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(HISTORY_STORE)) {
        database.createObjectStore(HISTORY_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open saved analysis storage."));
  });
}

async function saveHistoryDetail(record) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).put(record);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not save analysis details."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Saving analysis details was interrupted."));
    };
  });
}

async function saveHistoryDetails(records) {
  if (!records.length) return;
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    const store = transaction.objectStore(HISTORY_STORE);
    records.forEach((record) => store.put(record));
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not migrate saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Migrating saved history was interrupted."));
    };
  });
}

async function loadHistoryDetail(id) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readonly");
    const request = transaction.objectStore(HISTORY_STORE).get(id);
    request.onsuccess = () => {
      database.close();
      resolve(request.result || null);
    };
    request.onerror = () => {
      database.close();
      reject(request.error || new Error("Could not load saved analysis details."));
    };
  });
}

async function listHistoryDetails() {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readonly");
    const request = transaction.objectStore(HISTORY_STORE).getAll();
    request.onsuccess = () => resolve(request.result || []);
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not load saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Loading saved history was interrupted."));
    };
  });
}

async function deleteHistoryDetail(id) {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).delete(id);
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not delete this saved analysis."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Deleting this saved analysis was interrupted."));
    };
  });
}

async function clearHistoryDetails() {
  const database = await openHistoryDb();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE, "readwrite");
    transaction.objectStore(HISTORY_STORE).clear();
    transaction.oncomplete = () => {
      database.close();
      resolve();
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error || new Error("Could not clear saved history."));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error || new Error("Clearing saved history was interrupted."));
    };
  });
}

function readLegacyHistory() {
  try {
    const records = JSON.parse(localStorage.getItem("verdant-analysis-history") || "[]");
    return Array.isArray(records) ? records : [];
  } catch {
    return [];
  }
}

function Icon({ name }) {
  const paths = {
    grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    plus: "M12 5v14M5 12h14",
    history: "M4 12a8 8 0 1 0 2.34-5.66L4 8.68M4 4v4.68h4.68M12 8v4l2.5 2",
    arrow: "M5 12h13M13 6l6 6-6 6",
    upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
    leaf: "M20 4C10 4 5 8 5 15c0 3 2 5 5 5 6 0 10-6 10-16ZM5 20c1-5 4-8 9-10",
    check: "m5 12 4 4L19 6",
  };
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path d={paths[name]} />
    </svg>
  );
}

function StatusDot({ tone = "neutral" }) {
  return <span className={`status-dot ${tone}`} aria-hidden="true" />;
}

function ReadinessCard({ label, value, detail, ready, icon }) {
  return (
    <article className="readiness-card">
      <div className="card-icon"><Icon name={icon} /></div>
      <div className="card-copy">
        <p className="meta-label">{label}</p>
        <strong>{value}</strong>
        <p>{detail}</p>
      </div>
      <StatusDot tone={ready ? "good" : "warn"} />
    </article>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [classes, setClasses] = useState(null);
  const [error, setError] = useState("");
  const [view, setView] = useState("overview");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [analysisState, setAnalysisState] = useState("idle");
  const [analysisError, setAnalysisError] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [historyError, setHistoryError] = useState("");
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [openingHistoryId, setOpeningHistoryId] = useState(null);
  const [deletingHistoryId, setDeletingHistoryId] = useState(null);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [historyRestoreId, setHistoryRestoreId] = useState(null);
  const [analysisMode, setAnalysisMode] = useState("leaf");
  const [history, setHistory] = useState(() => {
    return readLegacyHistory();
  });
  const fileInput = useRef(null);
  const cameraVideo = useRef(null);
  const cameraStream = useRef(null);
  const currentView = useRef(view);

  useEffect(() => {
    currentView.current = view;
  }, [view]);

  useEffect(() => {
    if (cameraActive && cameraVideo.current && cameraStream.current) {
      cameraVideo.current.srcObject = cameraStream.current;
      cameraVideo.current.play().catch((playError) => {
        setCameraError(`Could not start the camera preview: ${playError.message}`);
      });
    }
  }, [cameraActive]);

  useEffect(() => () => {
    cameraStream.current?.getTracks().forEach((track) => track.stop());
    cameraStream.current = null;
  }, []);

  useEffect(() => {
    if (view !== "analysis" && cameraStream.current) {
      cameraStream.current.getTracks().forEach((track) => track.stop());
      cameraStream.current = null;
      setCameraActive(false);
    }
  }, [view]);

  useEffect(() => {
    let active = true;
    const restoreHistoryList = async () => {
      try {
        const [savedRecords, localRecords] = await Promise.all([
          listHistoryDetails(),
          Promise.resolve().then(readLegacyHistory),
        ]);
        if (!active) return;

        const localById = new Map(localRecords.map((record) => [String(record.id), record]));
        const savedIds = new Set(savedRecords.map((record) => String(record.id)));
        const mergedRecords = savedRecords.map((record) => ({
          ...localById.get(String(record.id)),
          ...record,
        }));
        const mergedIds = new Set(mergedRecords.map((record) => String(record.id)));
        const legacyOnlyRecords = localRecords.filter((record) => !mergedIds.has(String(record.id)));
        const allRecords = [...mergedRecords, ...legacyOnlyRecords]
          .sort((left, right) => new Date(right.createdAt || 0) - new Date(left.createdAt || 0));

        await saveHistoryDetails(
          allRecords.filter((record) => !savedIds.has(String(record.id))),
        );
        if (!active) return;
        setHistory(allRecords);
        localStorage.removeItem("verdant-analysis-history");
      } catch (loadError) {
        if (active) {
          setHistoryError(`Could not load the complete saved history: ${loadError.message}`);
        }
      } finally {
        if (active) setHistoryLoaded(true);
      }
    };
    restoreHistoryList();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    let retryTimer;
    const checkBackend = async () => {
      try {
        const [healthResponse, classesResponse] = await Promise.all([
          fetch(`${API_URL}/health`).then((response) => {
            if (!response.ok) throw new Error(`Backend health check failed (${response.status}).`);
            return response.json();
          }),
          fetch(`${API_URL}/classes`).then((response) => {
            if (!response.ok) throw new Error(`Backend class check failed (${response.status}).`);
            return response.json();
          }),
        ]);
        if (!active) return;
        setHealth(healthResponse);
        setClasses(classesResponse);
        setError("");
      } catch (requestError) {
        if (active) {
          setError(`${requestError.message} Retrying automatically; make sure FastAPI is running at ${API_URL}.`);
        }
      } finally {
        if (active) retryTimer = window.setTimeout(checkBackend, 5000);
      }
    };
    checkBackend();
    return () => {
      active = false;
      window.clearTimeout(retryTimer);
    };
  }, []);

  const classifierReady = Boolean(health?.inference_ready ?? health?.classifier_loaded ?? health?.classifier_ready ?? health?.classifier_checkpoint_present);
  const segmentationReady = Boolean(health?.segmentation_loaded ?? health?.segmentation_ready ?? health?.segmentation_checkpoint_present);
  const modelsReady = classifierReady;
  const navItems = [
    { id: "overview", label: "Overview", icon: "grid" },
    { id: "analysis", label: "New analysis", icon: "plus" },
    { id: "history", label: "History", icon: "history" },
  ];

  const readinessLabel = health ? (modelsReady ? "Ready for inference" : "Model setup needed") : "Checking backend";
  const readinessTone = modelsReady ? "good" : health ? "warn" : "neutral";
  const classCount = classes?.classes?.length ?? 0;
  const classText = classes ? (classCount ? `${classCount} classes loaded` : "Class mapping not configured") : "Waiting for backend";
  const scoreLabel = analysisResult?.screening?.confidence_calibrated
    ? "temperature-scaled score (PlantVillage calibration; not field calibration)"
    : "uncalibrated score";

  const stopCamera = () => {
    cameraStream.current?.getTracks().forEach((track) => track.stop());
    cameraStream.current = null;
    setCameraActive(false);
  };

  const useImageFile = (nextFile) => {
    if (!nextFile) return;
    const supportedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!supportedTypes.includes(nextFile.type)) {
      setAnalysisState("blocked");
      setAnalysisError("Choose a JPEG, PNG, or WEBP image.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setAnalysisState("blocked");
      setAnalysisError("Image exceeds the 10 MB upload limit.");
      return;
    }
    setFile(nextFile);
    setAnalysisError("");
    setAnalysisState("idle");
    setPreview(URL.createObjectURL(nextFile));
  };

  const handleFile = (selected) => useImageFile(selected?.[0]);

  const startCamera = async () => {
    if (cameraStarting) return;
    setCameraError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access is not available in this browser or page context. Open the app on localhost or HTTPS and choose an image file instead.");
      return;
    }
    setCameraStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" } },
      });
      if (currentView.current !== "analysis") {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream.current = stream;
      setCameraActive(true);
    } catch (cameraRequestError) {
      const message = cameraRequestError.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in your browser settings, or choose an image file."
        : cameraRequestError.name === "NotFoundError"
          ? "No camera was found on this device. Choose an image file instead."
          : `Could not access the camera: ${cameraRequestError.message}`;
      setCameraError(message);
    } finally {
      setCameraStarting(false);
    }
  };

  const captureCameraImage = () => {
    const video = cameraVideo.current;
    if (!video?.videoWidth || !video.videoHeight) {
      setCameraError("The camera is still starting. Wait for the preview, then try capture again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("Could not capture the camera image. Please try again.");
        return;
      }
      const capturedFile = new File([blob], `plant-camera-${new Date().toISOString().replaceAll(":", "-")}.jpg`, { type: "image/jpeg" });
      useImageFile(capturedFile);
      setCameraError("");
      stopCamera();
    }, "image/jpeg", 0.92);
  };

  const runAnalysis = async () => {
    if (!file) return;
    if (!classifierReady) {
      setAnalysisState("blocked");
      setAnalysisError("Analysis is unavailable until a trained classifier checkpoint is installed.");
      return;
    }
    setAnalysisState("loading");
    setAnalysisError("");
    setAnalysisResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("mode", analysisMode === "whole" ? "whole_plant" : "single_leaf");
      const response = await fetch(`${API_URL}/screen-plant`, { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Analysis endpoint is unavailable.");
      setAnalysisResult(payload);
      const record = {
        id: historyRestoreId ?? (window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`),
        filename: file.name,
        predictedClass: `${payload.screening?.candidate_count || 0} candidate region${payload.screening?.candidate_count === 1 ? "" : "s"}`,
        confidence: payload.screening?.overall_prediction?.confidence ?? payload.screening?.candidate_regions?.[0]?.confidence,
        createdAt: new Date().toISOString(),
      };
      setHistoryError("");
      try {
        await saveHistoryDetail({
          ...record,
          id: record.id,
          filename: file.name,
          imageType: file.type || "image/jpeg",
          imageBlob: file.slice(0, file.size, file.type || "image/jpeg"),
          result: payload,
        });
      } catch (storageError) {
        setHistoryError(`Analysis finished, but this record could not be saved for later: ${storageError.message}`);
      }
      setHistory((previousHistory) => [record, ...previousHistory.filter((entry) => entry.id !== record.id)]);
      setHistoryRestoreId(null);
      setAnalysisState("done");
    } catch (requestError) {
      setAnalysisState("error");
      setAnalysisError(requestError.message);
    }
  };

  const removeHistoryRecord = async (record) => {
    setDeletingHistoryId(record.id);
    setHistoryError("");
    try {
      await deleteHistoryDetail(record.id);
      setHistory((previousHistory) => previousHistory.filter((entry) => entry.id !== record.id));
      setHistoryRestoreId((currentId) => currentId === record.id ? null : currentId);
      localStorage.removeItem("verdant-analysis-history");
    } catch (deleteError) {
      setHistoryError(deleteError.message);
    } finally {
      setDeletingHistoryId(null);
    }
  };

  const removeAllHistory = async () => {
    if (!window.confirm("Delete all saved analysis history from this browser? This cannot be undone.")) return;
    setClearingHistory(true);
    setHistoryError("");
    try {
      await clearHistoryDetails();
      setHistory([]);
      setHistoryRestoreId(null);
      localStorage.removeItem("verdant-analysis-history");
    } catch (clearError) {
      setHistoryError(clearError.message);
    } finally {
      setClearingHistory(false);
    }
  };

  const reopenHistoryRecord = async (record) => {
    setOpeningHistoryId(record.id);
    setHistoryError("");
    try {
      const saved = await loadHistoryDetail(record.id);
      if (!saved?.result || !saved?.imageBlob) {
        setHistoryRestoreId(record.id);
        setHistoryError(`“${record.filename}” was saved before full reports were available. Its original image and detailed result are not stored, so select that image and rerun the analysis to restore this entry.`);
        return;
      }
      const restoredFile = new File([saved.imageBlob], saved.filename, { type: saved.imageType || "image/jpeg" });
      setFile(restoredFile);
      setPreview(URL.createObjectURL(saved.imageBlob));
      setAnalysisResult(saved.result);
      setAnalysisMode(saved.result.screening?.mode === "whole_plant" ? "whole" : "leaf");
      setAnalysisError("");
      setAnalysisState("done");
      setView("analysis");
    } catch (openError) {
      setHistoryError(openError.message);
    } finally {
      setOpeningHistoryId(null);
    }
  };

  const beginHistoryRestore = (recordId) => {
    setHistoryRestoreId(recordId);
    setFile(null);
    setPreview("");
    setAnalysisResult(null);
    setAnalysisError("");
    setAnalysisState("idle");
    setHistoryError("");
    setAnalysisMode("whole");
    setView("analysis");
    window.setTimeout(() => {
      if (fileInput.current) {
        fileInput.current.value = "";
        fileInput.current.click();
      }
    }, 0);
  };

  const subtitle = useMemo(() => ({
    overview: "A measured view of plant health, model readiness, and your next field note.",
    analysis: "Screen a single leaf or a whole plant. Review only candidate crops, with model and evidence limitations made explicit.",
    history: "Completed assessments saved in this browser, with nothing fabricated.",
  }[view]), [view]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Icon name="leaf" /></div>
          <div><strong>Verdant</strong><span>field laboratory</span></div>
        </div>
        <div className="side-rule" />
        <p className="meta-label nav-heading">Workspace</p>
        <nav aria-label="Primary navigation">
          {navItems.map((item) => (
            <button key={item.id} className={`nav-item ${view === item.id ? "active" : ""}`} aria-current={view === item.id ? "page" : undefined} onClick={() => setView(item.id)}>
              <Icon name={item.icon} /><span>{item.label}</span>{view === item.id && <span className="nav-marker" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className={`readiness-badge ${readinessTone}`}>
            <StatusDot tone={readinessTone} />
            <div><span className="meta-label">System status</span><strong>{readinessLabel}</strong></div>
          </div>
          <p className="side-note">Built for careful decisions.<br />No diagnosis is confirmed here.</p>
          <span className="version">VERDANT / 0.1.0</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><span className="breadcrumb">Field laboratory <b>/</b> {view}</span><p className="mobile-title">Verdant</p></div>
          <div className="top-status"><StatusDot tone={readinessTone} /><span>{health ? "API connected" : "Connecting to API"}</span></div>
        </header>

        <div className="content-wrap">
          <div className="page-intro">
            <div><p className="eyebrow">Plant health instrument · 01</p><h1>{view === "overview" ? <>Read the leaf.<br /><em>Respect the uncertainty.</em></> : view === "analysis" ? <>A new <em>field note.</em></> : <>Analysis <em>history.</em></>}</h1><p className="intro-copy">{subtitle}</p></div>
            {view !== "analysis" && <button className="primary-button" onClick={() => setView("analysis")}>Start an analysis <Icon name="arrow" /></button>}
          </div>

          {view === "overview" && (
            <>
              <section className="hero-grid">
                <div className="field-note">
                  <div className="note-top"><span>FIELD NOTE / 001</span><span>{new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase()}</span></div>
                  <figure className="specimen-illustration">
                    <svg viewBox="0 0 220 250" role="img" aria-labelledby="specimen-title specimen-description">
                      <title id="specimen-title">Illustrative leaf specimen</title>
                      <desc id="specimen-description">A decorative green leaf with drawn veins and warm-colored markings. It is not an analysis input or model result.</desc>
                      <defs>
                        <linearGradient id="specimen-leaf" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#e0e96b" />
                          <stop offset=".48" stopColor="#9cc94b" />
                          <stop offset="1" stopColor="#3f985f" />
                        </linearGradient>
                        <linearGradient id="specimen-shadow" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0" stopColor="#f2d579" />
                          <stop offset="1" stopColor="#d9814d" />
                        </linearGradient>
                      </defs>
                      <path d="M35 206C15 150 28 82 96 30c24-18 54-22 89-18-1 48-13 91-39 127-30 41-68 64-111 67Z" fill="url(#specimen-leaf)" stroke="#d7e69a" strokeWidth="2" />
                      <path d="M34 207c41-52 82-101 149-192M62 174c-6-28-15-47-31-59m58 23c-3-31-1-56 8-83m-10 110c25-25 46-37 75-43m-99 72c-2-26 3-46 14-66m23 44c20-12 37-18 57-19" fill="none" stroke="#f1f0be" strokeLinecap="round" strokeWidth="2.4" opacity=".8" />
                      <path d="M51 128c10-4 18 2 17 10-1 7-8 12-17 9-7-3-8-14 0-19Zm39-46c8-5 17 0 17 8 0 9-9 13-17 9-7-4-7-13 0-17Zm42 53c7-4 15 0 15 7 0 8-9 12-16 7-6-4-5-11 1-14Zm11-71c6-4 14 0 14 6 0 7-8 10-14 6-5-3-5-9 0-12Z" fill="url(#specimen-shadow)" opacity=".92" />
                      <path d="M39 202c38-50 80-98 143-181" fill="none" stroke="#244b34" strokeLinecap="round" strokeWidth="1.5" opacity=".65" />
                    </svg>
                    <figcaption>Illustrative specimen<br />Not an analysis result</figcaption>
                  </figure>
                  <div className="specimen-line"><span className="specimen-dot" /><span>LIVE MODEL READINESS</span></div>
                  <strong>{readinessLabel}</strong>
                  <p>{health?.message || "Connect the FastAPI service to inspect checkpoint readiness."}</p>
                  <div className="note-stamp"><span>STATUS</span><b>{health?.status?.toUpperCase() || "—"}</b></div>
                </div>
                <div className="workflow">
                  <div className="section-heading"><span className="meta-label">The workflow</span><span className="section-index">03 steps</span></div>
                  {["Upload a clear leaf image", "Check input quality", "Review measured output"].map((step, index) => <div className="workflow-step" key={step}><span>0{index + 1}</span><p>{step}</p><Icon name="arrow" /></div>)}
                </div>
              </section>
              <section className="section-block">
                <div className="section-heading"><span className="meta-label">Instrument readiness</span><span className="section-index">Live from backend /health</span></div>
                <div className="readiness-grid">
                  <ReadinessCard label="Classifier checkpoint" value={classifierReady ? "Online" : "Not ready"} detail={classifierReady ? "Ready to classify" : "Trained checkpoint required"} ready={classifierReady} icon="leaf" />
                  <ReadinessCard label="Segmentation checkpoint" value={segmentationReady ? "Online" : "Not ready"} detail={segmentationReady ? "Visible area available" : "Checkpoint not configured"} ready={segmentationReady} icon="grid" />
                  <ReadinessCard label="Runtime" value="CPU mode" detail={health ? "Local inference path" : "Awaiting health response"} ready={Boolean(health)} icon="history" />
                </div>
              </section>
              <section className="notice-strip"><span className="notice-symbol">!</span><p><strong>Referral policy.</strong> Unsupported, low-quality, or uncertain images should be referred for human review. Automated output is an aid — not a confirmed diagnosis.</p></section>
              {error && <div className="error-banner" role="alert">Could not reach the backend: {error}. Start FastAPI on port 8000 to see live readiness.</div>}
            </>
          )}

          {view === "analysis" && (
            <section className="analysis-layout">
              <div className="upload-panel">
                <div className="section-heading"><span className="meta-label">Input specimen</span><span className="section-index">JPEG · PNG · WEBP</span></div>
                <div className="mode-switch" role="group" aria-label="Analysis mode">
                  <button className={analysisMode === "leaf" ? "active" : ""} aria-pressed={analysisMode === "leaf"} onClick={() => { setAnalysisMode("leaf"); setAnalysisResult(null); }}>Single leaf</button>
                  <button className={analysisMode === "whole" ? "active" : ""} aria-pressed={analysisMode === "whole"} onClick={() => { setAnalysisMode("whole"); setAnalysisResult(null); }}>Whole plant</button>
                </div>
                <div className="camera-controls">
                  {!cameraActive ? (
                    <button className="history-open-button" type="button" disabled={cameraStarting} onClick={startCamera}>
                      {cameraStarting ? "Requesting camera access…" : "Use webcam to capture"}
                    </button>
                  ) : (
                    <>
                      <video ref={cameraVideo} className="camera-preview" autoPlay playsInline muted aria-label="Live plant camera preview" />
                      <div className="camera-action-row">
                        <button className="primary-button" type="button" onClick={captureCameraImage}>Capture photo</button>
                        <button className="history-open-button" type="button" onClick={() => { stopCamera(); setCameraError(""); }}>Turn camera off</button>
                      </div>
                      <p className="camera-help">After capture, review the photo below and select Screen image to analyze it.</p>
                    </>
                  )}
                  {cameraError && <div className="warning-banner" role="status">{cameraError}</div>}
                </div>
                <div
                  className={`dropzone ${file ? "has-file" : ""}`}
                  role="button"
                  tabIndex={0}
                  aria-label="Choose a plant image or drop one here"
                  onClick={() => fileInput.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      fileInput.current?.click();
                    }
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files); }}
                >
                  {preview ? <img src={preview} alt="Selected plant specimen" /> : <><div className="upload-icon"><Icon name="upload" /></div><strong>Drop a leaf or plant image here</strong><p>or choose a file from your device</p></>}
                  <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onClick={(event) => event.stopPropagation()} onChange={(event) => { handleFile(event.target.files); event.target.value = ""; }} />
                </div>
                {file && <div className="file-row"><span><StatusDot tone="good" />{file.name}</span><button onClick={(event) => { event.stopPropagation(); setFile(null); setPreview(""); }}>Remove</button></div>}
                <div className="validation-copy"><span className="meta-label">{historyRestoreId ? "Restore saved history" : "Before you submit"}</span><p>{historyRestoreId ? "Choose the original image used for this older history entry, then press Screen image. This browser does not have the original upload or full result, so reanalysis is required." : analysisMode === "whole" ? "Use a sharp photo that shows the whole plant and visible leaves. Overlapping image windows are classified, and only candidate crops are shown; they are not exact lesion masks." : "Use a sharp close-up of one leaf. The leaf is screened in overlapping crops; only non-healthy candidate crops above the score threshold are shown."}</p></div>
                <button className="primary-button full" disabled={!file || analysisState === "loading" || !classifierReady} onClick={runAnalysis}>{analysisState === "loading" ? "Screening regions…" : classifierReady ? "Screen image" : "Model unavailable"} <Icon name="arrow" /></button>
                {analysisState === "blocked" && <div className="warning-banner" role="status">{analysisError}</div>}
                {analysisState === "error" && <div className="error-banner" role="alert">{analysisError}</div>}
                {historyError && <div className="warning-banner" role="status">{historyError}</div>}
                {analysisState === "done" && analysisResult?.screening && <div className="screening-results">
                  <div className="result-card"><div><span className="meta-label">{analysisResult.screening.mode === "whole_plant" ? "Whole-plant screen" : "Single-leaf screen"}</span><strong>{analysisResult.screening.candidate_count ? `${analysisResult.screening.candidate_count} candidate regions` : "No regions met the candidate threshold"}</strong></div><span className="confidence">{analysisResult.screening.regions_checked} regions checked</span></div>
                  {analysisResult.screening.review_required && <div className="warning-banner" role="status">This result is below the model’s held-out review threshold. Treat it as uncertain and seek professional confirmation. The model has not been validated as an out-of-distribution detector.</div>}
                  <button className="secondary-button report-button" type="button" onClick={() => window.print()}>Print / Save report as PDF</button>
                  {analysisResult.screening.overall_prediction && <div className="result-card"><div><span className="meta-label">Whole-leaf classifier result</span><strong>{analysisResult.screening.overall_prediction.predicted_class}</strong></div><span className="confidence">{Math.round(analysisResult.screening.overall_prediction.confidence * 100)}% {scoreLabel}</span></div>}
                  {analysisResult.quality?.is_blurry && <div className="warning-banner" role="status">Image may be blurry (quality score {analysisResult.quality.blur_score.toFixed(1)}). Treat all model results cautiously and consider retaking a sharper photo.</div>}
                  {analysisResult.screening.overlay && <figure className="overlay-figure"><img src={analysisResult.screening.overlay} alt="Original image with candidate crop regions outlined in red and numbered" /><figcaption>Original image: red boxes mark candidate crops only, not exact diseased pixels.</figcaption></figure>}
                  {analysisResult.screening.explainability && <figure className="overlay-figure explainability-figure"><img src={analysisResult.screening.explainability.image} alt={`Grad-CAM attribution for ${analysisResult.screening.explainability.class_name}; highlighted areas influence the class score and are not confirmed lesions`} /><figcaption>Grad-CAM for {analysisResult.screening.explainability.class_name}: {analysisResult.screening.explainability.explanation}</figcaption></figure>}
                  {!analysisResult.screening.candidate_count && <p className="screening-empty">No candidate crop passed the screening threshold. This does not establish that the plant or leaf is healthy.</p>}
                  {analysisResult.screening.candidate_regions.map((region, index) => <article className="region-card" key={`${region.box.left}-${region.box.top}-${region.predicted_class}`}>
                    <img src={region.image} alt={`Candidate affected-region crop ${index + 1}; healthy-only crops are not displayed`} />
                    <div className="region-copy"><span className="meta-label">Candidate region {index + 1} · x {region.box.left}–{region.box.right}, y {region.box.top}–{region.box.bottom}</span><strong>{region.predicted_class}</strong><span className="confidence">{Math.round(region.confidence * 100)}% {scoreLabel}</span>{region.needs_review && <p className="warning-banner" role="status">Below the held-out review threshold; confirm with a qualified plant-health professional.</p>}
                      <p><b>Assessment:</b> {region.explanation}</p><p><b>Evidence:</b> {region.visible_evidence}</p><p><b>Possible cause:</b> {region.possible_causes.join(" ")}</p><p><b>Severity:</b> {region.severity}. {region.potential_impact}</p>
                      <span className="meta-label">Practical next steps</span><ul>{region.immediate_actions.map((action) => <li key={action}>{action}</li>)}</ul>
                      <p><b>Biological option:</b> {region.biological_treatment}</p><p><b>Chemical option:</b> {region.chemical_treatment}</p>
                      <span className="meta-label">General institutional references · not treatment trials</span><div className="source-list">{region.sources.map((source) => <div className="source-entry" key={source.name}><a href={source.url} target="_blank" rel="noreferrer">{source.name}: {source.title}</a><span>{source.role}</span></div>)}</div>
                    </div>
                  </article>)}
                  <div className="screening-limitations">{analysisResult.screening.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}<p>{analysisResult.screening.professional_attribution}</p><p><b>Professional sources:</b> {analysisResult.screening.sources.map((source) => source.name).join("; ")}.</p></div>
                  <article className="print-report" aria-label="Printable plant screening report">
                    <header className="print-report-header"><p>VERDANT · PLANT HEALTH SCREENING</p><h1>Image screening report</h1><p>Generated: {new Date().toLocaleString()} · File: {file?.name || "Uploaded image"}</p><p>Screening mode: {analysisResult.screening.mode === "whole_plant" ? "Whole plant" : "Single leaf"}</p></header>
                    <section><h2>Screen summary</h2><p>{analysisResult.screening.candidate_count} candidate region(s) from {analysisResult.screening.regions_checked} scanned crops. The result is a model-based screening aid, not a confirmed diagnosis. Unknown/out-of-distribution detection is not validated.</p>{analysisResult.screening.overall_prediction && <p>Whole-image class: {analysisResult.screening.overall_prediction.predicted_class} ({Math.round(analysisResult.screening.overall_prediction.confidence * 100)}% {scoreLabel}).</p>}{analysisResult.screening.review_required && <p>Review required: the confidence is below the calibrated review threshold.</p>}</section>
                    <section><h2>Original image with candidate crop locations</h2><img className="print-main-image" src={analysisResult.screening.overlay || preview} alt="Original uploaded image with candidate crop locations marked" /><p>Outlined boxes show candidate image crops, not exact diseased pixels or verified lesion boundaries.</p></section>
                    {analysisResult.screening.explainability && <section><h2>Grad-CAM model attribution</h2><img className="print-main-image" src={analysisResult.screening.explainability.image} alt={`Grad-CAM attribution for ${analysisResult.screening.explainability.class_name}`} /><p>{analysisResult.screening.explainability.explanation}</p></section>}
                    <section><h2>Candidate regions</h2>{analysisResult.screening.candidate_regions.length ? analysisResult.screening.candidate_regions.map((region, index) => <article className="print-region" key={`print-${region.box.left}-${region.box.top}-${region.predicted_class}`}><h3>Region {index + 1}: {region.predicted_class}</h3><img src={region.image} alt={`Candidate crop ${index + 1}`} /><p><b>Model score ({region.confidence_calibrated ? "temperature-scaled on PlantVillage; not field calibrated" : "uncalibrated"}):</b> {Math.round(region.confidence * 100)}%</p>{region.needs_review && <p>Below the held-out review threshold; professional confirmation is recommended.</p>}<p><b>Assessment:</b> {region.explanation}</p><p><b>Evidence:</b> {region.visible_evidence}</p><p><b>Possible cause:</b> {region.possible_causes.join(" ")}</p><p><b>Severity/impact:</b> {region.severity}; {region.potential_impact}</p><p><b>Suggested safe actions:</b></p><ul>{region.immediate_actions.map((action) => <li key={`print-${action}`}>{action}</li>)}</ul><p><b>Biological treatment:</b> {region.biological_treatment}</p><p><b>Chemical treatment:</b> {region.chemical_treatment}</p></article>) : <p>No crop met the candidate threshold. This does not demonstrate that the plant is healthy.</p>}</section>
                    <section><h2>Limitations</h2><ul>{analysisResult.screening.limitations.map((limitation) => <li key={`print-${limitation}`}>{limitation}</li>)}</ul><p>{analysisResult.screening.professional_attribution}</p></section>
                    <section><h2>Institutional references</h2>{analysisResult.screening.sources.map((source) => <p key={`print-${source.name}`}><b>{source.name}.</b> <i>{source.title}.</i> {source.role} <a href={source.url}>{source.url}</a></p>)}</section>
                    <footer>This report reproduces automated output and general references. It is not a professional diagnosis or treatment prescription.</footer>
                  </article>
                </div>}
              </div>
              <aside className="analysis-aside">
                <div className="aside-callout"><span className="notice-symbol">!</span><div><span className="meta-label">Model availability</span><strong>{modelsReady ? "Inference is available" : "Inference is not available yet"}</strong><p>{modelsReady ? "The configured checkpoints can process this image." : "This workspace has no trained checkpoints. Uploads cannot produce a prediction until setup is complete."}</p></div></div>
                <div className="aside-fact"><span className="meta-label">Supported classes</span><strong>{classText}</strong><p>{classes?.supported_plant ? `Plant: ${classes.supported_plant}` : "Loaded from the trained model mapping."}</p></div>
              </aside>
            </section>
          )}

          {view === "history" && (!historyLoaded ? (
            <section className="empty-history" aria-live="polite">
              <div className="empty-mark"><Icon name="history" /></div>
              <p className="eyebrow">Loading history</p>
              <h2>Restoring your saved field notes.</h2>
            </section>
          ) : history.length ? (
            <section className="history-list">
              <div className="section-heading">
                <span className="meta-label">Completed analyses · {history.length}</span>
                <button className="history-open-button" disabled={clearingHistory || Boolean(deletingHistoryId)} onClick={removeAllHistory}>
                  {clearingHistory ? "Deleting history…" : "Delete all history"}
                </button>
              </div>
              <p className="intro-copy">Saved in this browser and available to reopen later.</p>
              {historyError && <div className="warning-banner" role="status">{historyError}</div>}
              {history.map((record) => (
                <article className="history-row" key={record.id}>
                  <div>
                    <span className="meta-label">{record.createdAt ? new Date(record.createdAt).toLocaleString() : "Date not recorded"}</span>
                    <strong>{record.predictedClass || "Saved analysis"}</strong>
                    <p>{record.filename || "Image filename not recorded"}</p>
                    <button className="history-open-button" disabled={openingHistoryId === record.id || Boolean(deletingHistoryId) || clearingHistory} onClick={() => reopenHistoryRecord(record)}>
                      {openingHistoryId === record.id ? "Opening saved analysis…" : "Open saved analysis"}
                    </button>
                    {historyRestoreId === record.id && <button className="history-open-button history-restore-button" onClick={() => beginHistoryRestore(record.id)}>Select original image and restore</button>}
                    <button className="history-open-button" disabled={Boolean(deletingHistoryId) || clearingHistory} onClick={() => removeHistoryRecord(record)}>
                      {deletingHistoryId === record.id ? "Deleting…" : "Delete this analysis"}
                    </button>
                  </div>
                  <span className="confidence">{typeof record.confidence === "number" ? `${Math.round(record.confidence * 100)}% score` : "Score not reported"}</span>
                </article>
              ))}
              <button className="secondary-button" onClick={() => setView("analysis")}>Start a new analysis <Icon name="arrow" /></button>
            </section>
          ) : (
            <section className="empty-history">
              <div className="empty-mark"><Icon name="history" /></div>
              <p className="eyebrow">No records yet</p>
              <h2>Your field notes will live here.</h2>
              <p>Every completed analysis is saved in this browser so you can reopen or delete it later.</p>
              {historyError && <div className="warning-banner" role="status">{historyError}</div>}
              <button className="secondary-button" onClick={() => setView("analysis")}>Start a new analysis <Icon name="arrow" /></button>
            </section>
          ))}
        </div>
        <footer className="footer-note">Automated image-based assessment · Not a confirmed diagnosis · Refer uncertain cases to a qualified reviewer</footer>
      </main>
    </div>
  );
}
