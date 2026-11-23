import React, { useState, useRef, useEffect } from 'react';
import {
  Camera,
  Upload,
  X,
  Loader2,
  CheckCircle2,
  Cpu,
  Sparkles,
  AlertTriangle,
  ScanLine,
} from 'lucide-react';

export default function CameraScannerModal({
  isOpen,
  onClose,
  apiBaseUrl,
  scanMode = 'question',
  onApplyScan,
}) {
  const [sourceTab, setSourceTab] = useState('upload'); // 'camera' | 'upload'
  const [engine, setEngine] = useState('auto');
  const [enhanceHandwriting, setEnhanceHandwriting] = useState(true);
  const [streamActive, setStreamActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [capturedBase64, setCapturedBase64] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState(null);

  const [ocrResult, setOcrResult] = useState(null);
  const [editedProblem, setEditedProblem] = useState('');
  const [editedSteps, setEditedSteps] = useState('');

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setStreamActive(false);
  };

  useEffect(() => {
    if (!isOpen) {
      stopCamera();
    }
    return () => stopCamera();
  }, [isOpen]);

  const startCamera = async () => {
    setScanError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setScanError('Browser camera API unavailable on this connection. Upload a photo or use an attached Pi/USB camera.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStreamActive(true);
    } catch (err) {
      setScanError(`Could not access browser camera: ${err.message}`);
    }
  };

  const runOcrOnBase64 = async (b64DataUrl) => {
    setIsScanning(true);
    setScanError(null);
    setOcrResult(null);
    try {
      const res = await fetch(`${apiBaseUrl}/scan/ocr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: b64DataUrl,
          scan_mode: scanMode,
          engine,
          enhance_handwriting: enhanceHandwriting,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `OCR returned HTTP ${res.status}`);
      }
      const data = await res.json();
      setOcrResult(data);
      setEditedProblem(data.problem_text || '');
      setEditedSteps((data.detected_steps || []).join('\n'));
    } catch (err) {
      setScanError(err.message || 'OCR scan failed.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleCaptureBrowserFrame = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/png');
    setPreviewUrl(dataUrl);
    setCapturedBase64(dataUrl);
    stopCamera();
    runOcrOnBase64(dataUrl);
  };

  const handleHardwareCapture = async () => {
    setIsScanning(true);
    setScanError(null);
    setOcrResult(null);
    try {
      const params = new URLSearchParams({ scan_mode: scanMode, engine });
      const res = await fetch(`${apiBase64UrlClean(apiBaseUrl)}/scan/hardware-capture?${params.toString()}`, {
        method: 'POST',
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Hardware camera returned HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.captured_image_base64) {
        setPreviewUrl(data.captured_image_base64);
        setCapturedBase64(data.captured_image_base64);
      }
      setOcrResult(data);
      setEditedProblem(data.problem_text || '');
      setEditedSteps((data.detected_steps || []).join('\n'));
    } catch (err) {
      setScanError(err.message || 'Hardware camera capture unavailable.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleFileSelected = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || '');
      setPreviewUrl(dataUrl);
      setCapturedBase64(dataUrl);
      runOcrOnBase64(dataUrl);
    };
    reader.readAsDataURL(file);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-6 space-y-5">
        {/* Modal Header */}
        <div className="flex items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
              <ScanLine className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">
                {scanMode === 'question_and_answer'
                  ? 'Scan Handwritten Question & Derivation Sheet'
                  : 'Scan Math Question via Camera or Photo'}
              </h3>
              <p className="text-xs text-slate-400">
                Pretrained neural OCR extracts math notation cleanly; Lean 4 &amp; AST checks verify every answer before showing it.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close scanner modal"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Engine & Source Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => {
                setSourceTab('upload');
                stopCamera();
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 ${
                sourceTab === 'upload' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Upload className="w-3.5 h-3.5" /> Upload Image
            </button>
            <button
              type="button"
              onClick={() => {
                setSourceTab('camera');
                startCamera();
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 ${
                sourceTab === 'camera' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Camera className="w-3.5 h-3.5" /> Live Camera
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={enhanceHandwriting}
                onChange={(e) => setEnhanceHandwriting(e.target.checked)}
                className="rounded border-slate-700 bg-slate-950 text-emerald-500"
              />
              <span>Enhance handwriting ink</span>
            </label>

            <select
              value={engine}
              onChange={(e) => setEngine(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200"
            >
              <option value="auto">Auto (RapidOCR Neural)</option>
              <option value="rapidocr">RapidOCR PP-OCRv4</option>
              <option value="tesseract">Tesseract 5 LSTM</option>
              <option value="trocr">TrOCR Handwritten</option>
              <option value="pix2tex">Pix2Tex LaTeX-OCR</option>
              <option value="gemini_vision">Gemini Vision (Cloud)</option>
            </select>
          </div>
        </div>

        {/* Camera / Upload Viewport */}
        {sourceTab === 'camera' ? (
          <div className="space-y-3">
            <div className="relative bg-slate-950 border border-slate-800 rounded-xl overflow-hidden min-h-[220px] flex items-center justify-center">
              <video ref={videoRef} autoPlay playsInline muted className="w-full max-h-[300px] object-contain" />
              {!streamActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-xs text-slate-400 p-4 text-center gap-2">
                  <Camera className="w-8 h-8 text-slate-600" />
                  <span>Camera preview inactive. Click Start Camera or Capture from Pi/USB Hardware Camera.</span>
                </div>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={streamActive ? handleCaptureBrowserFrame : startCamera}
                className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5"
              >
                <Camera className="w-4 h-4" />
                {streamActive ? 'Capture & Scan Photo' : 'Start Browser Camera'}
              </button>
              <button
                type="button"
                onClick={handleHardwareCapture}
                disabled={isScanning}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium flex items-center gap-1.5"
              >
                <Cpu className="w-4 h-4 text-emerald-400" />
                Use Pi / USB Hardware Camera
              </button>
            </div>
          </div>
        ) : (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFileSelected(e.dataTransfer.files?.[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-emerald-500/60 bg-slate-950/70 rounded-xl p-6 text-center cursor-pointer transition-colors space-y-2"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileSelected(e.target.files?.[0])}
            />
            {previewUrl ? (
              <img src={previewUrl} alt="Scanned math preview" className="max-h-48 mx-auto rounded-lg border border-slate-800" />
            ) : (
              <>
                <Upload className="w-8 h-8 text-emerald-400 mx-auto" />
                <p className="text-sm text-slate-200 font-medium">Drop a photo of handwritten or printed math here</p>
                <p className="text-xs text-slate-500">Supports PNG, JPG, WEBP — or click to browse</p>
              </>
            )}
          </div>
        )}

        {capturedBase64 && (
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => runOcrOnBase64(capturedBase64)}
              disabled={isScanning}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Re-scan with selected OCR model
            </button>
          </div>
        )}

        {isScanning && (
          <div className="flex items-center gap-2 text-xs font-mono text-emerald-300 bg-emerald-950/30 border border-emerald-500/30 rounded-xl p-3">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Running math handwriting &amp; OCR recognition...</span>
          </div>
        )}

        {scanError && (
          <div className="flex items-start gap-2 text-xs font-mono text-rose-300 bg-rose-950/40 border border-rose-500/40 rounded-xl p-3">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <span>{scanError}</span>
          </div>
        )}

        {ocrResult && (
          <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-semibold text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" /> Recognized Math Text (Editable Before Verification)
              </span>
              <span className="font-mono text-slate-400">
                engine: {ocrResult.engine_used} • conf: {Math.round((ocrResult.confidence || 0) * 100)}% • {ocrResult.elapsed_ms} ms
              </span>
            </div>

            <div>
              <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Recognized Problem
              </label>
              <input
                type="text"
                value={editedProblem}
                onChange={(e) => setEditedProblem(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-100"
              />
            </div>

            {scanMode !== 'question' && (
              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                  Recognized Handwritten / Solution Steps (one per line)
                </label>
                <textarea
                  rows={4}
                  value={editedSteps}
                  onChange={(e) => setEditedSteps(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm font-mono text-slate-100"
                />
              </div>
            )}

            <div className="flex flex-wrap justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => {
                  onApplyScan({
                    problemText: editedProblem.trim(),
                    stepsText: editedSteps.trim(),
                    autoSubmit: false,
                  });
                  onClose();
                }}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold"
              >
                Insert Into Editor
              </button>
              <button
                type="button"
                onClick={() => {
                  onApplyScan({
                    problemText: editedProblem.trim(),
                    stepsText: editedSteps.trim(),
                    autoSubmit: true,
                  });
                  onClose();
                }}
                className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5"
              >
                <CheckCircle2 className="w-4 h-4" />
                {scanMode === 'question_and_answer' ? 'Grade Handwritten Steps' : 'Insert & Verify in Lean 4'}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function apiBase64UrlClean(u) {
  return String(u || '').replace(/\/$/, '');
}
