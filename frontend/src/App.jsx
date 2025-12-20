import React, { useState } from "react";
import {
  Upload,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Sparkles,
  X,
  Trash2,
} from "lucide-react";
import WebTitle from "./components/WebTitle";
import TitleDesc from "./components/TitleDesc";
import WarningDesc from "./components/warning";
import Info from "./components/info";
import { ShinyButton } from "./components/ui/shiny-button";
import { StarsBackground } from "./components/animate-ui/components/backgrounds/stars";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (
      droppedFile &&
      (droppedFile.type === "image/jpeg" || droppedFile.type === "image/png")
    ) {
      processFile(droppedFile);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        setResult(data);
        setError(null);
      }
    } catch (apiError) {
      console.error("API Error:", apiError);
      setError(
        "Failed to connect to AI server. Ensure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const [infoActive, setInfoActive] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  const closeInfo = () => {
    setFadeOut(true);
    setTimeout(() => {
      setInfoActive(false);
      setFadeOut(false);
    }, 1000);
  };

  return (
    <StarsBackground className="bg-[radial-gradient(circle,_rgba(0,0,0,1),_rgba(23,23,23,0))] relative w-screen h-screen flex justify-center items-center flex-col md:flex-row lg:flex-row">
      {infoActive && <Info fadeOut={fadeOut} onClose={closeInfo} />}

      {/* Left Section - Branding & Info */}
      <div className="flex-1 flex flex-col justify-center lg:justify-center md:items-end lg:items-end md:pr-20">
        <div className=" lg:w-md">
          <div className="flex flex-col items-center md:items-start lg:items-start">
            <WebTitle></WebTitle>
            <TitleDesc></TitleDesc>
          </div>

          {/* Result Display */}
          {result && (
            <div className="mb-6 flex flex-col">
              <div className="flex flex-col gap-y-1 justify-center items-center md:items-start lg:items-start md:gap-y-2 lg:gap-y-3">
                <h3 className="text-xl lg:text-2xl font-normal text-neutral-300">
                  {result.prediction}
                </h3>
                <div className="flex items-center gap-2 text-xs lg:text-sm">
                  <span className="text-neutral-300 text-neutral-300">
                    confidence:{" "}
                  </span>
                  <span className="text-neutral-300">{result.confidence}</span>
                </div>
              </div>

              <p className="text-xs w-64 md:text-[8px] lg:w-auto lg:text-sm font-light leading-relaxed md:text-left lg:text-left text-neutral-300">
                {result.is_tb ? (
                  <>
                    Medical attention required. Consult a healthcare
                    professional for proper diagnosis.
                  </>
                ) : (
                  <>
                    No TB indicators detected. Continue regular health
                    check-ups.
                  </>
                )}
              </p>
            </div>
          )}

          {error && (
            <div className="rounded-2xl backdrop-blur-xl border bg-red-500/10 border-red-500/20 p-6 mb-6">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center flex-shrink-0">
                  <XCircle className="w-5 h-5 text-red-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-light text-red-300 mb-1">
                    Analysis Failed
                  </h3>
                  <p className="text-red-200/80 text-sm font-light">{error}</p>
                </div>
              </div>
            </div>
          )}

          <WarningDesc onInfoClick={() => setInfoActive(true)}></WarningDesc>
        </div>
      </div>

      {/* Right Section - Upload Area */}
      <div className="flex-1 flex flex-col justify-start lg:justify-center md:items-start lg:items-start">
        <div className="lg:w-md">
          {/* Upload Zone */}
          {!preview && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`w-2xs h-72 lg:w-md lg:h-md relative rounded-4xl transition-all duration-500 backdrop-blur-3xl [mask-image:radial-gradient(circle,black_25%,transparent_70%)] lg:[mask-image:radial-gradient(circle,black_25%,transparent_55%)] ${
                isDragging ? "bg-neutral-400/10 scale-[0.99]" : "bg-transparent"
              }`}
              id="upload-wrapper"
            >
              <input
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />

              <div className="py-16 flex flex-col items-center justify-center text-center">
                {/* logo upload */}
                <div className="w-16 h-16 rounded-full bg-white/5 backdrop-blur-xl border border-white/10 flex items-center justify-center mb-6">
                  <Upload className="w-8 h-8 text-neutral-400" />
                </div>
                <h3 className="text-xl font-light text-white mb-2">
                  Drop X-Ray Here
                </h3>
                <p className="text-neutral-300 text-sm font-light">
                  or click to browse
                </p>
                <div className="mt-6 text-xs text-neutral-400">
                  PNG, JPG • Max 10MB
                </div>
              </div>
            </div>
          )}

          {/* Preview with Actions */}
          {preview && (
            <div className="space-y-4 relative flex flex-col">
              <div className="w-16 h-16 rounded-full bg-black/25 backdrop-blur-xl border border-white/10 flex items-center justify-center mb-6 absolute self-center top-2/7 z-9 hover:scale-[1.1] hover:bg-white/5 transition-all duration-300 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                <Trash2
                  onClick={handleReset}
                  disabled={loading}
                  className="w-8 h-8 text-neutral-400"
                />
              </div>
              <div
                className="w-3xs h-56 lg:w-md lg:h-md rounded-3xl overflow-hidden backdrop-blur-xl [mask-image:linear-gradient(to_bottom,transparent,rgba(0,0,0,0.6)_30%,rgba(0,0,0,0.6)_70%,transparent),linear-gradient(to_right,transparent,rgba(0,0,0,0.6)_30%,rgba(0,0,0,0.6)_70%,transparent)]
    [mask-composite:intersect]
    [-webkit-mask-image:linear-gradient(to_bottom,transparent,rgba(0,0,0,0.6)_30%,rgba(0,0,0,0.6)_70%,transparent),linear-gradient(to_right,transparent,rgba(0,0,0,0.6)_30%,rgba(0,0,0,0.6)_70%,transparent)]
    [-webkit-mask-composite:source-in]"
                id="image-result-wrapper"
              >
                <img
                  src={preview}
                  alt="X-Ray Preview"
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="flex gap-3 justify-center">
                {!result && !error && (
                  <>
                    <ShinyButton
                      onClick={handleUpload}
                      disabled={loading}
                      className="bg-black/0 text-neutral-200 px-8 py-4 rounded-4xl text-md font-semibold cursor-pointer transition-all duration-300 ease-in-out shadow-[0px_0px_20px_#262626] hover:shadow-[0px_0px_30px_#121212] hover:bg-neutral-900 hover:scale-[1.1]"
                    >
                      {loading ? (
                        <>
                          <span>Analyzing</span>
                        </>
                      ) : (
                        <>
                          <span>Analyze</span>
                        </>
                      )}
                    </ShinyButton>
                  </>
                )}

                {(result || error) && (
                  <ShinyButton
                    onClick={handleReset}
                    className="bg-black/0 text-neutral-200 px-8 py-4 rounded-4xl text-md font-semibold cursor-pointer transition-all duration-300 ease-in-out shadow-[0px_0px_20px_#262626] hover:shadow-[0px_0px_30px_#121212] hover:bg-neutral-900 hover:scale-[1.1]"
                  >
                    Continue
                  </ShinyButton>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </StarsBackground>
  );
}

export default App;
