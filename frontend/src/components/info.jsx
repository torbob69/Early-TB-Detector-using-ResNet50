// 1. Tambahkan import Download icon
import { X, Download } from "lucide-react";
import { useEffect, useState } from "react";

// 2. IMPORT FILE ZIP-NYA DI SINI
// Sesuaikan jumlah "../" tergantung lokasi file Info.js ini.
// Jika Info.js ada di folder components, mungkin butuh "../assets/..."
// Jika Info.js ada di root src, cukup "./assets/..."
// Tambahkan ?url di akhir path
import comparisonpdf from "../assets/metrics compare.pdf?url";

export default function Info({ onClose, fadeOut }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 10);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className={`absolute inset-0 w-screen h-screen bg-neutral-800/40 z-10 backdrop-blur-3xl flex justify-center items-center transition-all duration-1000 ${
        fadeOut || !isVisible ? "opacity-0" : "opacity-100"
      }`}
    >
      <div className="flex flex-col text-white text-md md:text-2xl text-left gap-4 w-2xs md:w-sm">
        <X
          onClick={onClose}
          className="self-end w-5 md:w-6 h-auto transition-transform duration-300 hover:rotate-180 cursor-pointer"
        ></X>
        
        {/* ... Bagian Header Model Name & Dataset ... */}
        <div>
          <h1 className="font-light text-left">Model Name</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            ResNet50: Fine-tuned for TB Detection - Robust Version
          </h3>
        </div>
        <div>
          <h1 className="font-light text-left">Dataset</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Kaggle TB Chest X-Ray Dataset -{" "}
            <span className="italic">Tawsifur Rahman</span>
          </h3>
        </div>

        {/* ... Bagian Details ... */}
        <div>
          <h1 className="font-light text-left">Details</h1>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Epoch: 5/30
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Batch size: 32
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Optimizer: Adam
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Unfreeze last 10 layers of ResNet50
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Model version: v1.2
          </h3>
          <h3 className="text-xs md:text-sm font-extralight text-neutral-300">
            Trained: 2025-12-13
          </h3>
        </div>

        {/* -------------------------------------------------- */}
        {/* 3. BAGIAN DOWNLOAD FILE BARU                       */}
        {/* -------------------------------------------------- */}
        <div>
          <a
            href={comparisonpdf} // Variable dari import di atas
            download="Metrics comparison.pdf" // Nama file saat didownload user
            className="flex items-center gap-2 text-xs md:text-sm font-extralight text-blue-400 hover:text-blue-300 transition-colors w-fit cursor-pointer"
          >
            <Download size={16} />
            Comparison Results (.pdf)
          </a>
        </div>
        {/* -------------------------------------------------- */}

        <div>
          <h1 className="font-normal text-left">DISCLAIMER</h1>
          <p className="text-xs md:text-sm font-extralight text-neutral-300">
            This AI model is intended solely for research, learning, and
            demonstration purposes. The outputs produced by this system may
            contain errors, biases, or inaccuracies, and should not be relied
            upon for critical or high-stakes decisions. This system{" "}
            <span className="font-semibold">
              does not provide professional medical, legal, or financial
              guidance.
            </span>{" "}
            The developers and contributors are not liable for any damages,
            losses, or consequences resulting from the use of this model. By
            continuing to use this application, you acknowledge and accept these
            limitations.
          </p>
        </div>
      </div>
    </div>
  );
}