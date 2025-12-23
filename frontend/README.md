# Frontend - TB Detector UI

A modern React application allowing users to upload Chest X-rays via drag-and-drop and receive real-time AI analysis.

## ⚡ Features

- **Drag & Drop Interface:** Intuitive upload zone with visual feedback (hover states, masking effects).
- **Real-time Inference:** Connects to the FastAPI backend (`/predict`) for immediate results.
- **Smart Result Display:**
    - Dynamic classification (Normal vs Tuberculosis).
    - Confidence percentage.
    - Context-aware medical advice text.
- **Visuals:**
    - Glassmorphism effects using Backdrop Filter.

## 🛠️ Installation & Run

1. **Install dependencies:**
   ```bash
   npm install
   ```
   *Note: This project relies on `lucide-react` for icons and potentially `framer-motion` for animations.*

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## 🔌 Configuration

The application is configured to look for the backend at `http://127.0.0.1:8000`.

If you need to change this (e.g., for deployment), edit the `API_URL` constant in `src/App.jsx`:

```javascript
const API_URL = "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)";
```

## 📂 Component Structure

- **`App.jsx`**: Main logic controller. Handles file state, API calls, and rendering.
- **`components/`**:
    - `WebTitle`, `TitleDesc`: Branding and headers.
    - `warning`, `info`: Alert modals and information displays.
    - `ui/`: Reusable UI elements (buttons, inputs).
    - `animate-ui/`: Complex background animations.

## 🎨 Styling
The project uses **Tailwind CSS** for styling. Ensure your `tailwind.config.js` is set up correctly to scan the `src` folder.