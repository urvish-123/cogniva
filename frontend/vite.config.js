import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FastAPI backend: uvicorn app.main:app --reload --port 8001
// (port 8000 is used by another app on this machine)
const BACKEND_URL = "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/recognize": BACKEND_URL,
      "/health": BACKEND_URL,
    },
  },
});
