import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// FastAPI backend: uvicorn app.main:app --reload
const BACKEND_URL = "http://127.0.0.1:8000";

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
