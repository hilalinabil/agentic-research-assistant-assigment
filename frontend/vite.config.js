import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development (`npm run dev`) API calls are proxied to the FastAPI server.
// In production the built files are served by FastAPI itself, so /ask is same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ask": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
