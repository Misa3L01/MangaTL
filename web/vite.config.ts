import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a running `mangatl ui --no-browser` (port 8765).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  // React + Konva is ~560 kB: fine for a local app, no need to split it.
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1000 },
});
