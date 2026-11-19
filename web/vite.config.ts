import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the interface on :5173 and forwards /api to `pruneagent web` on :8125
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8125" } },
});
