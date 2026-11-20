import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./lib";
import { ModelsPage } from "./ModelsPage";
import { ComparePage, HomePage, NewRunPage } from "./pages";
import { RunPage } from "./RunPage";
import "./styles.css";

function useHash() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => {
      setHash(window.location.hash);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

// Long work runs in the background workflow service, apart from this app; without it nothing starts.
function ServiceNotice() {
  const [down, setDown] = useState(false);
  useEffect(() => {
    const read = () => api<{ running: boolean }>("/api/service").then((s) => setDown(!s.running)).catch(() => undefined);
    read();
    const t = setInterval(read, 10000);
    return () => clearInterval(t);
  }, []);
  if (!down) return null;
  return (
    <div className="service-notice" role="status">
      <b>The background service isn't running,</b> so nothing will start. Start it in a terminal:
      <code>uv run pruneagent worker</code>
    </div>
  );
}

function App() {
  const hash = useHash();
  const run = hash.match(/^#\/run\/([\w.-]+)(?:\/(\d+))?/);
  let page;
  if (run) page = <RunPage key={hash} id={run[1]} start={Number(run[2] ?? 0)} />;
  else if (hash.startsWith("#/compare")) page = <ComparePage />;
  else if (hash.startsWith("#/new")) page = <NewRunPage key={hash} />;
  else if (hash.startsWith("#/models")) page = <ModelsPage />;
  else page = <HomePage />;
  return (
    <>
      <nav className="top">
        <a href="#/" className="brand">✂ PruneAgent</a>
        <a href="#/compare">Comparison</a>
        <a href="#/new">New run</a>
        <a href="#/models">Select your model</a>
      </nav>
      <ServiceNotice />
      <main>{page}</main>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
