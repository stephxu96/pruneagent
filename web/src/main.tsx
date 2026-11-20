import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api } from "./lib";
import { ModelsPage } from "./ModelsPage";
import { ActivityPage, ComparePage, HomePage } from "./pages";
import { PrunePage } from "./PrunePage";
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

// The app shows what the PruneAgent tools answer; without them there's nothing to show or start.
function ServiceNotice() {
  const [down, setDown] = useState(false);
  useEffect(() => {
    const read = () => api<{ running: boolean }>("/api/service").then((s) => setDown(!s.running)).catch(() => setDown(true));
    read();
    const t = setInterval(read, 10000);
    return () => clearInterval(t);
  }, []);
  if (!down) return null;
  return (
    <div className="service-notice" role="status">
      <b>The PruneAgent tools aren't running,</b> so nothing can be shown or started. Start them in a terminal:
      <code>scripts/start.sh</code>
    </div>
  );
}

function App() {
  const hash = useHash();
  const run = hash.match(/^#\/run\/([\w.-]+)(?:\/(\d+))?/);
  let page;
  if (run) page = <RunPage key={hash} id={run[1]} start={Number(run[2] ?? 0)} />;
  else if (hash.startsWith("#/compare")) page = <ComparePage />;
  else if (hash.startsWith("#/new")) page = <PrunePage key={hash} />;
  else if (hash.startsWith("#/activity")) page = <ActivityPage />;
  else if (hash.startsWith("#/models")) page = <ModelsPage />;
  else page = <HomePage />;
  return (
    <>
      <nav className="top">
        <a href="#/" className="brand">✂ PruneAgent</a>
        <a href="#/">Runs</a>
        <a href="#/new">Prune a model</a>
        <a href="#/models">Models</a>
        <a href="#/compare">Compare</a>
      </nav>
      <ServiceNotice />
      <main>{page}</main>
      <footer className="foot"><a href="#/activity">Activity</a></footer>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
