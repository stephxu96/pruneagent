import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
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
        <a href="#/models">Your models</a>
      </nav>
      <main>{page}</main>
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
