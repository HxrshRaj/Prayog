import React, { useEffect, useState } from "react";
import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Legend, Line, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { api, barRows, curveRows, fmt, VERDICT_CLASS } from "./lib.js";

const PALETTE = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0891b2", "#4b5563"];

function useApi(path) {
  const [state, set] = useState({ data: null, error: null });
  useEffect(() => {
    let live = true;
    set({ data: null, error: null });
    api(path).then((data) => live && set({ data, error: null }), (error) => live && set({ data: null, error }));
    return () => { live = false; };
  }, [path]);
  return state;
}

function useHash() {
  const [h, set] = useState(window.location.hash.slice(1) || "/");
  useEffect(() => {
    const on = () => set(window.location.hash.slice(1) || "/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return h;
}

const Verdict = ({ v }) => <span className={`pill ${VERDICT_CLASS[v] || "muted"}`}>{v}</span>;
const Loading = ({ s }) => (s.error ? <p className="err">Could not load: {String(s.error.message)}</p> : <p className="muted">Loading…</p>);

function Overview() {
  const s = useApi("/experiments");
  if (!s.data) return <Loading s={s} />;
  return (
    <>
      <h1>Prayog <small>experiment: paper claims, reproduced at small scale</small></h1>
      <p className="lede">Each card runs a paper's key claim on a laptop CPU with fixed seeds and several runs, then states
        honestly whether it held. GPU parts live in <code>gpu/</code> and are labelled by what has actually been run.</p>
      <div className="grid">
        {s.data.map((e) => (
          <a key={e.name} className="card" href={`#/card/${e.name}`}>
            <Verdict v={e.verdict} />
            <h3>{e.title}</h3>
            <p className="muted">{e.paper}</p>
            <p className="muted">{e.n_runs} runs · commit {e.git_commit || "–"}</p>
          </a>
        ))}
      </div>
    </>
  );
}

function RunsTable() {
  const s = useApi("/runs");
  const [filter, setFilter] = useState("");
  if (!s.data) return <Loading s={s} />;
  const rows = s.data.filter((r) => `${r.experiment} ${r.variant}`.includes(filter));
  const metricKeys = [...new Set(rows.flatMap((r) => Object.keys(r.metrics)))].filter((k) =>
    rows.some((r) => typeof r.metrics[k] === "number")).slice(0, 5);
  return (
    <>
      <h1>Runs <small>{rows.length} of {s.data.length}</small></h1>
      <input placeholder="filter by experiment / variant" value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="scroll">
        <table>
          <thead><tr><th>experiment</th><th>variant</th><th>seed</th>{metricKeys.map((k) => <th key={k}>{k}</th>)}<th>commit</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.experiment}</td><td>{r.variant}</td><td>{r.seed}</td>
                {metricKeys.map((k) => <td key={k}>{typeof r.metrics[k] === "number" ? r.metrics[k].toFixed(4) : ""}</td>)}
                <td><code>{r.git_commit}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function SummaryTable({ summary, metrics }) {
  const variants = Object.keys(summary || {});
  return (
    <div className="scroll">
      <table>
        <thead><tr><th>variant</th>{metrics.map((m) => <th key={m.metric}>{m.label}</th>)}</tr></thead>
        <tbody>
          {variants.map((v) => (
            <tr key={v}><td>{v}</td>{metrics.map((m) => <td key={m.metric}>{fmt(summary[v][m.metric], m.digits ?? 3)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricBars({ summary, metric, label }) {
  const rows = barRows(summary, metric);
  if (!rows.length) return null;
  return (
    <div className="chart">
      <h4>{label} <small>(mean, error bar = std over seeds)</small></h4>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="variant" angle={-25} textAnchor="end" interval={0} height={60} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="mean" fill={PALETTE[0]}><ErrorBar dataKey="std" width={4} stroke="#111" /></Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Curves({ name }) {
  const s = useApi(`/curves/${name}`);
  if (!s.data || !Object.keys(s.data).length) return null;
  const variants = Object.keys(s.data);
  const rows = curveRows(s.data);
  return (
    <div className="chart">
      <h4>Learning curves <small>(line = mean, band = ±1 std over {s.data[variants[0]].seeds} seeds)</small></h4>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} />
          <YAxis />
          <Tooltip />
          <Legend />
          {variants.map((v, i) => (
            <Area key={v + "b"} dataKey={`${v}_band`} stroke="none" fill={PALETTE[i]} fillOpacity={0.18} legendType="none" />
          ))}
          {variants.map((v, i) => <Line key={v} dataKey={v} stroke={PALETTE[i]} dot={false} strokeWidth={2} />)}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function ReproCard({ name }) {
  const s = useApi(`/experiments/${name}`);
  if (!s.data) return <Loading s={s} />;
  const { card, results } = s.data;
  if (!card) return <p>No repro card yet for {name}.</p>;
  return (
    <>
      <a href="#/">← all experiments</a>
      <h1>{card.title}</h1>
      <p className="muted">{card.paper}</p>
      <Verdict v={card.verdict} />
      <section><h2>Paper's claim</h2><p>{card.claim}</p></section>
      <section><h2>Our setup</h2><p>{card.setup}</p></section>
      <section><h2>Our result</h2><p>{card.result}</p></section>
      {card.metrics && results && <SummaryTable summary={results.summary} metrics={card.metrics} />}
      {card.metrics && results && (
        <div className="charts">{card.metrics.filter((m) => m.chart).map((m) =>
          <MetricBars key={m.metric} summary={results.summary} metric={m.metric} label={m.label} />)}</div>
      )}
      <Curves name={name} />
      <section><h2>Verdict: {card.verdict}</h2><p>{card.verdict_reason}</p></section>
      <section>
        <h2>What did not match the paper</h2>
        <ul>{(card.mismatches || []).map((m, i) => <li key={i}>{m}</li>)}</ul>
      </section>
      {results && <p className="muted">Provenance: commit <code>{results.env.git_commit}</code>, Python {results.env.python},
        {" "}{Object.entries(results.env.libraries || {}).map(([k, v]) => `${k} ${v}`).join(", ")}.</p>}
    </>
  );
}

function Compare() {
  const s = useApi("/experiments");
  const [all, setAll] = useState(null);
  useEffect(() => {
    if (!s.data) return;
    Promise.all(s.data.map((e) => api(`/experiments/${e.name}`))).then(setAll);
  }, [s.data]);
  if (!all) return <Loading s={s} />;
  const blocks = all.filter((e) => e.card?.compare && e.results);
  return (
    <>
      <h1>Compare <small>pairs the paper cares about</small></h1>
      {blocks.map((e) => (
        <section key={e.card.title}>
          <h2>{e.card.title}</h2>
          <div className="scroll"><table>
            <thead><tr><th>comparison</th><th>metric</th><th>A</th><th>B</th><th>A − B</th><th>gap vs spread</th></tr></thead>
            <tbody>
              {e.card.compare.map((c, i) => {
                const a = e.results.summary[c.a]?.[c.metric], b = e.results.summary[c.b]?.[c.metric];
                if (!a || !b) return null;
                const diff = a.mean - b.mean, spread = Math.max(a.std, b.std);
                return (
                  <tr key={i}>
                    <td>{c.label}</td><td>{c.metric}</td>
                    <td>{c.a}: {fmt(a, c.digits ?? 3)}</td><td>{c.b}: {fmt(b, c.digits ?? 3)}</td>
                    <td>{diff.toFixed(c.digits ?? 3)}</td>
                    <td>{spread > 0 && Math.abs(diff) > 2 * spread ? "gap > 2 std" : "within noise"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table></div>
        </section>
      ))}
    </>
  );
}

function TryIt() {
  const [text, setText] = useState("A warm, funny and beautifully acted film.");
  const [out, setOut] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true); setErr(null);
    try {
      setOut(await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }));
    } catch (e) { setErr(e.message); setOut(null); }
    setBusy(false);
  };
  return (
    <>
      <h1>Try it <small>LoRA-tuned sentiment classifier</small></h1>
      <p className="muted">A 4.4M-parameter bert-tiny with a rank-4 LoRA adapter, trained on a 4,000-sentence SST-2 subset.
        It is tiny: expect mistakes on subtle text.</p>
      <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} maxLength={1000} />
      <button onClick={go} disabled={busy || !text.trim()}>{busy ? "…" : "Classify"}</button>
      {err && <p className="err">{err}</p>}
      {out && (
        <div className="card">
          <h3>{out.label}</h3>
          <p>positive {(out.probabilities.positive * 100).toFixed(1)}% · negative {(out.probabilities.negative * 100).toFixed(1)}%</p>
          <p className="muted">{out.model}</p>
        </div>
      )}
    </>
  );
}

export default function App() {
  const hash = useHash();
  const m = hash.match(/^\/card\/(.+)$/);
  let page;
  if (m) page = <ReproCard name={m[1]} />;
  else if (hash === "/runs") page = <RunsTable />;
  else if (hash === "/compare") page = <Compare />;
  else if (hash === "/try") page = <TryIt />;
  else page = <Overview />;
  return (
    <>
      <nav>
        <a href="#/"><b>Prayog</b></a>
        <a href="#/">Papers</a><a href="#/runs">Runs</a><a href="#/compare">Compare</a><a href="#/try">Try it</a>
      </nav>
      <main>{page}</main>
    </>
  );
}
