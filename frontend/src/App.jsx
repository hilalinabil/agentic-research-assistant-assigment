import { useEffect, useRef, useState } from "react";
import Markdown from "./Markdown.jsx";

const EXAMPLES = [
  "What are the main contraindications for warfarin?",
  "Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?",
  "A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?",
  "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?",
];

const STATUS_LABEL = {
  success: "Reviewed: grounded",
  needs_review: "Reviewer flagged issues",
  insufficient_evidence: "Insufficient evidence",
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef(null);

  useEffect(() => () => clearInterval(timer.current), []);

  async function ask(e) {
    e?.preventDefault();
    const q = question.trim();
    if (q.length < 3 || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    setElapsed(0);
    const started = Date.now();
    timer.current = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
        throw new Error(detail || `Request failed (HTTP ${res.status})`);
      }
      setResult(body);
    } catch (err) {
      setError(err.message || "Could not reach the server.");
    } finally {
      clearInterval(timer.current);
      setLoading(false);
    }
  }

  const answerBody = result ? stripDisclaimer(result.answer, result.disclaimer) : "";

  return (
    <main className="page">
      <header>
        <h1>Pharmacology Research Assistant</h1>
        <p className="sub">
          Answers are researched by an agent over a local corpus of FDA DailyMed labels and MedlinePlus pages,
          cited to source lines, and checked by a reviewer step.
        </p>
      </header>

      <form onSubmit={ask} className="ask">
        <label htmlFor="q">Your question</label>
        <textarea
          id="q"
          rows={3}
          value={question}
          placeholder="e.g. What are the main contraindications for warfarin?"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(e);
          }}
        />
        <div className="row">
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? `Researching… ${elapsed}s` : "Ask"}
          </button>
          <span className="hint">Ctrl+Enter to send</span>
        </div>
        <div className="examples">
          <span>Examples:</span>
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex} className="chip" onClick={() => setQuestion(ex)} disabled={loading}>
              {ex.length > 60 ? ex.slice(0, 57) + "…" : ex}
            </button>
          ))}
        </div>
      </form>

      {loading && (
        <p className="loading" role="status">
          The agent is planning, searching the corpus, drafting and reviewing the answer. This usually takes 15–60 seconds.
        </p>
      )}

      {error && (
        <div className="error" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <section className="result">
          <div className={`status ${result.status}`}>{STATUS_LABEL[result.status] || result.status}</div>

          <article className="answer">
            <Markdown text={answerBody} />
          </article>

          <div className="disclaimer" role="note">
            ⚠️ {result.disclaimer}
          </div>

          <h2>Sources</h2>
          {result.citations.length === 0 ? (
            <p className="muted">No citations.</p>
          ) : (
            <ul className="citations">
              {result.citations.map((c) => (
                <li key={`${c.source}-${c.start_line}-${c.end_line}`}>
                  <code>{c.source}</code> lines {c.start_line}–{c.end_line}
                  {c.section && <span className="muted"> · {c.section}</span>}
                </li>
              ))}
            </ul>
          )}

          <details className="trace">
            <summary>How the agent worked ({result.trace?.tool_calls?.length ?? 0} tool calls, {result.trace?.elapsed_seconds}s)</summary>
            <Trace trace={result.trace} />
          </details>
        </section>
      )}
    </main>
  );
}

function Trace({ trace }) {
  if (!trace) return null;
  return (
    <div className="trace-body">
      <h3>1. Plan</h3>
      <ol>
        {trace.plan?.sub_questions?.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
      <h3>2. Tool calls (chosen by the model)</h3>
      <ol>
        {trace.tool_calls?.map((t, i) => (
          <li key={i} className={t.ok ? "" : "bad"}>
            <code>{t.tool}</code> <code className="args">{t.arguments}</code> → {t.summary}
          </li>
        ))}
      </ol>
      <h3>3. Critic reviews</h3>
      <ol>
        {trace.reviews?.map((r, i) => (
          <li key={i}>
            verdict: <strong>{r.verdict}</strong>
            {r.unsupported_claims?.length > 0 && <> · unsupported: {r.unsupported_claims.join("; ")}</>}
            {r.clinical_ambiguities?.length > 0 && <> · ambiguities: {r.clinical_ambiguities.join("; ")}</>}
            {r.overrides?.length > 0 && <> · overrides: {r.overrides.join("; ")}</>}
          </li>
        ))}
      </ol>
      {trace.revised && <p>The answer was revised once after review.</p>}
    </div>
  );
}

function stripDisclaimer(answer, disclaimer) {
  const tail = `*${disclaimer}*`;
  return answer.endsWith(tail) ? answer.slice(0, -tail.length).trim() : answer;
}
