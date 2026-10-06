/* Daily-Brief frontend.
 *
 * Plain framework-free JavaScript, no build step. The UI is a reading surface
 * over a small JSON API; the state that matters lives on the server. Keeping it
 * dependency-free means the app is `git clone` + run, with nothing to compile
 * and no lockfile to age.
 *
 * Rendering convention: every render_* function returns an HTML string built
 * from escaped values, so nothing the model or a feed produces can inject
 * markup.
 */
"use strict";

/* ───────────────────────────── helpers ───────────────────────────── */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const plural = (n, word, suffix = "s") => `${n} ${word}${n === 1 ? "" : suffix}`;

const state = {
  sessionId: null,
  prefs: null,
  busy: false,
  briefing: null,
  status: null,
  sources: [],
  sourceFilter: "all",
  topic: "all",
};

const SUGGESTIONS = [
  "What's happening with NVIDIA today?",
  "Latest AI news",
  "Biggest Linux stories today",
  "How are sources covering this differently?",
];

const ONBOARD_SUGGESTIONS = [
  "AI", "Linux", "gaming", "NVIDIA", "geopolitics",
  "US politics", "India", "technology", "space", "cybersecurity",
];

const LEAN_EXPLAIN = {
  "left": "Framing classified as left of centre",
  "center-left": "Framing classified as slightly left of centre",
  "center": "Framing classified as near the centre",
  "center-right": "Framing classified as slightly right of centre",
  "right": "Framing classified as right of centre",
  "state-affiliated": "Published by state-funded media",
  "not-applicable": "No political framing detected — a left/right axis does not apply here",
  "unknown": "Not classified",
};

/* Order along the scale, for the coverage spread strip. */
const LEAN_SCALE = ["left", "center-left", "center", "center-right", "right", "state-affiliated"];
const LEAN_SIDE = {
  "left": "left", "center-left": "left",
  "center": "center",
  "center-right": "right", "right": "right",
  "state-affiliated": "state-affiliated",
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function relTime(iso) {
  if (!iso) return "no date";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "no date";
  const mins = Math.round((Date.now() - then.getTime()) / 60000);
  // Scheduler rows carry *next* run times, so this has to read both directions.
  if (mins <= -1) {
    const ahead = -mins;
    if (ahead < 60) return `in ${ahead}m`;
    const hoursAhead = Math.round(ahead / 60);
    return hoursAhead < 24 ? `in ${hoursAhead}h` : `in ${Math.round(hoursAhead / 24)}d`;
  }
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days < 8 ? `${days}d ago` : then.toLocaleDateString();
}

/* A timestamp that keeps itself honest: the ticker below rewrites every
   [data-rel] on an interval, so "2m ago" does not quietly become an hour old. */
function timeTag(iso, className = "") {
  if (!iso) return "";
  return `<time class="${className}" datetime="${esc(iso)}" data-rel="${esc(iso)}">${
    esc(relTime(iso))}</time>`;
}

function tickRelativeTimes() {
  $$("[data-rel]").forEach((el) => {
    const next = relTime(el.dataset.rel);
    if (el.textContent !== next) el.textContent = next;
  });
}

function slugify(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "topic";
}

function setBusy(on) {
  document.body.classList.toggle("is-busy", Boolean(on));
}

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { el.hidden = true; }, 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* non-JSON error page */ }
  if (!response.ok) {
    const detail = (data && (data.detail || data.error)) || text || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

/* Minimal markdown: paragraphs, bullets, bold/italic, and [n] citations.
   Everything is escaped first, so model output can never inject markup. */
function renderAnswer(text, sourceCount) {
  const safe = esc(text);
  const blocks = safe.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) return "";
    const isList = lines.every((l) => /^[-*•]\s+/.test(l));
    const isNumbered = lines.length > 1 && lines.every((l) => /^\d+[.)]\s+/.test(l));
    if (isList) {
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
    }
    if (isNumbered) {
      return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
    }
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  });
  return blocks.join("");

  function inline(s) {
    return s
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)!?]|$)/g, "$1<em>$2</em>")
      .replace(/\[(\d{1,2})\]/g, (match, n) => {
        const num = parseInt(n, 10);
        if (!(num >= 1 && num <= sourceCount)) return "";
        return `<a class="cite" href="#src-${num}" data-cite="${num}" title="Jump to source ${num}">${num}</a>`;
      });
  }
}

/* ───────────────────────── framing presentation ───────────────────────── */

/* Render a framing label from a `lean_badge` object.
 *
 * `basis` matters: "article" means this specific piece was analysed, "source"
 * means we are falling back to the publication's prior. The UI must never blur
 * those together, so a source-prior badge is dashed, says "prior" in words, and
 * explains itself in its tooltip. `showNA` forces a visible n/a where the
 * layout expects a column (coverage rows) rather than an inline tag. */
function leanPill(badge, { showNA = false } = {}) {
  if (!badge) return "";
  const lean = badge.lean || "unknown";
  const basis = badge.basis || "article";

  if (lean === "unknown" || lean === "not-applicable") {
    if (!showNA) return "";
    const why = lean === "not-applicable"
      ? "No political framing detected in this article — a left/right axis does not apply here."
      : "Not classified.";
    return `<span class="lean lean-${esc(lean)}" title="${esc(why)}">n/a</span>`;
  }

  const conf = badge.confidence ? ` · ${badge.confidence} confidence` : "";
  const origin = basis === "source"
    ? " Publication-level prior — this article was not individually classified."
    : " Classified from this article's own text.";
  const title = `${LEAN_EXPLAIN[lean] || lean}${conf}.${origin}`
    + " Analytical classification by Daily-Brief, not a fact.";

  const mark = basis === "source" ? `<span class="lean-mark">prior</span>` : "";
  return `<span class="lean lean-${esc(lean)}${basis === "source" ? " lean-prior" : ""}" `
    + `title="${esc(title)}">${esc(lean.replace(/-/g, " "))}${mark}</span>`;
}

/* The proportional strip above a coverage list: where these outlets sit on the
   scale. Only drawn when there is an actual spread to show. */
function spreadStrip(entries) {
  const counts = new Map();
  let classified = 0;
  let unclassified = 0;
  for (const entry of entries) {
    const lean = entry.lean_badge?.lean || "unknown";
    if (!LEAN_SCALE.includes(lean)) { unclassified += 1; continue; }
    counts.set(lean, (counts.get(lean) || 0) + 1);
    classified += 1;
  }
  if (counts.size < 2) return "";

  const segments = LEAN_SCALE.filter((lean) => counts.has(lean)).map((lean) => {
    const share = (counts.get(lean) / classified) * 100;
    return `<span class="spread-seg lean-${esc(lean)}" style="width:${share.toFixed(1)}%"
             title="${esc(counts.get(lean))} × ${esc(lean.replace(/-/g, " "))}"></span>`;
  }).join("");

  const sides = new Map();
  for (const [lean, n] of counts) {
    const side = LEAN_SIDE[lean] || lean;
    sides.set(side, (sides.get(side) || 0) + n);
  }
  const legend = ["left", "center", "right", "state-affiliated"]
    .filter((side) => sides.has(side))
    .map((side) => `${sides.get(side)} ${side.replace(/-/g, " ")}`)
    .concat(unclassified ? [`${unclassified} unclassified`] : [])
    .join(" · ");

  return `<div class="spread-bar" role="img"
            aria-label="Framing spread across outlets: ${esc(legend)}">${segments}</div>
          <div class="coverage-legend">${esc(legend)}</div>`;
}

/* The coverage list for a story: every outlet running it, with its own label. */
function renderCoverage(entries, totalSources, compareKey) {
  const list = entries || [];
  if (list.length < 2) return "";
  const count = totalSources || list.length;
  const shown = list.length;

  const rows = list.map((c) => {
    const name = esc(c.source);
    const outlet = c.url
      ? `<a class="coverage-outlet" href="${esc(c.url)}" target="_blank"
            rel="noopener noreferrer" title="${esc(c.title || "")}">${name}${
            c.is_lead ? `<span class="coverage-lead-mark">lead</span>` : ""}</a>`
      : `<span class="coverage-outlet">${name}</span>`;
    return `<li class="coverage-row">
        ${outlet}
        <span class="coverage-headline">${esc(c.title || "")}</span>
        ${leanPill(c.lean_badge, { showNA: true })}
      </li>`;
  }).join("");

  return `<details class="why coverage-details">
      <summary>Covered by ${plural(count, "source")}</summary>
      <div class="why-body coverage-body">
        <div class="coverage-head">
          <span class="eyebrow">Coverage · ${plural(count, "source")}</span>
          ${count > shown ? `<span class="coverage-note">showing ${shown}</span>` : ""}
        </div>
        ${spreadStrip(list)}
        <ul class="coverage-list">${rows}</ul>
        <div class="coverage-foot">
          <span class="coverage-note">Labels are per-article classifications unless marked
            <em>prior</em>.</span>
          <button type="button" class="coverage-compare" data-compare="${esc(compareKey || "")}">
            Compare how these sources frame it <span class="arrow" aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    </details>`;
}

/* The "why this classification?" panel: transparent methodology, not a debug
   dump. Fields are labelled and ordered; the source prior is pulled out of the
   raw signal list so it never reads as evidence from the article itself. */
function biasDetails(bias) {
  if (!bias || (!bias.signals?.length && !bias.rationale)) return "";

  const allSignals = bias.signals || [];
  const priorSignal = allSignals.find((s) => /^source prior/i.test(s));
  const signals = allSignals.filter((s) => !/^source prior/i.test(s));

  const fields = [];

  const articleBadge = leanPill(
    { lean: bias.lean, basis: "article", confidence: bias.confidence },
    { showNA: true });
  fields.push(`<div class="evidence-field">
      <span class="evidence-key">This article</span>
      <div class="evidence-value">${articleBadge}
        <span class="muted">${esc(bias.confidence || "")} confidence${
          bias.method ? ` · ${esc(bias.method)}` : ""}</span></div>
    </div>`);

  if (bias.source_lean) {
    const detail = priorSignal ? priorSignal.replace(/^source prior:\s*/i, "") : "";
    fields.push(`<div class="evidence-field">
        <span class="evidence-key">Source prior</span>
        <div class="evidence-value">
          ${leanPill({ lean: bias.source_lean, basis: "source" }, { showNA: true })}
          ${detail ? `<span class="muted">${esc(detail)}</span>` : ""}
        </div>
      </div>`);
  }

  if ((bias.framing || []).length) {
    fields.push(`<div class="evidence-field">
        <span class="evidence-key">Markers</span>
        <div class="evidence-value">${
          bias.framing.map((f) => `<span class="tag">${esc(f)}</span>`).join("")}</div>
      </div>`);
  }

  fields.push(`<div class="evidence-field">
      <span class="evidence-key">Signals found</span>
      <ul class="evidence-list">${
        (signals.length ? signals : ["No further signals in this article's text."])
          .map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
    </div>`);

  const subjectivity = Math.round((bias.subjectivity ?? 0) * 100);
  fields.push(`<div class="evidence-field">
      <span class="evidence-key">Subjectivity</span>
      <div class="evidence-value">
        <span class="meter${subjectivity >= 40 ? " hot" : ""}"><span
          style="width:${subjectivity}%"></span></span>
        <span class="meter-value">${subjectivity}/100</span>
      </div>
    </div>`);

  return `<details class="why">
      <summary>Why this classification?</summary>
      <div class="why-body evidence">
        <div class="evidence-title">Why this classification?</div>
        ${bias.rationale ? `<p class="evidence-lede">${esc(bias.rationale)}</p>` : ""}
        ${fields.join("")}
        <p class="evidence-disclaimer">${esc(bias.disclaimer ||
          "Analytical classification produced by Daily-Brief, not a statement of fact.")}</p>
      </div>
    </details>`;
}

/* ───────────────────────────── onboarding ───────────────────────────── */

function showOnboarding() {
  const chips = $("#onboard-chips");
  chips.innerHTML = ONBOARD_SUGGESTIONS
    .map((s) => `<button type="button" class="chip">${esc(s)}</button>`).join("");
  chips.onclick = (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    const input = $("#onboard-input");
    const parts = input.value.split(",").map((p) => p.trim()).filter(Boolean);
    if (!parts.some((p) => p.toLowerCase() === chip.textContent.toLowerCase())) {
      parts.push(chip.textContent);
    }
    input.value = parts.join(", ");
    input.focus();
  };
  $("#onboarding").hidden = false;
  $("#onboard-input").focus();
}

async function submitOnboarding(event) {
  event.preventDefault();
  const input = $("#onboard-input");
  const error = $("#onboard-error");
  const button = $("#onboard-submit");
  const value = input.value.trim();

  error.hidden = true;
  if (!value) {
    error.textContent = "Tell me at least one thing you'd like to follow.";
    error.hidden = false;
    input.focus();
    return;
  }

  button.disabled = true;
  button.textContent = "Setting up…";
  try {
    state.prefs = await api("/api/preferences", {
      method: "POST",
      body: JSON.stringify({ interests: value }),
    });
    $("#onboarding").hidden = true;
    toast(`Following ${state.prefs.interests.length} topics. Building your first briefing…`);
    await loadBriefing({ refresh: true });
    $("#ask-input").focus();
  } catch (err) {
    error.textContent = String(err.message || err);
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = `Continue <span aria-hidden="true">→</span>`;
  }
}

/* ───────────────────────────── search / chat ───────────────────────────── */

function renderSources(sources) {
  if (!sources?.length) return "";
  const cards = sources.map((s) => {
    const bias = s.bias;
    const tags = [];
    if (bias?.is_opinion) tags.push(`<span class="tag opinion">opinion</span>`);
    if (s.source_count > 1) tags.push(`<span class="tag">${s.source_count} outlets</span>`);
    return `
      <article class="source-card" id="src-${s.n}">
        <div class="source-num">${String(s.n).padStart(2, "0")}</div>
        <div class="source-main">
          <h4 class="source-title">
            <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>
          </h4>
          <div class="meta-row">
            <span class="meta-outlet">${esc(s.source)}</span>
            <span class="dot">·</span>
            ${timeTag(s.published_at)}
            ${leanPill(s.lean_badge)}
            ${tags.join("")}
          </div>
          ${s.summary ? `<p class="source-summary">${esc(s.summary)}</p>` : ""}
          <div class="disclosures">
            ${renderCoverage(s.coverage, s.source_count, s.title)}
            ${biasDetails(bias)}
          </div>
        </div>
      </article>`;
  }).join("");

  return `<div class="sources">
      <div class="sources-head">
        <span class="eyebrow">Sources</span>
        <span class="coverage-note num">${plural(sources.length, "article")
          } used to build this answer</span>
      </div>
      <div class="source-list-grid">${cards}</div>
    </div>`;
}

function renderComparison(cmp) {
  if (!cmp) return "";
  const buckets = (cmp.buckets || []).map((b) => `
    <div class="cov-bucket">
      <div class="cov-bucket-head lean-${esc(b.lean)}">${
        esc(b.lean.replace(/-/g, " "))} · ${b.articles.length}</div>
      ${b.articles.map((a) => `
        <div class="cov-item lean-${esc(b.lean)}">
          <div><a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer">${esc(a.title)}</a></div>
          <div class="cov-src">${esc(a.source)}${a.is_opinion ? " · opinion" : ""}${
            (a.framing || []).length ? " · " + a.framing.slice(0, 2).map(esc).join(", ") : ""}</div>
        </div>`).join("")}
    </div>`).join("");

  const terms = [];
  if ((cmp.distinct_left_terms || []).length) {
    terms.push(`<div>Only in left-classified headlines: ${
      cmp.distinct_left_terms.map((t) => `<code>${esc(t)}</code>`).join(" ")}</div>`);
  }
  if ((cmp.distinct_right_terms || []).length) {
    terms.push(`<div>Only in right-classified headlines: ${
      cmp.distinct_right_terms.map((t) => `<code>${esc(t)}</code>`).join(" ")}</div>`);
  }

  return `<div class="comparison">
      <h4>Measured coverage spread</h4>
      <p class="muted">${plural(cmp.article_count, "article")} from ${
        plural(cmp.source_count, "outlet")}${
        cmp.opinion_count ? `, ${plural(cmp.opinion_count, "opinion piece")}` : ""}. ${
        esc(cmp.disclaimer || "")}</p>
      ${buckets}
      ${terms.length ? `<div class="cov-terms">${terms.join("")}</div>` : ""}
    </div>`;
}

function appendTurn(question) {
  // Research mode: the composer steps back so the answer owns the page.
  document.body.classList.add("has-chat");
  autoGrow($("#ask-input"));   // the inline height from autoGrow outlives the class change
  const log = $("#chat-log");
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="turn-head">
      <span class="eyebrow">Search result</span>
      ${timeTag(new Date().toISOString(), "turn-time")}
    </div>
    <h3 class="turn-q">${esc(question)}</h3>
    <div class="turn-a">
      <div class="thinking">
        <span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>
        <span>Retrieving articles…</span>
      </div>
    </div>`;
  log.appendChild(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "start" });
  return $(".turn-a", turn);
}

function renderResult(container, result) {
  const badges = [
    result.has_results
      ? `<span class="badge grounded">grounded in ${plural(result.sources.length, "source")}</span>`
      : `<span class="badge no-results">no matching articles</span>`,
    `<span class="badge">${esc(result.engine)}</span>`,
  ];
  if (result.intent && result.intent !== "search") {
    badges.push(`<span class="badge">${esc(result.intent.replace(/_/g, " "))}</span>`);
  }

  const warnings = (result.warnings || []).length
    ? `<div class="warnings"><ul>${
        result.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`
    : "";

  const empty = result.has_results ? "" : `
    <div class="empty inline">
      <p class="empty-title">No relevant coverage found.</p>
      <p class="empty-hint">Try a broader topic or different wording — or refresh your
        feeds if this is breaking right now.</p>
    </div>`;

  container.innerHTML = `
    <div class="answer">${renderAnswer(result.answer, result.sources.length)}</div>
    ${warnings}
    <div class="answer-foot">${badges.join("")}</div>
    ${empty}
    ${renderComparison(result.comparison)}
    ${renderSources(result.sources)}`;

  container.onclick = (event) => {
    const cite = event.target.closest(".cite");
    if (!cite) return;
    event.preventDefault();
    const card = $(`#src-${cite.dataset.cite}`, container);
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1300);
  };
}

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  setBusy(true);
  $("#ask-submit").disabled = true;
  $("#chat-log").setAttribute("aria-busy", "true");
  const container = appendTurn(question);
  $("#clear-chat").hidden = false;

  try {
    const result = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query: question, session_id: state.sessionId }),
    });
    state.sessionId = result.session_id;
    sessionStorage.setItem("db-session", result.session_id);
    renderResult(container, result);
  } catch (err) {
    container.innerHTML = `<div class="warnings">Search failed: ${esc(err.message || err)}</div>`;
    toast("Search failed", true);
  } finally {
    state.busy = false;
    setBusy(false);
    $("#ask-submit").disabled = false;
    $("#chat-log").setAttribute("aria-busy", "false");
  }
}

/* ───────────────────────────── briefing ───────────────────────────── */

function renderStory(story) {
  const bias = story.bias;
  const tags = [];
  if (bias?.is_opinion) tags.push(`<span class="tag opinion">opinion</span>`);
  if (story.source_count > 1) {
    tags.push(`<span class="tag">${story.source_count} outlets</span>`);
  }

  const coverage = renderCoverage(story.coverage, story.source_count, story.title);
  const evidence = biasDetails(bias);

  return `
    <article class="story">
      <h4 class="story-title">
        <a href="${esc(story.url)}" target="_blank" rel="noopener noreferrer">${esc(story.title)}</a>
      </h4>
      <div class="meta-row">
        <span class="meta-outlet">${esc(story.source)}</span>
        <span class="dot">·</span>
        ${timeTag(story.published_at)}
        ${leanPill(story.lean_badge)}
        ${tags.join("")}
      </div>
      ${story.summary ? `<p class="story-summary">${esc(story.summary)}</p>` : ""}
      ${coverage || evidence ? `<div class="disclosures">${coverage}${evidence}</div>` : ""}
    </article>`;
}

/* Topic tabs are navigation over what is already rendered: the briefing is
   generated server-side once, and filtering never re-queries. */
function renderTopicNav(sections, interests = []) {
  const nav = $("#topic-nav");
  if (!sections.length) { nav.hidden = true; nav.innerHTML = ""; return; }
  const total = sections.reduce((n, s) => n + s.stories.length, 0);
  const covered = new Set(sections.map((s) => s.title));

  // Interests with nothing today still get a tab, disabled and showing 0 —
  // "we follow this, there was nothing" is information, not an omission.
  const tabs = [{ title: "All", slug: "all", n: total }]
    .concat(sections.map((s) => ({ title: s.title, slug: slugify(s.title), n: s.stories.length })))
    .concat(interests.filter((i) => !covered.has(i))
      .map((i) => ({ title: i, slug: slugify(i), n: 0, empty: true })));

  nav.innerHTML = tabs.map((t) => `
    <button type="button" class="topic-tab" data-topic="${esc(t.slug)}"
            aria-pressed="${t.slug === state.topic}"${t.empty
              ? ` disabled title="Nothing matched ${esc(t.title)} in this briefing"` : ""}>
      ${esc(t.title)}<span class="count">${t.n}</span>
    </button>`).join("");
  nav.hidden = false;
  updateTopicNavAffordance();
}

/* Two columns, packed rather than paired.
 *
 * A plain grid places sections row by row, so a one-story section sitting
 * beside a five-story one leaves a column of dead space. Splitting the sections
 * into two explicit columns, each taking whichever is currently shorter, packs
 * them continuously — and keeps an expanded disclosure's reflow inside its own
 * column instead of shoving the whole row down. Narrow viewports keep the flat
 * DOM order, since there is only one column to read.
 */
const WIDE = window.matchMedia("(min-width: 1200px)");

function packSections(sections) {
  const columns = [[], []];
  const weight = [0, 0];
  for (const section of sections) {
    const target = weight[0] <= weight[1] ? 0 : 1;
    columns[target].push(section);
    weight[target] += 2 + section.stories.length;   // heading + its stories
  }
  return columns;
}

function sectionHTML(section) {
  return `
    <section class="brief-section" data-topic="${esc(slugify(section.title))}">
      <div class="brief-section-head">
        <h3>${esc(section.title)}</h3>
        <span class="count">${section.stories.length === 1
          ? "1 story" : `${section.stories.length} stories`}</span>
      </div>
      <div class="story-list">${section.stories.map(renderStory).join("")}</div>
    </section>`;
}

function briefGridHTML(sections) {
  if (!WIDE.matches) {
    return `<div class="brief-grid">${sections.map(sectionHTML).join("")}</div>`;
  }
  return `<div class="brief-grid is-packed">${
    packSections(sections)
      .map((column) => `<div class="brief-col">${column.map(sectionHTML).join("")}</div>`)
      .join("")}</div>`;
}

/* Fade the right edge only while the tab strip has more to reveal. */
function updateTopicNavAffordance() {
  const nav = $("#topic-nav");
  if (!nav || nav.hidden) return;
  const remaining = nav.scrollWidth - nav.clientWidth - nav.scrollLeft;
  nav.classList.toggle("is-scrollable", remaining > 4);
}

function applyTopicFilter() {
  const grid = $(".brief-grid");
  if (!grid) return;
  const topic = state.topic || "all";
  $$(".brief-section", grid).forEach((section) => {
    section.hidden = topic !== "all" && section.dataset.topic !== topic;
  });
  grid.classList.toggle("is-filtered", topic !== "all");
  $$("#topic-nav .topic-tab").forEach((tab) =>
    tab.setAttribute("aria-pressed", String(tab.dataset.topic === topic)));
}

function briefSkeleton(sections = 4) {
  const one = `
    <div class="skeleton-section" aria-hidden="true">
      <div class="skeleton s-head"></div>
      <div class="skeleton s-title"></div><div class="skeleton s-line"></div>
      <div class="skeleton s-line short"></div>
      <div class="skeleton s-title"></div><div class="skeleton s-line"></div>
      <div class="skeleton s-line short"></div>
    </div>`;
  return `<div class="brief-grid">${one.repeat(sections)}</div>`;
}

function renderBriefing(payload) {
  state.briefing = payload;
  const body = $("#brief-body");
  const meta = $("#brief-meta");

  meta.innerHTML = payload.generated_at
    ? `Updated ${timeTag(payload.generated_at)} · ${
        payload.story_count === 1 ? "1 story" : `${payload.story_count} stories`
        } · ${esc(payload.engine)}`
    : "";

  if (!payload.sections?.length) {
    $("#topic-nav").hidden = true;
    body.innerHTML = `
      <div class="empty">
        <p class="empty-title">Your briefing isn't ready yet.</p>
        <p class="empty-hint">${esc(payload.empty_reason
          || "No stories matched your interests in the current window.")}</p>
        <div class="empty-actions">
          <button class="btn btn-primary btn-sm" data-action="refresh">Refresh feeds</button>
          <button class="btn btn-ghost btn-sm" data-action="settings">Edit interests</button>
        </div>
      </div>`;
    body.setAttribute("aria-busy", "false");
    return;
  }

  // A topic that vanished between generations should not leave an empty page.
  const slugs = payload.sections.map((s) => slugify(s.title));
  if (state.topic !== "all" && !slugs.includes(state.topic)) state.topic = "all";

  renderTopicNav(payload.sections, payload.interests || []);

  body.innerHTML = briefGridHTML(payload.sections);

  applyTopicFilter();
  body.setAttribute("aria-busy", "false");
}

async function loadBriefing({ refresh = false } = {}) {
  const button = $("#regen-btn");
  button.disabled = true;
  button.classList.toggle("is-busy", refresh);
  if (refresh) {
    setBusy(true);
    $("#brief-body").setAttribute("aria-busy", "true");
    $("#brief-body").innerHTML = briefSkeleton(state.briefing?.sections?.length || 4);
  }
  try {
    const payload = refresh
      ? await api("/api/briefing/generate", { method: "POST" })
      : await api("/api/briefing");
    renderBriefing(payload);
  } catch (err) {
    $("#brief-body").innerHTML = `
      <div class="empty">
        <p class="empty-title">Could not load the briefing.</p>
        <p class="empty-hint">${esc(err.message || err)}</p>
      </div>`;
    $("#brief-body").setAttribute("aria-busy", "false");
  } finally {
    button.disabled = false;
    button.classList.remove("is-busy");
    if (refresh) setBusy(false);
  }
}

async function refreshFeeds() {
  const button = $("#refresh-btn");
  if (button.classList.contains("is-busy")) return;
  button.classList.add("is-busy");
  button.disabled = true;
  setBusy(true);
  setHeaderBusy("fetching feeds…");
  try {
    const report = await api("/api/refresh", { method: "POST", body: "{}" });
    toast(report.summary);
    setHeaderBusy("rebuilding your briefing…");
    await loadBriefing({ refresh: true });
  } catch (err) {
    toast(`Refresh failed: ${err.message || err}`, true);
  } finally {
    button.classList.remove("is-busy");
    button.disabled = false;
    setBusy(false);
    await loadStatus();
  }
}

/* ───────────────────────────── status & settings ───────────────────────────── */

/* The header carries reading context only: how much news there is, how broad
 * it is, and how fresh. Which model is loaded is an implementation detail and
 * lives in Settings › System. */
async function loadStatus() {
  try {
    const status = await api("/api/status");
    const a = status.articles;
    state.status = status;
    $("#topbar-status").innerHTML = `
      <span><b class="num">${a.total.toLocaleString()}</b> articles</span>
      <span class="status-sep">·</span>
      <span class="status-sources"><b class="num">${a.sources_with_articles}</b> sources</span>
      <span class="status-sep status-sources">·</span>
      <span>${a.last_ingest ? `updated ${timeTag(a.last_ingest)}` : "no feeds fetched yet"}</span>`;
    return status;
  } catch {
    $("#topbar-status").textContent = "";
    return null;
  }
}

function setHeaderBusy(message) {
  const el = $("#topbar-status");
  if (message) el.innerHTML = `<span class="status-live">${esc(message)}</span>`;
}

async function openSettings() {
  $("#settings").hidden = false;
  const input = $("#settings-interests");
  input.value = state.prefs?.raw_interests_text || (state.prefs?.interests || []).join(", ");
  $("#settings-error").hidden = true;
  renderParsedChips(state.prefs?.parsed || []);
  input.oninput = previewInterests;
  input.focus();

  const status = await loadStatus();
  if (status) {
    $("#settings-status").innerHTML = `
      <dt>articles</dt><dd>${status.articles.total.toLocaleString()} stored · ${
        status.articles.last_24h} in the last 24h</dd>
      <dt>sources</dt><dd>${status.sources.enabled} of ${status.sources.total} enabled${
        status.sources.failing.length ? ` · ${status.sources.failing.length} failing` : ""}</dd>
      <dt>model</dt><dd>${status.llm.provider === "none"
        ? "<span class='s-off'>off</span> — answers and summaries use the extractive engine"
        : `${esc(status.llm.provider)}${status.llm.model ? " · " + esc(status.llm.model) : ""} — ${
          status.llm.available
            ? "<span class='s-ok'>reachable</span>"
            : "<span class='s-off'>unreachable, using the extractive engine</span>"}`}</dd>
      <dt>briefing</dt><dd>${status.briefing.story_count} stories · ${
        status.briefing.generated_at ? relTime(status.briefing.generated_at) : "never generated"}</dd>
      <dt>scheduler</dt><dd>${
        (status.scheduler || [])
          .map((j) => `${esc(j.id)} → ${j.next_run ? relTime(j.next_run) : "—"}`)
          .join("<br>") || "disabled"}</dd>`;
  }

  try {
    const data = await api("/api/sources");
    state.sources = data.sources;
    renderSourceList();
  } catch { /* sources panel is optional */ }
}

/* Settings source list, filtered by the search box and the All/Enabled/Disabled
 * segmented control. 75 sources is too many to scroll blindly. */
function renderSourceList() {
  const all = state.sources || [];
  const term = ($("#source-search").value || "").trim().toLowerCase();
  const mode = state.sourceFilter || "all";

  const rows = all.filter((s) => {
    if (mode === "enabled" && !s.enabled) return false;
    if (mode === "disabled" && s.enabled) return false;
    if (!term) return true;
    return [s.name, s.id, s.lean, s.source_type, s.country, ...(s.categories || [])]
      .filter(Boolean).join(" ").toLowerCase().includes(term);
  });

  $("#sources-count").textContent =
    rows.length === all.length ? `${all.length}` : `${rows.length} of ${all.length}`;

  if (!rows.length) {
    $("#settings-sources").innerHTML =
      `<div class="source-empty">No sources match “${esc(term)}”.</div>`;
    return;
  }

  $("#settings-sources").innerHTML = rows.map((s) => {
    const cls = !s.enabled ? "s-off" : s.consecutive_failures ? "s-bad" : "s-ok";
    const label = !s.enabled ? "disabled"
      : s.consecutive_failures ? `${s.consecutive_failures} fails`
      : s.last_success_at ? relTime(s.last_success_at) : "not fetched";
    return `<div class="source-row">
        <span class="s-name" title="${esc(s.url)}">${esc(s.name)}</span>
        ${leanPill({ lean: s.lean, basis: "source", confidence: s.lean_confidence },
                   { showNA: true })}
        <span class="s-state ${cls}">${esc(label)}</span>
      </div>`;
  }).join("");
}

/* Show how the interests will be read while the user is still typing, using
   the same parser that saving does. Only the latest keystroke's answer lands. */
let previewTimer = null;
function previewInterests() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const text = $("#settings-interests").value.trim();
    if (!text) { renderParsedChips([]); return; }
    try {
      const data = await api("/api/preferences/preview", {
        method: "POST", body: JSON.stringify({ interests: text }),
      });
      if ($("#settings-interests").value.trim() === text) renderParsedChips(data.parsed);
    } catch { /* keep the last preview */ }
  }, 250);
}

function renderParsedChips(parsed) {
  $("#settings-parsed").innerHTML = (parsed || [])
    .map((p) => `<span class="chip static">${esc(p.label)}</span>`).join("");
}

async function saveSettings() {
  const value = $("#settings-interests").value.trim();
  const error = $("#settings-error");
  const button = $("#settings-save");
  error.hidden = true;
  if (!value) {
    error.textContent = "Enter at least one interest, or use Reset personalization.";
    error.hidden = false;
    return;
  }
  button.disabled = true;
  button.textContent = "Saving…";
  try {
    state.prefs = await api("/api/preferences", {
      method: "POST", body: JSON.stringify({ interests: value }),
    });
    renderParsedChips(state.prefs.parsed);
    toast(`Saved ${state.prefs.interests.length} interests. Rebuilding briefing…`);
    await loadBriefing({ refresh: true });
  } catch (err) {
    error.textContent = String(err.message || err);
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Save changes";
  }
}

async function resetPersonalization() {
  if (!confirm("Reset personalization? Your interests will be cleared and onboarding will run again.")) {
    return;
  }
  try {
    state.prefs = await api("/api/preferences/reset", { method: "POST" });
    $("#settings").hidden = true;
    $("#brief-body").innerHTML = "";
    $("#topic-nav").hidden = true;
    $("#onboard-input").value = "";
    showOnboarding();
  } catch (err) {
    toast(`Reset failed: ${err.message || err}`, true);
  }
}

/* ───────────────────────────── wiring ───────────────────────────── */

/* Grow with the typed text — but hand the height back to CSS when empty, so the
   idle/research-mode size change is a stylesheet decision rather than whatever
   scrollHeight happened to be mid-transition. */
function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = el.value ? `${Math.min(el.scrollHeight, 200)}px` : "";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("db-theme", theme);
}

function init() {
  setTheme(localStorage.getItem("db-theme") || "dark");
  state.sessionId = sessionStorage.getItem("db-session");
  $("#brief-body").innerHTML = briefSkeleton();

  $("#suggestion-chips").innerHTML = SUGGESTIONS
    .map((s) => `<button type="button" class="chip">${esc(s)}</button>`).join("");
  $("#suggestion-chips").onclick = (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    ask(chip.textContent.trim());
  };

  const askInput = $("#ask-input");
  autoGrow(askInput);
  askInput.addEventListener("input", () => autoGrow(askInput));
  askInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#ask-form").requestSubmit();
    }
  });

  $("#ask-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = askInput.value.trim();
    if (!question) return;
    askInput.value = "";
    autoGrow(askInput);
    ask(question);
  });

  $("#onboard-form").addEventListener("submit", submitOnboarding);
  $("#onboard-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      $("#onboard-form").requestSubmit();
    }
  });

  $("#clear-chat").addEventListener("click", async () => {
    if (state.sessionId) {
      try { await api(`/api/search/history?session_id=${encodeURIComponent(state.sessionId)}`,
                      { method: "DELETE" }); } catch { /* best effort */ }
    }
    state.sessionId = null;
    sessionStorage.removeItem("db-session");
    document.body.classList.remove("has-chat");
    autoGrow(askInput);
    $("#chat-log").innerHTML = "";
    $("#clear-chat").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
    askInput.focus();
  });

  $("#topic-nav").addEventListener("scroll", updateTopicNavAffordance, { passive: true });
  window.addEventListener("resize", updateTopicNavAffordance);

  // Topic tabs filter what is already on screen; no round trip.
  $("#topic-nav").addEventListener("click", (event) => {
    const tab = event.target.closest(".topic-tab");
    if (!tab) return;
    state.topic = tab.dataset.topic;
    applyTopicFilter();
  });

  // "Compare coverage" routes into the existing compare_coverage search path
  // rather than inventing a second comparison feature.
  document.addEventListener("click", (event) => {
    const compare = event.target.closest(".coverage-compare");
    if (compare) {
      event.preventDefault();
      const headline = compare.dataset.compare || "";
      if (!headline) return;
      window.scrollTo({ top: 0, behavior: "smooth" });
      ask(`How are different sources covering this story: "${headline}"?`);
      return;
    }
    const action = event.target.closest("[data-action]");
    if (!action) return;
    if (action.dataset.action === "refresh") refreshFeeds();
    if (action.dataset.action === "settings") openSettings();
  });

  $("#source-search").addEventListener("input", renderSourceList);
  $$("#settings .seg button").forEach((button) => {
    button.addEventListener("click", () => {
      state.sourceFilter = button.dataset.filter;
      $$("#settings .seg button").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === button)));
      renderSourceList();
    });
  });

  $("#refresh-btn").addEventListener("click", refreshFeeds);
  $("#regen-btn").addEventListener("click", () => loadBriefing({ refresh: true }));
  $("#settings-btn").addEventListener("click", openSettings);
  $("#settings-close").addEventListener("click", () => { $("#settings").hidden = true; });
  $("#settings-save").addEventListener("click", saveSettings);
  $("#settings-reset").addEventListener("click", resetPersonalization);
  $("#theme-btn").addEventListener("click", () =>
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  $("#brand").addEventListener("click", (event) => {
    event.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
    askInput.focus();
  });

  $("#settings").addEventListener("click", (event) => {
    if (event.target.id === "settings") $("#settings").hidden = true;
  });

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(event.target.tagName);
    if (event.key === "Escape") {
      if (!$("#settings").hidden) { $("#settings").hidden = true; return; }
      if (typing) event.target.blur();
      return;
    }
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "/") { event.preventDefault(); askInput.focus(); }
    else if (event.key === ",") { event.preventDefault(); openSettings(); }
    else if (event.key === "r") { event.preventDefault(); refreshFeeds(); }
    else if (event.key === "g") { event.preventDefault(); loadBriefing({ refresh: true }); }
    else if (event.key === "t") {
      event.preventDefault();
      setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    }
  });

  // Only on an actual breakpoint crossing: re-rendering resets open disclosures.
  WIDE.addEventListener("change", () => {
    if (state.briefing?.sections?.length) renderBriefing(state.briefing);
  });

  setInterval(tickRelativeTimes, 60000);
  bootstrap();
}

async function bootstrap() {
  loadStatus();
  try {
    state.prefs = await api("/api/preferences");
  } catch (err) {
    toast(`Could not reach the server: ${err.message || err}`, true);
    return;
  }
  if (!state.prefs.onboarded) {
    showOnboarding();
    $("#brief-body").innerHTML = `
      <div class="empty">
        <p class="empty-title">Tell Daily-Brief what you care about.</p>
        <p class="empty-hint">Your first briefing is built as soon as you pick a few interests.</p>
      </div>`;
    $("#brief-body").setAttribute("aria-busy", "false");
    return;
  }
  await loadBriefing();
  $("#ask-input").focus();
}

document.addEventListener("DOMContentLoaded", init);
