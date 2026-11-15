/* TrialGuide web demo. Replays states exported by trialguide.demo.export_web; nothing is decided in the browser.
   A state is found by the path of answers (e.g. "ys/nd"); each stored state keeps only the rules that changed
   from the state before it, so we apply the changes from the root down the path. */
"use strict";

const ICONS = {
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  note: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  rank: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  check: '<path d="M4 12l5 5L20 6"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4 6l1 1 2-2M4 12l1 1 2-2M4 18l1 1 2-2"/>',
  split: '<path d="M6 3v6a6 6 0 0 0 6 6h0a6 6 0 0 1 6 6M18 3v6"/><circle cx="6" cy="3" r="1"/><circle cx="18" cy="3" r="1"/>',
  ask: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 21l1.9-5.4A8 8 0 1 1 21 12z"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6M12 16.5h.01"/>',
  steth: '<path d="M6 3v6a4 4 0 0 0 8 0V3"/><path d="M10 13v3a5 5 0 0 0 10 0v-2"/><circle cx="20" cy="12" r="2"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
};
const icon = (n, s = 16, w = 2) =>
  `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[n] || ""}</svg>`;

const COL = { stated: "#16a34a", inferred: "#2563eb", answered: "#7c3aed", against: "#dc2626", open: "#a3afc2" };
const CLASS_LABEL = { stated: "Stated in the note", inferred: "Inferred, passed risk control", answered: "Settled by your answer",
  against: "Rules the trial out", open: "Open" };
const STATUS = { met: "met", not_met: "not met", not_enough_info: "not enough information", not_applicable: "not applicable" };
const DECISION = { use_stated: "Used as stated", accept_inference: "Inference accepted", ask: "Could be asked", abstain: "Abstained" };
const GOLD = { 2: ["Eligible", "ok"], 1: ["Excluded", "info"], 0: ["Not relevant", "none"] };
const CERT = { s: ["sure", "Sure"], p: ["probably", "Probably"], u: ["unsure", "Unsure"] };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

let DATA = null;
const S = { pi: 0, path: [], cert: "s", view: "tg", reveal: false, drawer: null, busy: false, shownTurns: 0, lastOrder: null };

/* ---------- state reconstruction ---------- */
function patient() { return DATA.patients[S.pi]; }
function keyOf(path) { return path.join("/"); }
function stateAt(p, path) {
  const rules = Object.assign({}, p.nodes[""].rules);
  const added = {};
  const turns = [];
  for (let i = 1; i <= path.length; i++) {
    const n = p.nodes[keyOf(path.slice(0, i))];
    Object.assign(rules, n.rules);
    Object.assign(added, n.added);
    turns.push({ q: p.nodes[keyOf(path.slice(0, i - 1))].q, turn: n.turn, code: path[i - 1], node: n });
  }
  return { node: p.nodes[keyOf(path)], rules, added, turns };
}
function rulesOf(st, tid) {
  const t = patient().trials.find((x) => x.id === tid);
  return t.rules.map((r) => {
    const v = st.rules[`${tid}|${r.id}`];
    return { ...r, status: v[0], category: v[1], decision: v[2], conf: v[3], ev: v[4], cls: v[5] };
  });
}
function order(scores) { return Object.keys(scores).sort((a, b) => scores[b] - scores[a] || (a < b ? -1 : 1)); }
function trialStatus(rules) {
  const n = rules.filter((r) => r.cls === "against").length;
  if (n) return [`Ruled out by ${n} rule${n > 1 ? "s" : ""}`, "no"];
  if (rules.some((r) => r.cls === "open")) return ["Needs information", "info"];
  return ["All rules settled", "ok"];
}
function sentenceText(st, id) {
  const p = patient();
  if (st.added[id] != null) return { text: st.added[id], doc: true };
  if (p.visible[id] != null) return { text: p.visible[id], doc: false };
  return null;
}

/* ---------- small drawings ---------- */
function gauge(v, b6) {
  const R = 42, C = Math.PI * R, f = Math.max(0, Math.min(1, v ?? 0)), fb = Math.max(0, Math.min(1, b6 ?? 0));
  const ang = Math.PI * (1 - fb), mx = 50 + R * Math.cos(ang), my = 52 - R * Math.sin(ang);
  return `<svg width="104" height="62" viewBox="0 0 100 60" aria-hidden="true">
    <path d="M8 52a42 42 0 0 1 84 0" fill="none" stroke="#e8edf3" stroke-width="10" stroke-linecap="round"/>
    <path d="M8 52a42 42 0 0 1 84 0" fill="none" stroke="url(#gg)" stroke-width="10" stroke-linecap="round"
      stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - f)}" style="transition:stroke-dashoffset .8s cubic-bezier(.2,.8,.2,1)"/>
    <defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#14b8a6"/><stop offset="1" stop-color="#0f766e"/></linearGradient></defs>
    <circle cx="${mx}" cy="${my}" r="4.5" fill="#fff" stroke="#64748b" stroke-width="2"><title>B6 ${fmt(b6)}</title></circle>
  </svg>`;
}
function donut(counts) {
  const keys = ["stated", "inferred", "answered", "against", "open"];
  const total = keys.reduce((s, k) => s + counts[k], 0) || 1;
  const R = 26, C = 2 * Math.PI * R;
  let off = 0, segs = "";
  for (const k of keys) {
    const len = (counts[k] / total) * C;
    if (len > 0) segs += `<circle r="${R}" cx="34" cy="34" fill="none" stroke="${COL[k]}" stroke-width="11" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 34 34)"><title>${CLASS_LABEL[k]}: ${counts[k]}</title></circle>`;
    off += len;
  }
  return `<svg width="68" height="68" viewBox="0 0 68 68" aria-hidden="true">${segs}<text x="34" y="38" text-anchor="middle" font-size="13" font-weight="800" fill="#0f172a">${total}</text></svg>`;
}
function ring(score, cls, size = 44) {
  const f = Math.max(0, Math.min(1, (score + 2) / 3)), R = size / 2 - 4, C = 2 * Math.PI * R, c = size / 2;
  const colour = { ok: "#16a34a", info: "#d97706", no: "#dc2626" }[cls] || "#64748b";
  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-hidden="true">
    <circle cx="${c}" cy="${c}" r="${R}" fill="none" stroke="#eef1f5" stroke-width="4"/>
    <circle cx="${c}" cy="${c}" r="${R}" fill="none" stroke="${colour}" stroke-width="4" stroke-linecap="round"
      stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - f)}" transform="rotate(-90 ${c} ${c})" style="transition:stroke-dashoffset .6s"/>
    <text x="${c}" y="${c + 3.5}" text-anchor="middle" font-size="${size > 50 ? 13 : 10.5}" font-weight="750" fill="#0f172a">${fmt(score)}</text></svg>`;
}
function heroArt() {
  return `<svg viewBox="0 0 560 120" width="100%" height="120" aria-hidden="true">
    <defs><linearGradient id="hb" x1="0" x2="1"><stop offset="0" stop-color="#0b1b34"/><stop offset="1" stop-color="#0f766e"/></linearGradient></defs>
    <rect width="560" height="120" rx="16" fill="url(#hb)"/>
    <path d="M0 78h120l16-30 22 56 18-40 14 14h120" fill="none" stroke="#5eead4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity=".9"/>
    <g transform="translate(330 22)"><rect width="78" height="78" rx="12" fill="#fff" opacity=".96"/>
      <rect x="14" y="16" width="50" height="6" rx="3" fill="#cbd5e1"/><rect x="14" y="30" width="38" height="6" rx="3" fill="#e2e8f0"/>
      <rect x="14" y="44" width="46" height="6" rx="3" fill="#94a3b8" opacity=".5" style="filter:blur(1.5px)"/><rect x="14" y="58" width="30" height="6" rx="3" fill="#a78bfa"/></g>
    <g transform="translate(430 30)"><rect width="110" height="62" rx="12" fill="#ffffff22" stroke="#ffffff44"/>
      <rect x="12" y="12" width="60" height="7" rx="3.5" fill="#5eead4"/><rect x="12" y="27" width="84" height="7" rx="3.5" fill="#ffffff88"/>
      <rect x="12" y="42" width="40" height="7" rx="3.5" fill="#fde68a"/></g>
    <circle cx="300" cy="61" r="7" fill="#5eead4"/><path d="M308 61h16M408 61h16" stroke="#5eead4" stroke-width="2" stroke-dasharray="3 4"/>
  </svg>`;
}

/* ---------- rendering ---------- */
function render(animate) {
  const p = patient(), st = stateAt(p, S.path), n = st.node;
  renderSidebar();
  renderHeader(p);
  renderKpis(p, st);
  renderNote(p, st);
  renderChat(p, st, animate);
  renderTrials(p, st, animate);
  if (S.drawer) openDrawer(S.drawer, true);
  $("btnUndo").disabled = S.path.length === 0;
  document.querySelectorAll(".ans").forEach((b) => (b.disabled = n.done || S.busy));
}

function renderSidebar() {
  $("plist").innerHTML = DATA.patients.map((p, i) => {
    const first = firstSentence(p);
    const tag = p.cohort === "sigir" ? "SIGIR 2016" : "TREC 2021";
    const bug = p.nodes[""].q && /prior chemotherapy/i.test(p.nodes[""].q.text);
    return `<button class="pt ${i === S.pi ? "active" : ""}" data-i="${i}"><div class="avatar" style="${avatarBg(i)}">P${i + 1}</div>
      <div style="min-width:0"><div class="nm">Patient ${i + 1} <span style="color:var(--faint);font-weight:500">· ${esc(p.id)}</span></div>
      <div class="ds">${esc(first)}</div><span class="tag">${tag}</span>${bug ? ' <span class="tag" style="color:#92400e;border-color:#fde68a" title="Recorded before the key fix: its first question is the copied “prior chemotherapy” key">before key fix</span>' : ""}</div></button>`;
  }).join("");
}
function avatarBg(i) {
  return ["", "background:linear-gradient(135deg,#8b5cf6,#6366f1)", "background:linear-gradient(135deg,#f59e0b,#ef4444)"][i % 3];
}
function firstSentence(p) {
  const all = { ...p.visible, ...p.hidden };
  const k = Object.keys(all).map(Number).sort((a, b) => a - b)[0];
  return all[k] || "";
}
function renderHeader(p) {
  $("ptAvatar").textContent = `P${S.pi + 1}`;
  $("ptAvatar").setAttribute("style", avatarBg(S.pi));
  $("ptTitle").textContent = `Patient ${S.pi + 1}`;
  $("ptDesc").textContent = firstSentence(p);
  $("ptCohort").textContent = `${p.id} · ${p.trials.length} candidate trials`;
}

function renderKpis(p, st) {
  const n = st.node, d = n.ndcg - p.b6.ndcg;
  $("kpiNdcg").innerHTML = `${gauge(n.ndcg, p.b6.ndcg)}<div><div class="lbl">Ranking · NDCG@10</div>
    <div class="val">${fmt(n.ndcg)}</div><div class="sub">B6 (abstain only) ${fmt(p.b6.ndcg)} <span class="${d >= 0 ? "up" : "down"}">${d >= 0 ? "▲ +" : "▼ "}${fmt(d)}</span></div></div>`;
  const c = { stated: 0, inferred: 0, answered: 0, against: 0, open: 0 };
  Object.values(st.rules).forEach((v) => c[v[5]]++);
  $("kpiRules").innerHTML = `${donut(c)}<div><div class="lbl">Rules checked</div>
    <div class="sub" style="line-height:1.6"><b style="color:${COL.stated}">${c.stated}</b> stated · <b style="color:${COL.inferred}">${c.inferred}</b> inferred · <b style="color:${COL.answered}">${c.answered}</b> answered<br><b style="color:${COL.against}">${c.against}</b> rule a trial out · <b style="color:#64748b">${c.open}</b> open</div></div>`;
  const used = st.turns.length, budget = DATA.meta.budget;
  $("kpiQs").innerHTML = `<div style="flex:1"><div class="lbl">Questions to the doctor</div><div class="val">${used}<span style="font-size:15px;color:var(--muted)"> / ${budget}</span></div>
    <div class="qdots">${Array.from({ length: budget }, (_, i) => `<i class="${i < used ? "used" : ""}"></i>`).join("")}</div>
    <div class="sub">${n.done ? "Conversation finished" : "One at a time, best question first"}</div></div>`;
  $("kpiHeld").innerHTML = `<div style="width:46px;height:46px;border-radius:12px;background:#fef3c7;color:#b45309;display:grid;place-items:center;flex:none">${icon("shield", 24)}</div>
    <div><div class="lbl">Held back by <span style="text-transform:none">λ</span></div><div class="val">${n.held}</div>
    <div class="sub" title="Inferences that would make the patient look eligible, with confidence below λ = ${fmt(DATA.meta.threshold, 3)}: left open instead of guessed">risky inferences left open (λ = ${fmt(DATA.meta.threshold, 3)})</div></div>`;
}

function renderNote(p, st) {
  const ids = [...Object.keys(p.visible).map((k) => [Number(k), "v"]), ...Object.keys(p.hidden).map((k) => [Number(k), "h"])].sort((a, b) => a[0] - b[0]);
  let html = ids.map(([id, kind]) => kind === "v"
    ? `<div class="sent" data-sid="${id}"><span class="ln">${id}</span><span class="tx">${esc(p.visible[id])}</span></div>`
    : `<div class="sent hid ${S.reveal ? "reveal" : ""}" title="Hidden from the AI by the 40% rule-targeted mask"><span class="ln">${icon("lock", 11, 2.4)}</span><span class="tx">${esc(p.hidden[id])}</span></div>`).join("");
  const added = Object.entries(st.added);
  html += added.map(([id, s], i) => `<div class="sent doc" data-sid="${id}"><span class="ln">D${i + 1}</span><span class="tx">${esc(s.replace(/^Doctor's answer about [^:]*:\s*/, ""))}<span class="badge-doc">YOUR ANSWER</span></span></div>`).join("");
  $("note").innerHTML = html;
  $("noteMeta").textContent = `${Object.keys(p.visible).length} visible · ${Object.keys(p.hidden).length} hidden${added.length ? ` · ${added.length} added` : ""}`;
}

function answerLabel(code) {
  if (code === "nd") return ["Not in the record", ""];
  return [code[0] === "y" ? "Yes" : "No", CERT[code[1]][1]];
}
function effectHtml(turn, prevNdcg, ndcg) {
  const s = turn.settled || [];
  const d = ndcg - prevNdcg;
  const nd = `<div class="why">Ranking quality ${fmt(prevNdcg)} → <b>${fmt(ndcg)}</b> <span class="${d > 0 ? "up" : d < 0 ? "down" : ""}" style="color:${d > 0 ? COL.stated : d < 0 ? COL.against : "inherit"}">(${d >= 0 ? "+" : ""}${fmt(d)})</span></div>`;
  if (!s.length) {
    const why = turn.certainty === "not_documented" ? "The record does not say, so the rules stay open and are reported as open."
      : turn.certainty === "unsure" ? "You were unsure, so nothing is settled yet; I may ask a follow-up."
      : turn.certainty === "probably" ? "“Probably” is treated as an inference, and it did not pass risk control, so the rules stay open."
      : "No rule changed.";
    return `<div class="tiny">Result</div>${esc(why)}${nd}`;
  }
  const trials = new Set(s.map((x) => x[0])).size;
  const chips = s.slice(0, 8).map(([t, r, a, b]) => `<span class="chipbtn" data-tid="${esc(t)}" title="Open ${esc(t)}">${esc(t)} · ${esc(r)}: ${esc(STATUS[a] || a)} → <b>${esc(STATUS[b] || b)}</b></span>`).join("");
  const more = s.length > 8 ? `<span>+${s.length - 8} more</span>` : "";
  return `<div class="tiny">Result</div>Your answer settled <b>${s.length} rule${s.length > 1 ? "s" : ""}</b> in <b>${trials} trial${trials > 1 ? "s" : ""}</b>.<div class="effect">${chips}${more}</div>${nd}`;
}
function questionHtml(q) {
  const fact = esc(q.fact);
  const text = esc(q.text).replace(fact, `<u>${fact}</u>`);
  const label = q.followup ? "Follow-up · you were unsure" : `Question ${q.turn} of ${DATA.meta.budget}`;
  return `<div class="tiny">${label}</div><div class="q">${text}</div>
    <div class="why">Why this one: an answer can settle <b>${q.rules} rule${q.rules > 1 ? "s" : ""}</b> in <b>${q.trials.length} trial${q.trials.length > 1 ? "s" : ""}</b> (${q.trials.map(esc).join(", ")}), and it is the question most likely to change the top of the ranking.</div>`;
}
function renderChat(p, st, animate) {
  const root = p.nodes[""], msgs = [];
  const c = { stated: 0, inferred: 0, against: 0, open: 0 };
  Object.values(root.rules).forEach((v) => { if (c[v[5]] != null) c[v[5]]++; });
  const bug = root.q && /prior chemotherapy/i.test(root.q.text);
  msgs.push(ai(`<div class="tiny">Note checked</div>I read the ${Object.keys(p.visible).length} sentences I can see (${Object.keys(p.hidden).length} are hidden from me) and checked <b>${Object.keys(root.rules).length} rules</b> across <b>${p.trials.length} trials</b>.
    <div class="effect"><span style="border-color:#bbf7d0">${c.stated} stated</span><span style="border-color:#bfdbfe">${c.inferred} inferred</span><span style="border-color:#fecaca">${c.against} rule a trial out</span><span>${c.open} open</span></div>
    <div class="why">${root.held ? `Risk control held back ${root.held} inference${root.held > 1 ? "s" : ""} that would make the patient look eligible. ` : ""}I can ask you up to ${DATA.meta.budget} questions to settle the open ones.</div>
    ${bug ? `<div class="why" style="color:#92400e">Note: this patient was recorded before our key fix, so the first question is the copied “prior chemotherapy” key. The fixed run removes it (see results/b7_keyfix).</div>` : ""}`));
  let prev = root.ndcg;
  st.turns.forEach((t) => {
    msgs.push(ai(questionHtml(t.q)));
    const [a, cert] = answerLabel(t.code);
    msgs.push(me(`${a}${cert ? ` <span style="opacity:.8">· ${cert}</span>` : ""}`));
    msgs.push(ai(effectHtml(t.turn, prev, t.node.ndcg)));
    prev = t.node.ndcg;
  });
  const n = st.node;
  if (!n.done && n.q) msgs.push(ai(questionHtml(n.q)));
  else {
    let open = 0;
    Object.values(st.rules).forEach((v) => { if (v[5] === "open") open++; });
    msgs.push(ai(`<div class="tiny" style="color:${COL.stated}">Finished</div>${st.turns.length} question${st.turns.length === 1 ? "" : "s"} asked. ${open} rules are still open; they are reported as open for the care team to decide, never guessed.
      <div class="why">TrialGuide stops early when no remaining question can change the top ten. Click any trial to see each rule, or <a href="#" data-act="save">save the session</a>.</div>`));
  }
  const old = S.shownTurns;
  $("chat").innerHTML = msgs.map((m, i) => m.replace('class="msg', `class="msg${animate && i >= old ? "" : " static"}`)).join("");
  S.shownTurns = msgs.length;
  $("chatMeta").textContent = n.done ? "finished" : "waiting for your answer";
  const box = $("chat").parentElement;
  box.scrollTo({ top: box.scrollHeight, behavior: animate ? "smooth" : "auto" });
}
const ai = (h) => `<div class="msg"><div class="av ai">${icon("spark", 16)}</div><div class="bubble">${h}</div></div>`;
const me = (h) => `<div class="msg me"><div class="av">${icon("steth", 16)}</div><div class="bubble">${h}</div></div>`;

function renderTrials(p, st, animate) {
  const n = st.node, scores = S.view === "tg" ? n.scores : p.b6.scores;
  const ord = order(scores), b6ord = order(p.b6.scores);
  const prevOrd = S.lastOrder;
  const before = {};
  if (animate) document.querySelectorAll("#trials .trial").forEach((el) => (before[el.dataset.tid] = el.getBoundingClientRect().top));
  $("trials").innerHTML = ord.map((tid, i) => {
    const t = p.trials.find((x) => x.id === tid), rules = rulesOf(st, tid), [stxt, scls] = trialStatus(rules);
    let mv = "";
    if (S.view === "tg") {
      const m = b6ord.indexOf(tid) - i;
      mv = m > 0 ? `<span class="mv up" title="vs abstain only (B6)">▲${m}</span>` : m < 0 ? `<span class="mv down" title="vs abstain only (B6)">▼${-m}</span>` : `<span class="mv same" title="same as B6">=</span>`;
    }
    const moved = prevOrd && prevOrd.indexOf(tid) !== i && animate ? " flash" : "";
    const [g, gc] = GOLD[t.gold] || ["", "none"];
    const strip = rules.map((r) => `<i class="${r.cls}" title="${esc(r.id)}: ${esc(CLASS_LABEL[r.cls])}"></i>`).join("");
    return `<div class="trial${moved}" data-tid="${esc(tid)}" tabindex="0" role="button" aria-label="Open ${esc(tid)}">
      <div class="rk">${i + 1}${mv}</div>
      <div style="min-width:0"><div class="tt">${esc(t.title || tid)}</div><div class="id">${esc(tid)} · ${rules.length} rules</div><div class="strip">${strip}</div></div>
      <div style="text-align:right">${ring(scores[tid], scls)}<div><span class="status ${scls}">${stxt}</span></div><div class="expert">Expert label: <b>${g}</b></div></div></div>`;
  }).join("");
  if (animate) {
    document.querySelectorAll("#trials .trial").forEach((el) => {
      const b = before[el.dataset.tid];
      if (b == null) return;
      const dy = b - el.getBoundingClientRect().top;
      if (!dy) return;
      el.style.transform = `translateY(${dy}px)`;
      el.style.transition = "none";
      requestAnimationFrame(() => { el.style.transition = "transform .6s cubic-bezier(.2,.8,.2,1)"; el.style.transform = ""; });
    });
  }
  S.lastOrder = ord;
}

/* ---------- drawer ---------- */
function openDrawer(tid, refresh) {
  const p = patient(), st = stateAt(p, S.path), t = p.trials.find((x) => x.id === tid);
  if (!t) return;
  S.drawer = tid;
  const rules = rulesOf(st, tid), [stxt, scls] = trialStatus(rules), score = st.node.scores[tid];
  const rank = order(st.node.scores).indexOf(tid) + 1, b6rank = order(p.b6.scores).indexOf(tid) + 1;
  $("dRing").innerHTML = ring(score, scls, 58);
  $("dTitle").textContent = t.title || tid;
  const [g] = GOLD[t.gold] || [""];
  $("dSub").innerHTML = `${esc(tid)} · rank <b>${rank}</b> (B6: ${b6rank}) · <span class="status ${scls}">${stxt}</span> · expert label: <b>${g}</b>`;
  const c = { stated: 0, inferred: 0, answered: 0, against: 0, open: 0 };
  rules.forEach((r) => c[r.cls]++);
  const summary = `<div class="dsum">${Object.keys(c).map((k) => `<div><span class="sq ${k}"></span><b>${c[k]}</b> ${CLASS_LABEL[k].toLowerCase()}</div>`).join("")}</div>
    <div class="why" style="margin:6px 0 0;color:var(--muted);font-size:12.5px">Score = share of inclusion rules met, minus 1 if any inclusion rule is not met, minus 1 if any exclusion rule is met (TrialGPT's linear score). Open rules count as not enough information. Hover a rule to light up the sentence it cites in the note.</div>`;
  const group = (kind, title) => {
    const rs = rules.filter((r) => r.kind === kind);
    if (!rs.length) return "";
    return `<div class="grp">${title} · ${rs.length}</div>` + rs.map((r) => {
      const txt = (r.text.match(/[a-z]/gi) || []).length < 3 ? `<span style="color:var(--faint)">(empty criterion left over from the trial text)</span>` : esc(r.text);
      const evs = r.ev.map((id) => [id, sentenceText(st, String(id))]).filter((x) => x[1]);
      const ev = evs.length ? `<div class="ev">Cites ${evs.map(([id, s]) => `<b>${s.doc ? "your answer" : "sentence " + id}</b>: “${esc(s.text.replace(/^Doctor's answer about [^:]*:\s*/, "").slice(0, 140))}”`).join("<br>")}</div>`
        : r.cls === "open" ? `<div class="ev">No sentence in the visible note settles it${r.decision === "abstain" ? "; TrialGuide abstained." : "."}</div>` : "";
      const cc = r.conf >= DATA.meta.threshold ? COL.stated : "#d97706";
      return `<div class="rule" data-ev="${r.ev.join(",")}"><span class="dot" style="background:${COL[r.cls]}"></span>
        <div><div class="txt">${txt}</div>${ev}</div>
        <div class="side"><span class="dec">${esc(DECISION[r.decision] || r.decision)}</span><div style="margin-top:4px">${esc(STATUS[r.status] || r.status)}</div>
        <div class="conf" title="model confidence ${fmt(r.conf)} (λ = ${fmt(DATA.meta.threshold, 3)})"><i style="width:${Math.round(r.conf * 100)}%;background:${cc}"></i></div>confidence ${fmt(r.conf)}</div></div>`;
    }).join("");
  };
  $("dBody").innerHTML = summary + group("inclusion", "Inclusion criteria · the patient must meet") + group("exclusion", "Exclusion criteria · the patient must not meet");
  if (!refresh) { $("drawer").classList.add("on"); $("scrim").classList.add("on"); $("drawer").setAttribute("aria-hidden", "false"); }
}
function closeDrawer() {
  S.drawer = null;
  $("drawer").classList.remove("on"); $("scrim").classList.remove("on"); $("drawer").setAttribute("aria-hidden", "true");
  highlight([]);
}
function highlight(ids) {
  document.querySelectorAll("#note .sent.hl").forEach((el) => el.classList.remove("hl"));
  ids.forEach((id) => { const el = document.querySelector(`#note .sent[data-sid="${id}"]`); if (el) { el.classList.add("hl"); el.scrollIntoView({ block: "nearest", behavior: "smooth" }); } });
}

/* ---------- actions ---------- */
function answer(a) {
  if (S.busy) return;
  const p = patient(), code = a === "d" ? "nd" : a + S.cert;
  const next = [...S.path, code];
  if (!p.nodes[keyOf(next)]) { toast("That answer was not recorded for this patient."); return; }
  S.busy = true;
  document.querySelectorAll(".ans").forEach((b) => (b.disabled = true));
  const [lbl, cert] = answerLabel(code);
  const chat = $("chat");
  chat.insertAdjacentHTML("beforeend", me(`${lbl}${cert ? ` <span style="opacity:.8">· ${cert}</span>` : ""}`) +
    `<div class="msg" id="typing"><div class="av ai">${icon("spark", 16)}</div><div class="bubble"><span class="typing"><i></i><i></i><i></i></span> <span style="color:var(--muted);font-size:12.5px">re-checking rules and re-ranking…</span></div></div>`);
  chat.parentElement.scrollTo({ top: chat.parentElement.scrollHeight, behavior: "smooth" });
  S.shownTurns += 1;
  setTimeout(() => { S.path = next; S.busy = false; render(true); }, 850);
}
function setPatient(i) { S.pi = i; S.path = []; S.shownTurns = 0; S.lastOrder = null; closeDrawer(); render(true); }
function save() {
  const p = patient(), st = stateAt(p, S.path);
  const out = { tool: "TrialGuide web demo", exported: new Date().toISOString(), meta: DATA.meta, patient: p.id, path: keyOf(S.path),
    turns: st.turns.map((t) => ({ question: t.turn.question, answer: t.turn.answer, certainty: t.turn.certainty, settled: t.turn.settled, ndcg_after: t.node.ndcg })),
    ndcg: st.node.ndcg, b6_ndcg: p.b6.ndcg, held_back: st.node.held,
    ranking: order(st.node.scores).map((t, i) => ({ rank: i + 1, trial: t, score: st.node.scores[t], counts: st.node.counts[t] })) };
  const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = `trialguide_${p.id}${S.path.length ? "_" + S.path.join("-") : ""}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Session saved as JSON.");
}
function toast(msg) {
  let t = $("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("on");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("on"), 2200);
}
function showModal(on) { $("modal").classList.toggle("hidden", !on); if (!on) { try { localStorage.setItem("tg-seen", "1"); } catch (e) {} } }

function bind() {
  document.querySelectorAll("[data-icon]").forEach((el) => (el.innerHTML = icon(el.dataset.icon, 15)));
  $("plist").addEventListener("click", (e) => { const b = e.target.closest(".pt"); if (b) setPatient(Number(b.dataset.i)); });
  document.querySelectorAll(".ans").forEach((b) => b.addEventListener("click", () => answer(b.dataset.a)));
  $("seg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S.cert = b.dataset.c;
    $("seg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    $("segHint").textContent = { s: "Used as a stated fact", p: "Treated as an inference, must pass risk control", u: "May trigger one follow-up question" }[S.cert];
  });
  $("view").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S.view = b.dataset.v;
    $("view").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    renderTrials(patient(), stateAt(patient(), S.path), true);
  });
  $("trials").addEventListener("click", (e) => { const t = e.target.closest(".trial"); if (t) openDrawer(t.dataset.tid); });
  $("trials").addEventListener("keydown", (e) => { const t = e.target.closest(".trial"); if (t && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openDrawer(t.dataset.tid); } });
  $("chat").addEventListener("click", (e) => {
    const c = e.target.closest(".chipbtn"); if (c) openDrawer(c.dataset.tid);
    const a = e.target.closest("[data-act=save]"); if (a) { e.preventDefault(); save(); }
  });
  $("dBody").addEventListener("mouseover", (e) => { const r = e.target.closest(".rule"); highlight(r && r.dataset.ev ? r.dataset.ev.split(",") : []); });
  $("dBody").addEventListener("mouseleave", () => highlight([]));
  $("dClose").addEventListener("click", closeDrawer);
  $("scrim").addEventListener("click", closeDrawer);
  $("reveal").addEventListener("change", (e) => { S.reveal = e.target.checked; renderNote(patient(), stateAt(patient(), S.path)); });
  $("btnUndo").addEventListener("click", () => { if (S.path.length) { S.path = S.path.slice(0, -1); S.shownTurns = 0; render(true); } });
  $("btnRestart").addEventListener("click", () => { S.path = []; S.shownTurns = 0; render(true); });
  $("btnSave").addEventListener("click", save);
  $("btnHelp").addEventListener("click", () => showModal(true));
  $("linkHelp").addEventListener("click", (e) => { e.preventDefault(); showModal(true); });
  $("btnStart").addEventListener("click", () => showModal(false));
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") showModal(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); showModal(false); } });
}

function buildModal() {
  $("heroArt").innerHTML = heroArt();
  const steps = [["note", "Read the note", "Only the visible sentences"], ["list", "Check every rule", "Met, not met, or no info, with a cited sentence"],
    ["split", "Infer · ask · abstain", "Inferences toward eligibility must pass λ"], ["ask", "Ask the doctor", "Up to 3 questions, best first"], ["rank", "Re-rank", "Every answer moves the trials"]];
  $("flow").innerHTML = steps.map(([i, b, s]) => `<div class="st"><div class="ic">${icon(i, 22)}</div><b>${b}</b><span>${s}</span></div>`).join("");
}

async function main() {
  bind();
  buildModal();
  try {
    const r = await fetch("data.json");
    DATA = await r.json();
  } catch (e) {
    document.querySelector(".main").innerHTML = `<div class="card" style="padding:24px">Could not load data.json. Serve this folder over HTTP (for example <code>python3 -m http.server</code>) and open it again.</div>`;
    return;
  }
  $("chipModel").textContent = `Checker: ${DATA.meta.model.split("/").pop()}`;
  $("chipLambda").textContent = `Risk control λ = ${fmt(DATA.meta.threshold, 3)} · α = 5%`;
  render(false);
  let seen = false;
  try { seen = localStorage.getItem("tg-seen") === "1"; } catch (e) {}
  if (!seen && !/[?&]nointro/.test(location.search)) showModal(true);
}
main();
