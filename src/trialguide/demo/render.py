"""HTML pieces of the demo page. Pure functions: a session (and the bundle's patient record) in,
an HTML string out, so the look can be tested without a browser. Owner: Prakash."""

from __future__ import annotations

from html import escape

from trialguide.demo.session import DemoSession

# One colour per way a rule ended up, used everywhere on the page
COLOURS = {
    "stated": "#16a34a",      # the note says it
    "inferred": "#2563eb",    # inferred from evidence and accepted
    "answered": "#7c3aed",    # settled by the doctor's answer
    "open": "#9ca3af",        # still open: asked, abstained, or held back
    "against": "#dc2626",     # a settled rule that rules the trial out
}
GOLD = {0: ("not relevant", "#6b7280"), 1: ("excluded / potential", "#d97706"), 2: ("eligible", "#16a34a")}
STEPS = ["Read the note", "Check every rule", "Infer · ask · abstain", "Ask the doctor", "Re-rank"]

CSS = """
.tg-hero{background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#4f46e5 100%);color:#fff;border-radius:18px;
  padding:26px 30px;box-shadow:0 10px 30px rgba(30,58,138,.25)}
.tg-hero h1{margin:0;font-size:34px;letter-spacing:-.5px;color:#fff}
.tg-hero p{margin:6px 0 14px;opacity:.95;font-size:15px;color:#e0e7ff}
.tg-hero b{color:#fde68a;font-weight:700}
.tg-pill{display:inline-block;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);
  border-radius:999px;padding:3px 11px;margin:3px 6px 0 0;font-size:12px;color:#fff}
.tg-steps{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 4px}
.tg-step{flex:1;min-width:120px;border-radius:12px;padding:9px 12px;font-size:13px;font-weight:600;
  background:#eef2ff;color:#475569;border:1px solid #e0e7ff}
.tg-step.done{background:#dcfce7;color:#166534;border-color:#bbf7d0}
.tg-step.now{background:#4f46e5;color:#fff;border-color:#4f46e5;box-shadow:0 4px 14px rgba(79,70,229,.35)}
.tg-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:6px 0}
.tg-kpi{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.tg-kpi .v{font-size:26px;font-weight:800;color:#0f172a;line-height:1.1}
.tg-kpi .l{font-size:12px;color:#64748b;margin-top:3px}
.tg-kpi .bar{height:4px;border-radius:4px;margin-top:8px}
.tg-card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:16px 18px;box-shadow:0 2px 8px rgba(15,23,42,.05);color:#0f172a}
.tg-card h3{margin:0 0 10px;font-size:16px;color:#0f172a}
.tg-sent{display:flex;gap:10px;padding:7px 0;border-bottom:1px dashed #eef2f7;font-size:14px;color:#1f2937;align-items:flex-start}
.tg-sent .id{min-width:26px;height:22px;border-radius:7px;background:#f1f5f9;color:#475569;font:600 12px monospace;
  display:flex;align-items:center;justify-content:center}
.tg-sent.hidden .txt{background:repeating-linear-gradient(45deg,#e2e8f0,#e2e8f0 6px,#f1f5f9 6px,#f1f5f9 12px);
  color:transparent;border-radius:6px;user-select:none;flex:1}
.tg-sent.revealed .txt{color:#94a3b8;font-style:italic;flex:1}
.tg-sent.doctor{background:#f5f3ff;border-radius:10px;padding:7px 8px}
.tg-sent.doctor .id{background:#7c3aed;color:#fff}
.tg-tag{display:inline-block;font-size:11px;font-weight:700;border-radius:6px;padding:1px 7px;margin-left:6px;vertical-align:1px}
.tg-trial{display:grid;grid-template-columns:44px 1fr 150px;gap:12px;align-items:center;background:#fff;border:1px solid #e5e7eb;
  border-radius:14px;padding:12px 14px;margin-bottom:9px;color:#0f172a;transition:box-shadow .2s}
.tg-trial:hover{box-shadow:0 6px 18px rgba(15,23,42,.08)}
.tg-rank{font-size:24px;font-weight:800;color:#1e3a8a;text-align:center}
.tg-move{font-size:11px;font-weight:700;display:block}
.tg-title{font-weight:650;font-size:14px;color:#0f172a}
.tg-sub{font-size:12px;color:#64748b;margin:2px 0 6px}
.tg-dots span{display:inline-block;width:11px;height:11px;border-radius:3px;margin:0 3px 3px 0}
.tg-score{font-size:12px;color:#334155}
.tg-track{height:8px;background:#f1f5f9;border-radius:6px;overflow:hidden;margin:4px 0 6px}
.tg-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,#6366f1,#22c55e)}
.tg-q{background:linear-gradient(135deg,#eef2ff,#faf5ff);border:1px solid #c7d2fe;border-radius:16px;padding:18px;color:#0f172a}
.tg-q .big{font-size:19px;font-weight:750;line-height:1.35;margin:4px 0 10px;color:#1e1b4b}
.tg-q .why{font-size:13px;color:#475569}
.tg-chip{display:inline-block;background:#fff;border:1px solid #c7d2fe;color:#3730a3;border-radius:999px;padding:2px 9px;
  font-size:12px;margin:3px 4px 0 0}
.tg-done{background:linear-gradient(135deg,#ecfdf5,#eff6ff);border:1px solid #a7f3d0;border-radius:16px;padding:18px;color:#064e3b}
.tg-turn{border-left:3px solid #7c3aed;padding:6px 0 8px 12px;margin:0 0 10px 4px;font-size:13px;color:#1f2937}
.tg-turn b{color:#4c1d95}
.tg-legend span{display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:12px;color:#475569}
.tg-legend i{width:11px;height:11px;border-radius:3px;display:inline-block}
.tg-rules{width:100%;border-collapse:separate;border-spacing:0 6px;font-size:13px;color:#0f172a}
.tg-rules td{background:#f8fafc;padding:8px 10px;vertical-align:top}
.tg-rules td:first-child{border-radius:10px 0 0 10px}.tg-rules td:last-child{border-radius:0 10px 10px 0}
.tg-rules,.tg-rules tr,.tg-rules td,.tg-rules th{border:none !important}
.tg-rules th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#64748b;padding:0 10px}
.tg-conf{height:6px;background:#e2e8f0;border-radius:4px;width:70px;overflow:hidden;display:inline-block;vertical-align:middle}
.tg-conf i{display:block;height:100%}
"""


def _tag(text: str, colour: str) -> str:
    return f'<span class="tg-tag" style="background:{colour}1a;color:{colour}">{escape(text)}</span>'


def hero(meta: dict, live: bool) -> str:
    pills = [f"Model: {meta.get('model', '?')}", f"Masking: {meta.get('mask', '?')}",
             f"Risk threshold λ = {meta.get('threshold', '?')}", "Mode: live model" if live else "Mode: recorded run",
             "Development patients only"]
    return (f'<div class="tg-hero"><h1>TrialGuide</h1>'
            f'<p>When a note does not settle an eligibility rule, TrialGuide <b>infers</b> it from evidence, '
            f'<b>asks</b> the doctor one question at a time, or <b>abstains</b>. You play the doctor.</p>'
            + "".join(f'<span class="tg-pill">{escape(p)}</span>' for p in pills) + "</div>")


def stepper(session: DemoSession | None) -> str:
    if session is None:
        now = 0
    elif session.finished:
        now = len(STEPS)
    else:
        now = 3 if not session.turns else 4
    cells = []
    for i, name in enumerate(STEPS):
        cls = "done" if i < now else ("now" if i == now else "")
        mark = "✓ " if i < now else f"{i + 1}. "
        cells.append(f'<div class="tg-step {cls}">{mark}{escape(name)}</div>')
    return '<div class="tg-steps">' + "".join(cells) + "</div>"


def kpis(session: DemoSession, gold: dict) -> str:
    total = {"stated": 0, "inferred": 0, "answered": 0, "open": 0}
    for t in session.trial_ids():
        for k, v in session.counts(t).items():
            total[k] += v
    n = sum(total.values()) or 1
    inf = session.inference_counts()
    ours, b6 = session.ndcg(gold) if gold else (None, None)
    tiles = [(total["stated"], "rules stated in the note", COLOURS["stated"]),
             (total["inferred"], "rules inferred from evidence", COLOURS["inferred"]),
             (total["answered"], "rules settled by your answers", COLOURS["answered"]),
             (total["open"], "rules still open", COLOURS["open"]),
             (inf["held_back"], "inferences held back by λ", "#d97706"),
             (f"{session.state.questions_left()}/{session.state.budget}", "questions left", "#4f46e5")]
    html = []
    for value, label, colour in tiles:
        width = (value / n * 100) if isinstance(value, int) else 100
        html.append(f'<div class="tg-kpi"><div class="v" style="color:{colour}">{value}</div>'
                    f'<div class="l">{escape(label)}</div><div class="bar" style="width:{width:.0f}%;background:{colour}"></div></div>')
    if ours is not None:
        html.append(f'<div class="tg-kpi"><div class="v">{ours:.2f}</div><div class="l">NDCG@10, TrialGuide '
                    f'(B6: {b6:.2f}) vs expert labels</div><div class="bar" style="width:{ours * 100:.0f}%;'
                    f'background:linear-gradient(90deg,#6366f1,#22c55e)"></div></div>')
    return '<div class="tg-kpis">' + "".join(html) + "</div>"


def note_html(patient: dict, session: DemoSession | None, reveal: bool) -> str:
    """The note in its original order: visible sentences, hidden ones as redacted bars
    (or greyed when revealed), and the doctor's answers added by the conversation."""
    visible = {int(i): s for i, s in patient["visible"].items()}
    hidden = {int(i): s for i, s in patient["hidden"].items()}
    extra = {}
    if session is not None:
        extra = {i: s for i, s in session.note.sentences.items() if i not in visible}
    rows = []
    for i in sorted(set(visible) | set(hidden) | set(extra)):
        if i in visible:
            rows.append(f'<div class="tg-sent"><span class="id">{i}</span><span>{escape(visible[i])}</span></div>')
        elif i in hidden:
            cls = "revealed" if reveal else "hidden"
            text = escape(hidden[i]) if reveal else "hidden from TrialGuide " * 3
            rows.append(f'<div class="tg-sent {cls}"><span class="id">{i}</span><span class="txt">{text}</span></div>')
        else:
            rows.append(f'<div class="tg-sent doctor"><span class="id">{i}</span><span>{escape(extra[i])}'
                        f'{_tag("doctor", COLOURS["answered"])}</span></div>')
    n_hidden = len(hidden)
    return (f'<div class="tg-card"><h3>The patient note <span style="font-weight:400;color:#64748b;font-size:13px">'
            f'· {len(visible)} sentences visible, {n_hidden} hidden by masking</span></h3>' + "".join(rows) + "</div>")


def _rule_colour(session: DemoSession, ref) -> str:
    c, a = session.checks[ref], session.action_of()[ref]
    from trialguide.pipeline import ACCEPTED
    from trialguide.schemas import ActionType, RuleKind, RuleStatus
    if a.action not in ACCEPTED:
        return COLOURS["open"]
    against = (c.kind == RuleKind.INCLUSION and c.status == RuleStatus.NOT_MET) or \
              (c.kind == RuleKind.EXCLUSION and c.status == RuleStatus.MET)
    if against:
        return COLOURS["against"]
    if any(i in session.doctor_certainty for i in c.evidence_ids):
        return COLOURS["answered"]
    return COLOURS["inferred"] if a.action == ActionType.ACCEPT_INFERENCE else COLOURS["stated"]


def legend() -> str:
    items = [("stated", "stated in the note"), ("inferred", "inferred (passed λ)"), ("answered", "answered by you"),
             ("against", "rules the trial out"), ("open", "open")]
    return '<div class="tg-legend">' + "".join(
        f'<span><i style="background:{COLOURS[k]}"></i>{escape(v)}</span>' for k, v in items) + "</div>"


def trials_html(session: DemoSession, patient: dict) -> str:
    info = {t["trial_id"]: t for t in patient["trials"]}
    ours, b6 = session.scores(), session.b6_scores()
    order, b6_order = session.order(ours), session.order(b6)
    lo, hi = min(list(ours.values()) + [-2.0]), 1.0
    cards = []
    for rank, tid in enumerate(order, 1):
        move = b6_order.index(tid) + 1 - rank
        move_html = (f'<span class="tg-move" style="color:#16a34a">▲ {move} vs B6</span>' if move > 0 else
                     f'<span class="tg-move" style="color:#dc2626">▼ {-move} vs B6</span>' if move < 0 else
                     '<span class="tg-move" style="color:#94a3b8">= B6</span>')
        t = info.get(tid, {})
        gold_text, gold_colour = GOLD.get(t.get("gold"), ("", "#6b7280"))
        refs = sorted(r for r in session.rules if r[0] == tid)
        dots = "".join(f'<span title="{escape(r[1])}" style="background:{_rule_colour(session, r)}"></span>' for r in refs)
        c = session.counts(tid)
        pct = (ours[tid] - lo) / (hi - lo) * 100 if hi > lo else 0
        cards.append(
            f'<div class="tg-trial"><div class="tg-rank">{rank}{move_html}</div>'
            f'<div><div class="tg-title">{escape(t.get("title", "")[:110] or tid)}</div>'
            f'<div class="tg-sub">{escape(tid)} · {c["stated"]} stated · {c["inferred"]} inferred · '
            f'{c["answered"]} answered · {c["open"]} open {_tag("gold: " + gold_text, gold_colour) if gold_text else ""}</div>'
            f'<div class="tg-dots">{dots}</div></div>'
            f'<div class="tg-score">score {ours[tid]:.2f} <span style="color:#94a3b8">(B6 {b6[tid]:.2f})</span>'
            f'<div class="tg-track"><div class="tg-fill" style="width:{max(pct, 3):.0f}%"></div></div></div></div>')
    return (f'<div class="tg-card"><h3>Candidate trials, best first</h3>{legend()}'
            f'<div style="height:10px"></div>' + "".join(cards) + "</div>")


def question_html(session: DemoSession | None) -> str:
    if session is None:
        return ('<div class="tg-q"><div class="why">Step 1</div><div class="big">Pick a development patient and press '
                '“Check this patient”.</div></div>')
    if session.finished:
        open_rules = sum(session.counts(t)["open"] for t in session.trial_ids())
        return (f'<div class="tg-done"><div style="font-size:19px;font-weight:750">Conversation finished</div>'
                f'<div style="margin-top:6px;font-size:14px">{len(session.turns)} question(s) asked. '
                f'{open_rules} rule(s) are still open; they are reported as abstentions, and the doctor decides them. '
                f'TrialGuide stops early when no remaining question could change the top ten.</div></div>')
    q = session.question
    trials = sorted({t for t, _ in q.rule_refs})
    kind = "Follow-up (your last answer was unsure)" if q.is_followup else f"Question {q.turn} of {session.state.budget}"
    chips = "".join(f'<span class="tg-chip">{escape(t)}</span>' for t in trials)
    score = f" · selection score {q.score:.2f}" if q.score else ""
    return (f'<div class="tg-q"><div class="why">{escape(kind)}{score}</div><div class="big">{escape(q.text)}</div>'
            f'<div class="why">Why this question: one answer can settle <b>{len(q.rule_refs)} rule(s)</b> in '
            f'<b>{len(trials)} trial(s)</b>, chosen because it can move the top of the ranking the most.</div>'
            f'<div style="margin-top:6px">{chips}</div></div>')


def timeline_html(session: DemoSession | None) -> str:
    if session is None or not session.turns:
        return '<div class="tg-card"><h3>Conversation</h3><div style="color:#64748b;font-size:13px">No questions yet.</div></div>'
    items = []
    for n, t in enumerate(session.turns, 1):
        effects = "".join(f'<span class="tg-chip">{escape(tr)} {escape(r)}: {escape(a)} → {escape(b)}</span>'
                          for tr, r, a, b in t.settled) or '<span class="tg-chip">no rule changed</span>'
        items.append(f'<div class="tg-turn"><b>Q{n}.</b> {escape(t.question)}<br><b>You:</b> {escape(t.answer)} '
                     f'{_tag(t.certainty, COLOURS["answered"])}<br>{effects}</div>')
    return '<div class="tg-card"><h3>Conversation</h3>' + "".join(items) + "</div>"


def rules_html(session: DemoSession | None, trial_id: str | None) -> str:
    if session is None or not trial_id:
        return ""
    rows = []
    for ref in sorted(r for r in session.rules if r[0] == trial_id):
        row = next(x for x in session.rule_rows(trial_id) if x["rule"] == ref[1])
        colour = _rule_colour(session, ref)
        conf = row["confidence"]
        rows.append(
            f'<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:3px;background:{colour}"></span> '
            f'<b>{escape(row["rule"])}</b><div style="color:#64748b;font-size:11px">{escape(row["kind"])}</div></td>'
            f'<td>{escape(row["text"])}</td><td>{escape(row["status"].replace("_", " "))}'
            f'<div style="color:#64748b;font-size:11px">{escape(row["category"])}</div></td>'
            f'<td>{escape(row["decision"].replace("_", " "))}</td>'
            f'<td><span class="tg-conf"><i style="width:{conf * 100:.0f}%;background:{colour}"></i></span> {conf:.2f}</td>'
            f'<td style="color:#334155">{escape(row["evidence"]) or "<span style=color:#94a3b8>none</span>"}</td></tr>')
    return (f'<div class="tg-card"><h3>Rules of {escape(trial_id)}</h3><table class="tg-rules"><tr><th>rule</th>'
            f'<th>criterion</th><th>status</th><th>decision</th><th>confidence</th><th>evidence cited</th></tr>'
            + "".join(rows) + "</table></div>")
