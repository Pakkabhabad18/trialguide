"""HTML pieces of the live page (live_app.py). Kept apart from render.py so the recorded demo
does not change. Pure functions: a TwoModelSession in, an HTML string out. Owner: Prakash."""

from __future__ import annotations

from html import escape

from trialguide.demo import render

# Extra styles for the live page, added after render.CSS. Every colour is set explicitly so the
# page reads the same whatever the browser's light / dark setting is.
CSS = """
.gradio-container{max-width:1180px !important;margin:0 auto !important;background:#f6f7fb !important}
html,body,.gradio-container,.main{background:#f6f7fb !important}
footer{display:none !important}
.lv-top{display:flex;align-items:center;gap:14px;padding:14px 20px;border-radius:16px;background:#0f172a;color:#fff}
.lv-logo{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#6366f1,#22d3ee);display:flex;
  align-items:center;justify-content:center;font-weight:800;font-size:18px;color:#fff}
.lv-name{font-size:20px;font-weight:800;letter-spacing:-.3px;color:#fff}
.lv-tag{font-size:13px;color:#cbd5e1}
.lv-sp{flex:1}
.lv-badge{font-size:12px;padding:4px 10px;border-radius:999px;background:#334155;color:#e2e8f0;white-space:nowrap}
.lv-link{font-size:13px;font-weight:600;padding:7px 12px;border-radius:10px;background:#1e293b;color:#fff !important;
  text-decoration:none !important;border:1px solid #334155;white-space:nowrap}
.lv-link:hover{background:#334155}
.lv-h{font-size:22px;font-weight:800;color:#0f172a;margin:4px 0 2px;letter-spacing:-.3px}
.lv-sub{font-size:14px;color:#475569;margin:0 0 6px}
.lv-panel{background:#fff !important;border:1px solid #e5e7eb !important;border-radius:18px !important;padding:18px !important;
  box-shadow:0 4px 18px rgba(15,23,42,.06) !important}
.lv-how{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.lv-how div{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:16px;color:#0f172a}
.lv-how b{display:inline-flex;width:28px;height:28px;border-radius:9px;background:#eef2ff;color:#4338ca;align-items:center;
  justify-content:center;margin-right:8px;font-size:14px}
.lv-how span{font-weight:700;font-size:15px}
.lv-how p{margin:8px 0 0;font-size:13px;color:#64748b;line-height:1.45}
.lv-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.lv-tile{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:14px 16px}
.lv-tile .v{font-size:28px;font-weight:800;color:#0f172a;line-height:1.1}
.lv-tile .l{font-size:13px;color:#64748b;margin-top:2px}
.lv-q{background:linear-gradient(135deg,#4f46e5,#7c3aed);border-radius:18px;padding:20px 22px;color:#fff}
.lv-q .k{font-size:12px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:#c7d2fe}
.lv-q .t{font-size:21px;font-weight:750;line-height:1.35;margin:8px 0 10px;color:#fff}
.lv-q .w{font-size:13px;color:#e0e7ff}
.lv-done{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:18px;padding:20px 22px;color:#064e3b}
.lv-done .t{font-size:19px;font-weight:750}
.lv-done p{margin:6px 0 0;font-size:14px}
.lv-trial{display:grid;grid-template-columns:40px 1fr;gap:12px;background:#fff;border:1px solid #e5e7eb;border-radius:16px;
  padding:14px 16px;margin-bottom:10px;color:#0f172a}
.lv-rank{width:34px;height:34px;border-radius:10px;background:#eef2ff;color:#3730a3;font-weight:800;display:flex;
  align-items:center;justify-content:center}
.lv-title{font-weight:700;font-size:15px;color:#0f172a;line-height:1.35}
.lv-meta{font-size:12px;color:#64748b;margin:3px 0 8px}
.lv-meta a{color:#4f46e5 !important;text-decoration:none}
.lv-bar{height:8px;background:#f1f5f9;border-radius:6px;overflow:hidden}
.lv-bar i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,#22c55e,#16a34a)}
.lv-row{display:flex;justify-content:space-between;align-items:center;margin-top:7px;font-size:12px;color:#475569}
.lv-pill{font-size:12px;font-weight:700;padding:3px 9px;border-radius:999px}
.lv-conv{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:14px 16px;color:#0f172a}
.lv-conv h4{margin:0 0 8px;font-size:14px;color:#0f172a}
.lv-turn{padding:8px 0;border-top:1px solid #f1f5f9;font-size:13px;color:#334155}
.lv-turn b{color:#0f172a}
.lv-turn .r{color:#16a34a;font-weight:600}
.lv-muted{font-size:13px;color:#94a3b8}
.lv-patient{display:flex;gap:10px;align-items:center;background:#fff;border:1px solid #e5e7eb;border-radius:14px;
  padding:10px 14px;font-size:14px;color:#334155}
.lv-patient b{color:#0f172a}
.lv-answer button{font-size:16px !important;font-weight:700 !important;min-height:52px !important}
.lv-big button,.lv-big{font-size:17px !important;font-weight:700 !important;min-height:54px !important}
@media (max-width:760px){.lv-how,.lv-tiles{grid-template-columns:1fr 1fr}.lv-top{flex-wrap:wrap}.lv-tag{display:none}}
"""

# Forces the light look: a Gradio page follows the browser's dark setting by default, and the
# cards below are designed for light (white cards with dark text).
FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""


def header(fake: bool, home: str = "") -> str:
    """Top bar: name, one line on what it does, a demo-mode badge, and the link to the recorded demo."""
    badge = '<span class="lv-badge">Demo mode · no GPU</span>' if fake else '<span class="lv-badge">Live models</span>'
    back = f'<a class="lv-link" href="{escape(home)}" target="_top">Recorded demo ↗</a>' if home else ""
    return ('<div class="lv-top"><div class="lv-logo">T</div><div><div class="lv-name">TrialGuide</div>'
            '<div class="lv-tag">Find clinical trials for a patient. It checks every rule and asks only what it must.</div>'
            f'</div><div class="lv-sp"></div>{badge}{back}</div>')


def how_it_works() -> str:
    """Three short steps shown before the first patient."""
    steps = [("Add your patient", "Paste a note, upload a file, or type a few facts."),
             ("TrialGuide checks the trials", "It finds matching trials and checks each rule against the note."),
             ("Answer a few questions", "Only what the note does not say. Each answer updates the trials.")]
    return '<div class="lv-how">' + "".join(
        f'<div><b>{i}</b><span>{escape(h)}</span><p>{escape(p)}</p></div>' for i, (h, p) in enumerate(steps, 1)) + "</div>"


def patient_bar(session) -> str:
    """One line naming the patient being checked (the first sentence of the note)."""
    first = next(iter(session.note.sentences.values()), "")
    first = first if len(first) < 140 else first[:137] + "…"
    return f'<div class="lv-patient">🩺 <b>Patient:</b> {escape(first)}</div>'


def _counts(session) -> dict:
    """Rule counts over all trials: total, settled, settled by the doctor, still open."""
    out = {"total": 0, "open": 0, "answered": 0}
    for t in session.trial_ids():
        c = session.counts(t)
        out["total"] += sum(c.values())
        out["open"] += c["open"]
        out["answered"] += c["answered"]
    out["settled"] = out["total"] - out["open"]
    return out


def tiles(session) -> str:
    """Four numbers: trials checked, rules settled, settled by you, questions left."""
    c = _counts(session)
    left = max(session.state.questions_left(), 0) if not session.finished else 0
    items = [(len(session.trial_ids()), "trials checked"),
             (f'{c["settled"]}<span style="font-size:16px;color:#94a3b8"> / {c["total"]}</span>', "rules settled"),
             (c["answered"], "settled by your answers"),
             (left, "questions left")]
    return '<div class="lv-tiles">' + "".join(
        f'<div class="lv-tile"><div class="v">{v}</div><div class="l">{escape(l)}</div></div>' for v, l in items) + "</div>"


def question(session) -> str:
    """The current question in a bold card, or a 'done' card when the conversation is over."""
    if session.finished or session.question is None:
        c = _counts(session)
        return ('<div class="lv-done"><div class="t">✓ All done</div>'
                f'<p>{len(session.turns)} question(s) answered. {c["open"]} rule(s) are still open: the note does not '
                'settle them, so they are left for the doctor and never guessed.</p>'
                '<p>Add new information below to re-check, or start over with a new patient.</p></div>')
    q = session.question
    trials = len({t for t, _ in q.rule_refs})
    kind = "Follow-up question" if q.is_followup else f"Question {q.turn} of {session.state.budget}"
    return (f'<div class="lv-q"><div class="k">{escape(kind)}</div><div class="t">{escape(q.text)}</div>'
            f'<div class="w">Your answer can settle {len(q.rule_refs)} rule(s) in {trials} trial(s).</div></div>')


def trials(session, record: dict) -> str:
    """Trial cards, best first: rank, title, link, how many rules are settled, and a status pill."""
    info = {t["trial_id"]: t for t in record["trials"]}
    against = render.COLOURS["against"]
    cards = []
    for rank, tid in enumerate(session.order(session.scores()), 1):
        c = session.counts(tid)
        total = sum(c.values()) or 1
        settled = total - c["open"]
        refs = [r for r in session.rules if r[0] == tid]
        out = sum(render._rule_colour(session, r) == against for r in refs)
        if out:
            pill = ("#fee2e2", "#b91c1c", f"Ruled out by {out} rule{'s' if out > 1 else ''}")
        elif c["open"] == 0:
            pill = ("#dcfce7", "#166534", "Meets every rule checked")
        else:
            pill = ("#fef3c7", "#92400e", f"{c['open']} rule{'s' if c['open'] > 1 else ''} still open")
        title = info.get(tid, {}).get("title", "") or tid
        cards.append(
            f'<div class="lv-trial"><div class="lv-rank">{rank}</div><div>'
            f'<div class="lv-title">{escape(title[:120])}</div>'
            f'<div class="lv-meta"><a href="https://clinicaltrials.gov/study/{escape(tid)}" target="_blank">{escape(tid)} ↗</a></div>'
            f'<div class="lv-bar"><i style="width:{max(settled / total * 100, 3):.0f}%"></i></div>'
            f'<div class="lv-row"><span>{settled} of {total} rules settled</span>'
            f'<span class="lv-pill" style="background:{pill[0]};color:{pill[1]}">{escape(pill[2])}</span></div>'
            "</div></div>")
    return '<div class="lv-h" style="font-size:17px">Matching trials</div>' + "".join(cards)


def conversation(session) -> str:
    """Questions and answers so far, each with how many rules it settled."""
    if not session.turns:
        return '<div class="lv-conv"><h4>Conversation</h4><div class="lv-muted">Your answers will appear here.</div></div>'
    rows = []
    for t in session.turns:
        n = len(t.settled)
        rows.append(f'<div class="lv-turn"><b>{escape(t.question)}</b><br>You: {escape(t.answer)} '
                    f'<span class="lv-muted">({escape(t.certainty)})</span> · '
                    f'<span class="r">{n} rule{"s" if n != 1 else ""} settled</span></div>')
    return '<div class="lv-conv"><h4>Conversation</h4>' + "".join(rows) + "</div>"
