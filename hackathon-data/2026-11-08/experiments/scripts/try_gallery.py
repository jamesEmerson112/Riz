"""Live gallery of every gentest try: one GIF each (BEFORE -> sampled frames / jaws close / probe -> AFTER), plus
index.html (newest first, refreshes itself). Rebuilds every 8 s; only new tries get a GIF. Run from prototypes/arm-sim."""
import html, json, sys, time
from pathlib import Path
import armsim.gifs as G

OUT = G.RUNS / "gallery"
OUT.mkdir(parents=True, exist_ok=True)


def gif_for(r):
    if r.get("success") or not r.get("ts"):
        return None
    name = f"{time.strftime('%H%M%S', time.localtime(r['ts']))}-{r.get('try')}.gif"
    out = OUT / name
    if out.exists():
        return name
    if r.get("probe"):
        steps = [(r.get("before"), "BEFORE"), (r.get("after"), "PROBE: jaws open at the grip point")]
    else:
        steps = G.sequence(r, "AFTER")
    frames = [G.frame(p, stage, r, 480) for p, stage in steps if G.existing(p)]
    return name if G.save(frames, out, 900 if len(frames) > 3 else 1200) else None


def page(cards):
    body = "\n".join(cards)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Arm tries</title><style>
body{{margin:0;padding:12px;background:#111;color:#eee;font:13px ui-monospace,Menlo,monospace}}
h1{{font-size:15px;margin:0 0 10px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
.c{{background:#1c1c1c;border-radius:8px;padding:6px}} .c img{{width:100%;border-radius:4px;display:block}}
.c p{{margin:6px 2px 2px}} .hold{{color:#7fd17f}} .big{{outline:2px solid #e5484d}}
</style></head><body><h1>Arm tries, newest first ({len(cards)}) . refreshes every 10 s</h1><div class="grid">{body}</div></body></html>"""


def build():
    rows = [r for r in G.rows() if r.get("ts")]
    cards = []
    for r in reversed(rows):
        name = gif_for(r)
        if not name:
            continue
        text, clock = G.caption(r)
        big = " big" if (r.get("holding_s") or 0) > 5 else ""
        cards.append(f'<div class="c{big}"><img src="{name}" loading="lazy"><p>{clock} . {html.escape(text)}</p></div>')
    (OUT / "index.html").write_text(page(cards))
    return len(cards)


if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        n = build()
        print(f"{time.strftime('%H:%M:%S')} gallery: {n} tries", flush=True)
        if once:
            break
        time.sleep(8)
