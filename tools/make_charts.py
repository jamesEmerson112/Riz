"""Draw the README's charts as static SVGs (standard library only).

    python tools/make_charts.py        # writes media/charts/9b_before_now.svg and runs_passed.svg

Every number is copied from README.md's "Measured in the sim" section; nothing is computed except
bar lengths. Each SVG carries its own light and dark colors (prefers-color-scheme) and paints its
own background in GitHub's page color, so it stays readable if the two schemes ever disagree.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "media" / "charts"

# The per-job table, the five jobs with a "9B, before this pass" cell:
# (job, before passed, before median calls, now passed, now median calls)
NINE_B = [
    ("Pick up the block and put it back", "2/2", 7, "3/3", 3),
    ("The same, with the block bumped up to 35 mm", "4/8", 24, "9/9", 4),
    ("The same, with a frozen wrist sensor", "2/2", 10, "9/9", 6),
    ('"Reach for the block" with a dead elbow: hand it to a person', "2/2", 4, "9/9", 3),
    ('"Wave at me" while a person has frozen the arm', "2/2", 2, "3/3", 1),
]
HIGHLIGHT = 1  # the bumped block

# Totals, grouped by the README's own wording: (section, [(model, passed, of)])
TOTALS = [
    ("On the final code", [
        ("Qwen3.5-9B", 63, 63),
        ("Qwen3.5-4B", 27, 27),
        ("gpt-oss-20b, text only", 26, 27),
        ("Qwen3.5-2B", 21, 27),
    ]),
    ("With thinking on (Qwen3.5's default)", [
        ("Qwen3.5-9B", 27, 30),
        ("Qwen3.5-4B", 29, 30),
    ]),
    ("After 87e77b1, the jobs that use arm_pick_up", [
        ("Qwen3.5-9B", 18, 18),
        ("Qwen3.5-4B", 18, 18),
        ("gpt-oss-20b, text only", 29, 30),
    ]),
]

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

# Light values first, then the dark steps (same blue ramp, chosen for the dark page).
STYLE = """
.bg{fill:#ffffff}
.t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#6e6d68}
.axis{stroke:#c3c2b7}.link{stroke:#c3c2b7}
.pale{fill:#86b6ef}.strong{fill:#256abf}
.ring{stroke:#ffffff}.gap{fill:#ffffff}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117}
.t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#9a998f}
.axis{stroke:#3d3d3a}.link{stroke:#4a4a46}
.pale{fill:#1c5cab}.strong{fill:#5598e7}
.ring{stroke:#0d1117}.gap{fill:#0d1117}
}
text{font-family:%s}
.b{font-weight:600}
""" % FONT


def svg(w, h, title, desc, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{escape(title)}</title>\n<desc id=\"desc\">{escape(desc)}</desc>\n"
        f"<style>{STYLE}</style>\n"
        f'<rect class="bg" width="{w}" height="{h}" rx="6"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def text(x, y, s, cls="t1", size=13, anchor="start", extra=""):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:g}" y="{y:g}" class="{cls}{extra}" font-size="{size}"{a}>{escape(s)}</text>'


def dot(cx, cy, cls):
    return f'<circle cx="{cx:g}" cy="{cy:g}" r="5.5" class="{cls} ring" stroke-width="2"/>'


def nine_b_chart():
    w, pad = 760, 24
    x0, x1, vmax = 44, 560, 25          # value 0 sits at x0, value 25 at x1
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    top, pitch = 136, 54
    axis_y = top + pitch * (len(NINE_B) - 1) + 22 + 26   # under the last row's dumbbell
    h = axis_y + 62

    body = [
        text(pad, 36, "The 9B now takes fewer tool calls on each job it ran before", size=17, extra=" b"),
        text(pad, 58, "Qwen3.5-9B, median tool calls per job: before this pass, and now (commit a4128c9).",
             "t2"),
        text(pad, 76, "The five jobs with a “before” cell in the table. Right: runs passed.", "t2"),
        dot(pad + 6, 98, "pale"), text(pad + 18, 102, "Before this pass", "t2", 12),
        dot(pad + 138, 98, "strong"), text(pad + 150, 102, "Now", "t2", 12),
        text(w - pad, 102, "Runs passed", "t2", 12, "end", " b"),
    ]
    for i, (job, bp, bc, np_, nc) in enumerate(NINE_B):
        y = top + pitch * i
        cy = y + 22
        hi = i == HIGHLIGHT
        body.append(text(pad, y + 2, job, "t1", 13, extra=" b" if hi else ""))
        body.append(f'<line x1="{sx(nc):g}" y1="{cy}" x2="{sx(bc):g}" y2="{cy}" class="link" '
                    f'stroke-width="2" stroke-linecap="round"/>')
        body.append(dot(sx(bc), cy, "pale"))
        body.append(dot(sx(nc), cy, "strong"))
        body.append(text(sx(nc) - 11, cy + 4, str(nc), "t1", 12, "end"))
        body.append(text(sx(bc) + 11, cy + 4, str(bc), "t2", 12))
        body.append(text(w - pad, cy + 4, f"{bp} → {np_}", "t1" if hi else "t2", 13, "end",
                         " b" if hi else ""))
    body.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" class="axis" stroke-width="1"/>')
    for v in range(0, vmax + 1, 5):
        body.append(f'<line x1="{sx(v):g}" y1="{axis_y}" x2="{sx(v):g}" y2="{axis_y + 4}" class="axis" '
                    f'stroke-width="1"/>')
        body.append(text(sx(v), axis_y + 18, str(v), "t3", 11, "middle"))
    body.append(text((x0 + x1) / 2, axis_y + 40, "Median tool calls", "t2", 12, "middle"))

    title = "Qwen3.5-9B before this pass and now: median tool calls per job"
    desc = "Median tool calls and runs passed, before this pass then now. " + "; ".join(
        f"{job}: {bc} calls, {bp} passed, then {nc} calls, {np_} passed" for job, bp, bc, np_, nc in NINE_B
    ) + "."
    return svg(w, h, title, desc, body)


def totals_chart():
    w, pad = 760, 24
    mx0, mx1, bar_h = 196, 640, 12
    y = 110
    body = [
        text(pad, 36, "Runs passed, by model", size=17, extra=" b"),
        text(pad, 58, "Bar length is the share of runs passed; the faint end is the runs missed.", "t2"),
        text(pad, 76, "Qwen3.5 ran with thinking off unless noted.", "t2"),
    ]
    clip_id = 0
    lines = []
    for section, rows in TOTALS:
        y += 6
        body.append(text(pad, y, section, "t2", 12, extra=" b"))
        y += 10
        for model, passed, of in rows:
            cy = y + 14
            by = cy - bar_h / 2
            fw = (mx1 - mx0) * passed / of
            clip_id += 1
            body.append(text(pad, cy + 4.5, model, "t1", 13))
            body.append(f'<clipPath id="c{clip_id}"><rect x="{mx0}" y="{by:g}" width="{mx1 - mx0}" '
                        f'height="{bar_h}" rx="4"/></clipPath>')
            body.append(f'<g clip-path="url(#c{clip_id})">'
                        f'<rect x="{mx0}" y="{by:g}" width="{mx1 - mx0}" height="{bar_h}" class="pale"/>'
                        f'<rect x="{mx0}" y="{by:g}" width="{fw:.2f}" height="{bar_h}" class="strong"/>'
                        + (f'<rect x="{mx0 + fw:.2f}" y="{by:g}" width="2" height="{bar_h}" class="gap"/>'
                           if passed < of else "")
                        + "</g>")
            body.append(text(mx1 + 16, cy + 4.5, f"{passed} of {of}", "t1", 13))
            lines.append(f"{section}, {model}: {passed} of {of}")
            y += 30
        y += 14
    h = y + 4
    title = "Runs passed, by model"
    desc = "Runs passed out of runs, per model. " + "; ".join(lines) + "."
    return svg(w, h, title, desc, body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("9b_before_now.svg", nine_b_chart), ("runs_passed.svg", totals_chart)):
        path = OUT / name
        path.write_text(make(), encoding="utf-8")
        print(f"{path.relative_to(OUT.parents[1])}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
