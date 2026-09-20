"""The HTML shell, stylesheet, and viewers a report carries.

All of it is ours: no CDN, no third-party embed — the file must open on
a locked-down machine with no network and audit cleanly. The scene
viewer is the desktop animator's phase math in an orthographic canvas;
the plot viewer is a wheel-zoom, drag-pan line plot with decade ticks.
"""

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<main id="report"></main>
<script id="data" type="application/json">__DATA__</script>
<script>__JS__</script>
</body>
</html>
"""

_CSS = """
:root { --ink: #1f2328; --faint: #6a737d; --line: #d0d7de;
        --accent: #4c92d9; --paper: #ffffff;
        color-scheme: light dark; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e6edf3; --faint: #8b949e; --line: #30363d;
          --paper: #000000; } }
/* the toggle's explicit choice outranks the OS preference */
:root[data-theme="light"] { --ink: #1f2328; --faint: #6a737d;
  --line: #d0d7de; --paper: #ffffff; color-scheme: light; }
:root[data-theme="dark"] { --ink: #e6edf3; --faint: #8b949e;
  --line: #30363d; --paper: #000000; color-scheme: dark; }
.themetoggle { position: fixed; top: 10px; right: 12px; z-index: 5;
  font: inherit; font-size: .8rem; color: var(--ink);
  background: var(--paper); border: 1px solid var(--line);
  border-radius: 6px; padding: .2rem .6rem; cursor: pointer;
  opacity: .85; }
.themetoggle:hover { opacity: 1; }
* { box-sizing: border-box; }
/* the classification banners: one marking, top and bottom, always in
   view — fixed elements also repeat on every printed page, which is
   what a marking is for */
.marking { position: fixed; left: 0; right: 0; z-index: 6;
  text-align: center; font-weight: 700; font-size: 1.05rem;
  letter-spacing: .12em; padding: .35rem 0; color: var(--ink);
  background: var(--paper); }
.marking.top { top: 0; border-bottom: 1px solid var(--line); }
.marking.bottom { bottom: 0; border-top: 1px solid var(--line); }
.marking.red { color: #d02b2b; }
.marking input { font: inherit; color: inherit; letter-spacing: inherit;
  text-align: center; background: none; border: none; outline: none;
  border-bottom: 1px dashed var(--line); min-width: 16rem; }
.marking select { font: inherit; font-size: .7rem; font-weight: 400;
  letter-spacing: normal; margin-left: .8rem; color: var(--ink);
  background: var(--paper); border: 1px solid var(--line);
  border-radius: 4px; }
body.marked { padding-top: 4.2rem; }
.marked .themetoggle { top: 2.6rem; }
body.marked::after { content: ""; display: block; height: 2.4rem; }
body { margin: 0 auto; padding: 2rem 1.5rem 4rem; max-width: 900px;
       background: var(--paper); color: var(--ink);
       font: 16px/1.55 -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.6rem; border-bottom: 1px solid var(--line);
     padding-bottom: .4rem; }
section { margin: 1.6rem 0; }
.caption { color: var(--faint); font-size: .9rem; margin-top: .35rem; }
.note { margin-top: .3rem; padding-left: .6rem;
        border-left: 3px solid var(--line); font-style: italic; }
.text { white-space: pre-wrap; }
canvas { width: 100%; border: 1px solid var(--line); border-radius: 6px;
         display: block; touch-action: none; }
.legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem;
          font-size: .8rem; margin-top: .3rem; }
.legend span::before { content: "—"; font-weight: 700; margin-right: .3rem;
                       color: var(--swatch); }
.legend span.dashed::before { content: "╌"; }
.controls { display: flex; gap: .6rem; align-items: center;
            margin-top: .3rem; font-size: .85rem; }
.controls button, .controls select {
  font: inherit; color: var(--ink); background: transparent;
  border: 1px solid var(--line); border-radius: 5px; padding: .15rem .6rem; }
.hint { color: var(--faint); font-size: .75rem; }
.modeinfo { position: absolute; top: 8px; left: 12px; font-size: .8rem;
            color: var(--ink); pointer-events: none; line-height: 1.35;
            background: color-mix(in srgb, var(--paper) 65%, transparent);
            border-radius: 5px; padding: .15rem .4rem; }
option { background: var(--paper); color: var(--ink); }
img.photo { max-width: 100%; border-radius: 6px; display: block; }
table { border-collapse: collapse; font-size: .85rem; width: 100%; }
/* a row is one line: the columns take the width of what is in them —
   a channel table's comments wrapped every row onto four (Brandon,
   2026-09-18) — and a table wider than the page scrolls in its wrap */
th, td { border: 1px solid var(--line); padding: .25rem .55rem;
         text-align: left; white-space: nowrap; }
.tablewrap { overflow-x: auto; }
/* the pass/fail box: its two colors are the box's own, one for each
   word, and are not the exceedance red — a verdict is not a mark */
.verdict { border-radius: 8px; padding: 1rem 1.25rem; border: 2px solid;
           line-height: 1.5; }
.verdict.pass { border-color: #2e8b57;
                background: color-mix(in srgb, #2e8b57 16%, var(--paper)); }
.verdict.fail { border-color: #c0392b;
                background: color-mix(in srgb, #c0392b 16%, var(--paper)); }
.verdict .word { font-size: 2rem; font-weight: 700; letter-spacing: .08em;
                 margin-bottom: .3rem; }
.verdict.pass .word { color: #2e8b57; }
.verdict.fail .word { color: #c0392b; }
/* the grid figure: headings across, node labels down, a cell's own
   legend, controls and caption at a size that leaves room for the plot */
.grid { display: grid; gap: .5rem .6rem; align-items: start; }
.gridhead { font-size: .85rem; font-weight: 600; text-align: center; }
.gridrow { font-size: .85rem; font-weight: 600; padding-top: .3rem;
           white-space: nowrap; }
.gridcell { min-width: 0; }
.gridcell .legend, .gridcell .controls, .gridcell .caption {
  font-size: .72rem; margin-top: .2rem; }
.gridcell .controls button { padding: 0 .4rem; }
th { background: color-mix(in srgb, var(--line) 30%, transparent); }
/* a cell the app marks is marked here too, in the same two colors the
   plots shade an exceedance with: a channel red on screen is red on the
   page. Both read on either theme, which is why they are translucent
   rather than flat. */
td.mark-over { background: rgba(229, 83, 75, 0.30); }
td.mark-under { background: rgba(76, 146, 217, 0.30); }
@media print { canvas { break-inside: avoid; } }
"""

_JS = r"""
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const COLORS = ['#4c92d9','#ff8c2b','#3fb950','#e5534b','#a371f7',
                '#b07d62','#e668c3','#8b949e','#d2c14e','#39c5cf'];
const root = document.getElementById('report');
const h1 = document.createElement('h1');
h1.textContent = DATA.title; root.appendChild(h1);
/* the classification marking: one value, two banners. Blank means
   unmarked and draws nothing — edit mode replaces these with the
   editable pair, blank included, so the marking stays reachable. */
function markingStrips(build) {
  document.querySelectorAll('.marking').forEach(e => e.remove());
  document.body.classList.remove('marked');
  const strips = [];
  if (build) {
    for (const place of ['top', 'bottom']) {
      const strip = document.createElement('div');
      strip.className = 'marking ' + place
        + (DATA.marking_color === 'red' ? ' red' : '');
      document.body.appendChild(strip); strips.push(strip);
    }
    document.body.classList.add('marked');
  }
  return strips;
}
// in both modes: the editor's page used to build its own strips with a
// text box in each, and losing that chrome (2026-09-08) lost the strips
// with it — the marking is typed in the pane now, and the page shows it
// the way the export will
if (DATA.marking)
  markingStrips(true).forEach(s => s.textContent = DATA.marking);
// every viewer registers its draw here, so a theme flip repaints the
// canvases in the new ink
const REDRAWS = [];

function captionFor(block, extra) {
  if (!block.label && !block.caption && !extra && !block.note) return null;
  const c = document.createElement('div'); c.className = 'caption';
  if (block.label) { const strong = document.createElement('strong');
    strong.textContent = block.label + (block.caption ? ': ' : '');
    c.appendChild(strong); }
  c.appendChild(document.createTextNode(
    (block.caption || '') + (extra || '')));
  /* a `note` is something the figure cannot show about itself — an
     overlay normalizes each shape to its own peak, so a scale
     difference between the two sets is invisible in the picture and
     has to be said in words. Set apart from the caption, because the
     caption says what the figure is and this says what to watch for. */
  if (block.note) {
    const n = document.createElement('div'); n.className = 'note';
    n.textContent = block.note;
    c.appendChild(n);
  }
  return c;
}
function section(block, extra) {
  const s = document.createElement('section'); root.appendChild(s);
  const c = block ? captionFor(block, extra) : null;
  if (c) s.appendChild(c);
  return s;
}
function sized(canvas, height) {
  const scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 850;
  canvas.width = width * scale; canvas.height = height * scale;
  canvas.style.height = height + 'px';
  const g = canvas.getContext('2d'); g.scale(scale, scale);
  return [g, width, height];
}
/* the one color scale, handed over rather than carried: DATA.viridis
   is `theme.VIRIDIS`, the very stops the app paints a moving model by
   and draws its MAC grids on */
const viridis = t => {
  const stops = DATA.viridis;
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(Math.floor(t), stops.length - 2), f = t - i;
  return 'rgb(' + stops[i].map((v, k) =>
    Math.round(v + f * (stops[i + 1][k] - v))).join(',') + ')';
};

/* The averaging marks' two colors: the app's own theme values, sent
   with the page (`DATA.marks_color`) rather than restated here. Read
   per draw and per side, so a theme flip repaints them — and shared
   by the flat figure and the stage, which draw the same marks. */
const darkPage = () => (document.documentElement.dataset.theme
  || (matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light')) === 'dark';
const rgbOf = hex => [1, 3, 5].map(
  at => parseInt(hex.slice(at, at + 2), 16)).join(', ');
const markColor = which => rgbOf(
  DATA.marks_color[darkPage() ? 'dark' : 'light'][which]);
const markOrange = () => markColor('window');
const markBlue = () => markColor('band');

/* ---- the view a 3-D figure is looked at from ---------------------------
   An orthographic basis plus the gestures that move it: drag to rotate,
   wheel to zoom. Shared by the geometry scenes and the data stage, so a
   reader who has learned to turn one figure has learned them all — and
   so the two cannot drift into two feels. */
const normalized = v => { const l = Math.hypot(...v) || 1;
                          return v.map(a => a / l); };
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1],
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0]];

function orbitView(canvas, homeBasis, redraw) {
  const home = homeBasis.map(v => v.slice());
  const view = {
    basis: home.map(v => v.slice()), zoom: 1,
    home() { view.basis = home.map(v => v.slice()); view.zoom = 1; },
    /* the view normal: what depth sorting and flat shading read */
    normal: () => cross3(view.basis[0], view.basis[1]),
    /* one point, world to canvas, about a center and at a scale the
       caller owns — a scene fits its model, a stage fits its box */
    project: (p, center, scale, width, height) => {
      const d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
      return [width / 2 + scale * (d[0] * view.basis[0][0]
                                   + d[1] * view.basis[0][1]
                                   + d[2] * view.basis[0][2]),
              height / 2 - scale * (d[0] * view.basis[1][0]
                                    + d[1] * view.basis[1][1]
                                    + d[2] * view.basis[1][2])];
    }};
  function rotate(axis, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    view.basis = view.basis.map(v => {
      const dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
      const perp = cross3(axis, v);
      return [0, 1, 2].map(a =>
        v[a] * c + perp[a] * s + axis[a] * dot * (1 - c));
    });
  }
  let dragging = null;
  canvas.addEventListener('pointerdown', e => {
    dragging = [e.clientX, e.clientY];
    canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    rotate(view.basis[1], -(e.clientX - dragging[0]) * 0.008);
    rotate(view.basis[0], -(e.clientY - dragging[1]) * 0.008);
    dragging = [e.clientX, e.clientY];
    redraw(); });
  canvas.addEventListener('pointerup', () => dragging = null);
  canvas.addEventListener('wheel', e => { e.preventDefault();
    view.zoom *= Math.exp(-e.deltaY * 0.002);
    redraw(); }, { passive: false });
  return view;
}

/* tick positions and their labels — shared, because the bar charts read
   their value axis the same way the line plots read theirs, and two
   implementations would drift into two conventions */
function ticks(lo, hi, log) {
  if (log) { const out = [];
    for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) out.push(d);
    if (out.length >= 2) return out; }
  const span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span)));
  const fine = span / step >= 5 ? step : span / step >= 2 ? step / 2
                                      : step / 5;
  const out = [];
  for (let v = Math.ceil(lo / fine) * fine; v <= hi; v += fine)
    out.push(v);
  return out;
}
/* A log tick's value is its exponent, so a whole one is 1e<n>. Under
   two decades of range there are no whole ones to put ticks on and
   `ticks` steps by fifths of a decade instead — those are real
   positions and get their real values, because rounding them to the
   nearest exponent labeled four different heights '1e-9' (Brandon,
   2026-08-23, on the system ID densities). */
const label = (v, log) => log
  ? (Math.abs(v - Math.round(v)) < 1e-9 ? '1e' + Math.round(v)
                                        : Math.pow(10, v).toExponential(1))
  : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
    ? v.toExponential(1) : String(Math.round(v * 1000) / 1000);

/* ---- a bar per control channel, with a threshold you can drag ------- */
function barsBlock(block) {
  const s = document.createElement('section'); root.appendChild(s);
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  /* named, so anything asking whether the bar charts have drawn can say
     which canvases those are rather than counting from the end of a page
     that is still being built */
  canvas.className = 'bars';
  const c = captionFor(block); if (c) s.appendChild(c);
  /* channels down the side, not along the bottom: a channel is named
     '101Z+', and a column of names is legible where a row of them is a
     smear of overlapping ticks. One axis however many there are — the
     chart grows taller, which a page handles perfectly well. */
  const ROW_HEIGHT = 18;
  const margin = { left: 78, right: 14, top: 26, bottom: 34 };
  /* a tolerance is quoted to a tenth of a dB or a tenth of a percent, so
     a dragged threshold lands on one */
  const SNAP = 0.1;
  const snap = v => Math.round(Math.round(v / SNAP) * SNAP * 1e10) / 1e10;
  /* the thresholds live on the block, so dragging one and toggling the
     page theme does not put it back where it started */
  /* fixed, wherever the app was left: they are read here, not set */
  block.low_at = block.low;
  block.high_at = block.high;

  const beyond = v => {
    if (v === null || !isFinite(v)) return null;
    if (block.high_at === null) return v >= block.low_at ? 'over' : null;
    if (v > block.high_at) return 'over';
    if (v < block.low_at) return 'under';
    return null;
  };

  function draw() {
    const values = block.values;
    const [g, width, height] = sized(
      canvas, margin.top + margin.bottom + values.length * ROW_HEIGHT);
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const ink = getComputedStyle(document.body).color;
    /* where a bar grows from: zero for a reading whose zero means no
       error at all, the nominal for one that has a nominal — a
       kurtosis bar drawn from zero is three units of agreement
       dressed up as a measurement (Brandon, 2026-08-24) */
    const base = block.baseline === undefined ? 0 : block.baseline;
    let lo = Math.min(base, ...values, block.low_at);
    let hi = Math.max(base, ...values, block.high_at === null
                                       ? block.low_at : block.high_at);
    const pad = 0.12 * (hi - lo || 1); lo -= pad; hi += pad;
    if (block.floor !== null && block.floor !== undefined) {
      lo = Math.min(block.floor, hi - 1e-9);
    }
    /* the value axis is shared, so bars stay comparable between sets */
    const px = v => margin.left + (v - lo) / (hi - lo) * plotW;
    g.clearRect(0, 0, width, height);
    {
      const first = 0;
      const block_ = values;
      const top = margin.top;
      const step = plotH / Math.max(block_.length, 1);
      g.strokeStyle = ink; g.globalAlpha = 0.35; g.lineWidth = 1;
      g.beginPath();
      g.moveTo(px(base), top); g.lineTo(px(base), top + plotH);
      g.stroke(); g.globalAlpha = 1;

      block_.forEach((v, k) => {
        const which = beyond(v);
        g.fillStyle = which === 'over' ? 'rgba(229, 83, 75, 0.85)'
                    : which === 'under' ? 'rgba(76, 146, 217, 0.85)'
                    : 'rgba(140, 140, 148, 0.75)';
        const Y = top + step * (k + 0.16), H = step * 0.68;
        const X0 = px(base), X = px(v);
        g.fillRect(Math.min(X, X0), Y, Math.abs(X - X0), H);
        g.fillStyle = ink; g.globalAlpha = 0.8;
        g.font = '10px sans-serif'; g.textAlign = 'right';
        g.fillText(block.labels[first + k], margin.left - 6, Y + H * 0.8);
        g.globalAlpha = 1;
      });

      /* the thresholds as shaded ground rather than as lines: past here
         is out, and a filled region says it where a line leaves it to be
         inferred. Red above, blue below — the same two colors, at the
         same weight, the specification plot shades its abort zones with.
         A one-sided chart has only a ceiling, so its one threshold is
         the red one. */
      [block.low_at, block.high_at].forEach((at, j) => {
        if (at === null || at === undefined) return;
        const X = px(at);
        const over = j === 1 || block.high_at === null
                     || block.high_at === undefined;
        g.fillStyle = over ? 'rgba(229, 83, 75, 0.16)'
                           : 'rgba(76, 146, 217, 0.16)';
        if (over) g.fillRect(X, top, margin.left + plotW - X, plotH);
        else g.fillRect(margin.left, top, X - margin.left, plotH);
        g.fillStyle = over ? '#e5534b' : '#4c92d9';
        g.font = '10px sans-serif'; g.textAlign = 'center';
        g.fillText(at.toFixed(1) + block.units, X, top - 4);

      });
      g.strokeStyle = ink; g.globalAlpha = 0.45;
      g.strokeRect(margin.left, top, plotW, plotH); g.globalAlpha = 1;

      /* the value axis: a grid line, a tick and a label at each step,
         and the quantity named underneath. Without them the bars are
         lengths with nothing to read them against. */
      ticks(lo, hi, false).forEach(t => {
        if (t < lo || t > hi) return;
        const X = px(t);
        g.strokeStyle = ink; g.globalAlpha = 0.14; g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X, top); g.lineTo(X, top + plotH); g.stroke();
        g.globalAlpha = 0.55;
        g.beginPath();
        g.moveTo(X, top + plotH); g.lineTo(X, top + plotH + 4); g.stroke();
        g.globalAlpha = 0.85;
        g.fillStyle = ink; g.font = '10px sans-serif'; g.textAlign = 'center';
        g.fillText(label(t, false), X, top + plotH + 15);
        g.globalAlpha = 1;
      });
      g.fillStyle = ink; g.globalAlpha = 0.85;
      g.font = '11px sans-serif'; g.textAlign = 'center';
      g.fillText(block.ylabel, margin.left + plotW / 2, top + plotH + 29);
      g.globalAlpha = 1;
    }


    const out = values.filter(v => beyond(v)).length;
    const share = values.length ? 100 * out / values.length : 0;
    const where = block.high_at === null
      ? 'over ' + block.low_at.toFixed(1) + block.units
      : 'outside ' + block.low_at.toFixed(1) + ' to '
        + block.high_at.toFixed(1) + block.units;
    g.fillStyle = ink; g.font = '12px sans-serif'; g.textAlign = 'center';
    g.fillText(out + ' of ' + values.length + ' channels ' + where
               + ' — ' + share.toFixed(0) + '%',
               margin.left + plotW / 2, 14);
    /* how many bars this canvas carries, stamped once the drawing has
       actually got to the end of them. A report that comes back blank
       is otherwise a canvas with nothing on it and nothing to say why:
       the mark distinguishes "the drawing threw" from "the drawing ran
       and the data was empty", which are different faults. Reading the
       pixels back cannot answer that — a canvas on a machine with no
       working GPU path hands back a fully transparent buffer however
       well it drew. */
    canvas.dataset.bars = values.length;
  }

  /* No dragging here. A threshold is a judgment, and the place to
     make it is the app, where the data is in front of you; a report is
     what that judgment produced, and a reader who could quietly move
     the line would be reading a different document from the one that
     was written. */
  REDRAWS.push(draw);
  draw();
}

/* ---- zoomable line plot ------------------------------------------------ */
function controlBar(s, block, hintText, onChannel) {
  /* the row under a figure: a channel pick when the block holds more
     than one and the caller draws one at a time, the Reset Axes
     button, and the hint. Returns the button for the caller to wire.
     One function for the plot and the MAC: the MAC's copy once kept
     a channel pick that called the plot's own applyChannel (a
     ReferenceError waiting for a MAC with channels, 2026-09-12). */
  const controls = document.createElement('div');
  controls.className = 'controls';
  if (onChannel && block.channels && block.channels.length > 1) {
    const pick = document.createElement('select');
    pick.className = 'channel';
    block.channels.forEach((ch, i) => {
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = ch.label; pick.appendChild(opt); });
    pick.addEventListener('change', () => onChannel(+pick.value));
    controls.appendChild(pick);
  }
  const reset = document.createElement('button');
  reset.className = 'reset'; reset.textContent = 'Reset Axes';
  controls.appendChild(reset);
  const hint = document.createElement('span'); hint.className = 'hint';
  hint.textContent = hintText;
  controls.appendChild(hint);
  s.appendChild(controls);
  return reset;
}

function plotBlock(block, into) {
  /* `into`: a cell of a grid figure to draw in, rather than a section
     of the page's own — the same figure, smaller, with the pan/zoom
     hint left to the grid's caption (a cell is not a section: edit
     mode maps sections to blocks by position) */
  const s = into || document.createElement('section');
  if (!into) root.appendChild(s);
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  const legend = document.createElement('div'); legend.className = 'legend';
  function fillLegend() {
    legend.textContent = '';
    block.curves.slice(0, 12).forEach((curve, i) => {
      const item = document.createElement('span');
      const which = curve.color === undefined ? i : curve.color;
      item.style.setProperty('--swatch', COLORS[which % COLORS.length]);
      if (curve.dash) item.className = 'dashed'; // the swatch says so too
      item.textContent = curve.label; legend.appendChild(item); });
  }
  fillLegend();
  s.appendChild(legend);
  /* one control channel is drawn and the rest are a pick away: six
     targets and two dozen limit lines stacked together are unreadable,
     which is the same reason the app draws one at a time */
  let picked = 0;
  const reset = controlBar(s, block, into ? '' :
    'drag to pan, wheel to zoom — over an axis, that axis alone; '
    + 'double-click also resets',
    i => { picked = i; applyChannel(); draw(); });
  const c = captionFor(block);
  if (c) s.appendChild(c);

  const finite = v => v !== null && isFinite(v);
  /* the drawn curves become whichever channel is picked; the extents
     below still span every channel, so switching never leaves the view */
  function applyChannel() {
    if (!block.channels) return;
    const ch = block.channels[picked];
    block.curves[0].y = ch.y;
    block.curves[0].label = ch.label;
    if (ch.response && block.curves[1]) block.curves[1].y = ch.response;
    if (ch.responses) {
      /* a channel carrying several measured curves — a shock series'
         events over one specification — replaces everything after the
         target, and the legend follows */
      block.curves.length = 1;
      ch.responses.forEach(r =>
        block.curves.push({label: r.label, x: null, y: r.y}));
      fillLegend();
      return;
    }
    legend.querySelectorAll('span')[0].textContent = ch.label;
  }
  applyChannel();
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  (block.channels || []).forEach(ch => {
    ch.y.forEach(v => { if (finite(v)) { y0 = Math.min(y0, v);
                                         y1 = Math.max(y1, v); } });
    (ch.response || []).forEach(v => { if (finite(v)) {
      y0 = Math.min(y0, v); y1 = Math.max(y1, v); } });
    (ch.responses || []).forEach(r => r.y.forEach(v => { if (finite(v)) {
      y0 = Math.min(y0, v); y1 = Math.max(y1, v); } }));
    ch.zones.forEach(z => [z.lower, z.upper].forEach(edge =>
      (edge || []).forEach(v => { if (finite(v)) {
        y0 = Math.min(y0, v); y1 = Math.max(y1, v); } })));
  });
  block.curves.forEach(curve => {
    const xs = curve.x || block.x;
    xs.forEach(v => { if (finite(v)) { x0 = Math.min(x0, v);
                                       x1 = Math.max(x1, v); } });
    curve.y.forEach(v => { if (finite(v)) { y0 = Math.min(y0, v);
                                            y1 = Math.max(y1, v); } });
  });
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  // the data's own bounding box is the outer limit: zoom and pan roam
  // inside it, never beyond it
  const full = { x0, x1, y0: y0 - 0.05 * (y1 - y0),
                 y1: y1 + 0.05 * (y1 - y0) };
  const home = Object.assign({}, full);
  if (block.home_x) {
    // the block brought its own opening window (the CMIF's mode band);
    // y then fits only the data visible inside it
    home.x0 = block.home_x[0]; home.x1 = block.home_x[1];
    let lo = Infinity, hi = -Infinity;
    block.curves.forEach(curve => {
      const xs = curve.x || block.x;
      for (let i = 0; i < curve.y.length; i++) {
        if (!finite(curve.y[i]) || !finite(xs[i])) continue;
        if (xs[i] < home.x0 || xs[i] > home.x1) continue;
        lo = Math.min(lo, curve.y[i]); hi = Math.max(hi, curve.y[i]);
      }
    });
    if (hi > lo) { home.y0 = lo - 0.05 * (hi - lo);
                   home.y1 = hi + 0.05 * (hi - lo); }
  }
  if (block.averaging) {
    /* the app's make_room: the rail sits above the trace, not on it,
       so the data keeps its span and the ceiling rises until the
       trace holds only the share the rail leaves (Brandon,
       2026-08-23: the figure should match the GUI view). `foot` is
       that share, worked out by `Averaging.rail` — the app's own
       overlay asks the same question of the same object. */
    const foot = block.averaging.rail.foot;
    const grow = (home.y1 - home.y0) * (1 - foot) / foot;
    home.y1 += grow; full.y1 = Math.max(full.y1, home.y1);
  }
  canvas.dataset.home = JSON.stringify(home);
  canvas.dataset.full = JSON.stringify(full);
  let view = Object.assign({}, home);
  const margin = { left: 64, right: 12, top: 10, bottom: 40 };
  function clampView() {
    const spanX = Math.min(view.x1 - view.x0, full.x1 - full.x0);
    view.x0 = Math.max(full.x0, Math.min(view.x0, full.x1 - spanX));
    view.x1 = view.x0 + spanX;
    const spanY = Math.min(view.y1 - view.y0, full.y1 - full.y0);
    view.y0 = Math.max(full.y0, Math.min(view.y0, full.y1 - spanY));
    view.y1 = view.y0 + spanY;
    canvas.dataset.view = JSON.stringify(view);
  }
  clampView();

  function draw() {
    const [g, width, height] = sized(canvas, into ? 220 : 340);
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const px = v => margin.left + (v - view.x0) / (view.x1 - view.x0) * plotW;
    const py = v => margin.top + (view.y1 - v) / (view.y1 - view.y0) * plotH;
    const ink = getComputedStyle(document.body).color;
    g.clearRect(0, 0, width, height);
    g.save(); g.beginPath();
    g.rect(margin.left, margin.top, plotW, plotH); g.clip();
    /* the frames a PSD would be averaged over, drawn the way the app
       draws them: a column per frame under the window that shapes it,
       so the figure says which part of the run was analyzed and how.
       A trace on its own says neither. */
    if (block.averaging) {
      const a = block.averaging;
      const bottom = margin.top + plotH;
      /* one slot per level: overlapping frames cannot share a slot
         without their windows running through each other, which is
         what put every frame on top of every other one. The geometry
         arrives worked out (`Averaging.rail`), the very numbers the
         app's own overlay draws with, so the printed figure and the
         screen agree in proportion and not only in spirit. */
      const rail = a.rail;
      const glyphF = rail.glyph;
      const orange = markOrange(), blue = markBlue();
      const wAt = t => {
        const at = t * (a.window.length - 1), i = Math.floor(at);
        const next = a.window[Math.min(i + 1, a.window.length - 1)];
        return a.window[i] + (next - a.window[i]) * (at - i);
      };
      a.frames.forEach((span, k) => {
        const X0 = px(span[0]), X1 = px(span[1]);
        const baseline = margin.top
                       + plotH * (1 - rail.baselines[k]);
        /* the band falls from this frame's own slot to the foot of
           the plot, fading across with the window — the shading is
           the weight each moment of the record actually carries, and
           where two frames overlap the shadings add as the averaging
           does (BAND_PEAK_ALPHA 64/255) */
        const wash = g.createLinearGradient(X0, 0, X1, 0);
        for (let s = 0; s <= 48; s++) {
          const t = s / 48;
          wash.addColorStop(t, 'rgba(' + blue + ', '
            + (rail.band_alpha * Math.max(wAt(t), 0)).toFixed(3) + ')');
        }
        g.fillStyle = wash;
        g.fillRect(X0, baseline, X1 - X0, bottom - baseline);
        /* and the window that shapes it, on its own slot — the app's
           orange, filled at the overlay's alpha (GLYPH_ALPHA 110/255)
           and outlined at full strength */
        g.fillStyle = 'rgba(' + orange + ', ' + rail.glyph_alpha + ')';
        g.strokeStyle = 'rgb(' + orange + ')';
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X0, baseline);
        a.window.forEach((w, i) => {
          const t = i / (a.window.length - 1);
          g.lineTo(X0 + (X1 - X0) * t, baseline - plotH * glyphF * w);
        });
        g.lineTo(X1, baseline);
        g.closePath(); g.fill(); g.stroke();
        /* the tick under each end: a tapering window touches its
           baseline at both ends and would otherwise say nothing about
           where it stopped */
        const cap = plotH * rail.cap;
        g.beginPath();
        g.moveTo(X0, baseline - cap); g.lineTo(X0, baseline + cap);
        g.moveTo(X1, baseline - cap); g.lineTo(X1, baseline + cap);
        g.stroke();
      });
    }
    /* The windows each shock's spectrum was computed from. A shock
       trace is four bumps in a noise floor; without these it says
       nothing about which stretches the spectra downstream came from,
       and the numbers are the only way 'shock 2' points at anything.
       Colors and numbering copy `plot/shocks.py` deliberately — the
       reader has just been looking at the same bands in the app, and
       a report that recolors them reads as a different measurement.
       Full height rather than the averaging view's rows: shock
       windows never overlap (`suggest` holds them apart), so there is
       nothing to stack. */
    if (block.shocks) {
      const top = margin.top, height = plotH;
      block.shocks.windows.forEach((span, k) => {
        const X0 = px(span[0]), X1 = px(span[1]);
        g.fillStyle = 'rgba(76, 146, 217, 0.18)';
        g.fillRect(X0, top, X1 - X0, height);
        g.strokeStyle = 'rgba(194, 65, 12, 0.59)';
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(X0, top); g.lineTo(X0, top + height);
        g.moveTo(X1, top); g.lineTo(X1, top + height);
        g.stroke();
        g.fillStyle = ink; g.globalAlpha = 0.85;
        g.font = '11px sans-serif';
        g.textAlign = 'center';
        g.fillText(String(k + 1), (X0 + X1) / 2, top + 12);
        g.globalAlpha = 1; g.textAlign = 'left';
      });
    }
    /* the zones a response must not be in, shaded behind everything:
       warm between warning and abort, hot past abort, read the same way
       above the target and below it. A null edge is the edge of the
       plot — the zone past abort has no far side. */
    if (block.channels) {
      const xs = block.curves[0].x || block.x;
      /* the zones step with the target they bound: on a banded or a
         lines specification each edge is flat across its bin, on the
         same edges the target steps on (2026-09-19) */
      const zoneSteps = block.curves[0].steps !== undefined
        ? block.curves[0].steps : block.steps;
      const zoneEdges = block.curves[0].edges
        || (block.curves[0].x ? null : block.edges);
      const stepped = !!(zoneSteps && zoneEdges);
      const xAt = (i, side) => stepped
        ? px(zoneEdges[i + side]) : px(xs[i]);
      block.channels[picked].zones.forEach(zone => {
        /* past abort, and which way: red above, blue below — the same
           reading every other mark on the plot gives */
        g.fillStyle = zone.severity !== 'abort'
          ? 'rgba(210, 193, 78, 0.16)'
          : zone.upper === null ? 'rgba(229, 83, 75, 0.16)'
          : 'rgba(76, 146, 217, 0.16)';
        const top = margin.top, bottom = margin.top + plotH;
        let open = false;
        g.beginPath();
        for (let i = 0; i < xs.length; i++) {
          const lo = zone.lower ? zone.lower[i] : null;
          const hi = zone.upper ? zone.upper[i] : null;
          const bad = (zone.lower && !finite(lo))
                   || (zone.upper && !finite(hi));
          if (bad || !finite(xs[i])) { open = false; continue; }
          const YL = zone.lower ? py(lo) : bottom;
          const YU = zone.upper ? py(hi) : top;
          if (!open) { g.moveTo(xAt(i, 0), YL); open = true; }
          g.lineTo(xAt(i, 0), YL);
          if (stepped) g.lineTo(xAt(i, 1), YL);
        }
        for (let i = xs.length - 1; i >= 0; i--) {
          const lo = zone.lower ? zone.lower[i] : null;
          const hi = zone.upper ? zone.upper[i] : null;
          if ((zone.lower && !finite(lo)) || (zone.upper && !finite(hi))
              || !finite(xs[i])) continue;
          const YU = zone.upper ? py(hi) : top;
          if (stepped) g.lineTo(xAt(i, 1), YU);
          g.lineTo(xAt(i, 0), YU);
        }
        g.closePath(); g.fill();
      });
    }
    /* every line that went outside an abort limit, from the limit to
       the edge of the plot: red above, blue below. Which way it went is
       the first thing to know, and a stripe running off the plot is
       seen at a glance where a few pixels of height is not. */
    if (block.channels) {
      const ch = block.channels[picked];
      /* the marks stand on the measurement's lines — the block's grid —
         whatever grid the target is drawn on */
      const xs = block.x;
      const top = margin.top, bottom = margin.top + plotH;
      [['over', 'rgba(229, 83, 75, 0.60)', top],
       ['under', 'rgba(76, 146, 217, 0.60)', bottom]].forEach(
        ([key, color, edge]) => {
          const marks = ch[key];
          if (!marks) return;
          g.fillStyle = color;
          for (let i = 0; i < marks.length; i++) {
            if (!finite(marks[i]) || !finite(xs[i])) continue;
            /* the bin this line stands for, half a width either side */
            const before = i > 0 ? xs[i - 1] : xs[i];
            const after = i < xs.length - 1 ? xs[i + 1] : xs[i];
            const X0 = px((before + xs[i]) / 2), X1 = px((xs[i] + after) / 2);
            const Y = py(marks[i]);
            g.fillRect(X0, Math.min(Y, edge), Math.max(X1 - X0, 1),
                       Math.abs(edge - Y));
          }
        });
    }
    block.curves.forEach((curve, index) => {
      const xs = curve.x || block.x;
      const which = curve.color === undefined ? index : curve.color;
      /* the reference is gray and what is being looked at is the page's
         own ink, exactly as the app colors the pair */
      g.strokeStyle = curve.ink ? ink
                    : curve.gray ? '#9a9aa2'
                    : COLORS[which % COLORS.length];
      g.setLineDash(curve.dash ? [6, 4] : []);
      g.lineWidth = 1.2; g.beginPath();
      let pen = false;
      /* a curve drawn on its own grid brings its own shape and edges —
         a banded specification stepped on its own bands beside a
         response stepped on the measurement's (2026-09-18) */
      const steps = curve.steps !== undefined ? curve.steps : block.steps;
      const edges = curve.edges || (curve.x ? null : block.edges);
      for (let i = 0; i < curve.y.length; i++) {
        if (!finite(curve.y[i]) || !finite(xs[i])) { pen = false; continue; }
        const Y = py(curve.y[i]);
        if (steps) {
          /* A density is flat across its own bin: a polyline through
             the line centers draws a slope that is not in the data,
             and the area drawn stops being the area summed.

             The edges come from the payload, computed by the same
             `bin_edges` the app steps on. They used to be guessed here
             as midpoints between neighboring centers, which is right
             only for a spectrum that does not know its own widths — an
             octave band's center is the geometric mean of its edges,
             so midpoints missed them by several percent of a band, and
             the first and last bins came out half width. */
          const e = edges;
          const X0 = px(e ? e[i] : (i > 0 ? (xs[i - 1] + xs[i]) / 2 : xs[i]));
          const X1 = px(e ? e[i + 1]
                          : (i < xs.length - 1 ? (xs[i] + xs[i + 1]) / 2
                                               : xs[i]));
          pen ? g.lineTo(X0, Y) : g.moveTo(X0, Y);
          g.lineTo(X1, Y); pen = true;
        } else {
          const X = px(xs[i]);
          pen ? g.lineTo(X, Y) : g.moveTo(X, Y); pen = true;
        }
      }
      g.stroke();
    });
    g.setLineDash([]);
    // mode bookmarks: a dashed line and staggered frequency label at
    // every mode the synthesis carries, in the page's own ink — white
    // on dark, black on light — so they read apart from the grid
    (block.marks || []).forEach((f, index) => {
      const X = px(f);
      if (X < margin.left || X > margin.left + plotW) return;
      g.strokeStyle = ink; g.globalAlpha = 0.5;
      g.setLineDash([5, 4]); g.lineWidth = 1; g.beginPath();
      g.moveTo(X, margin.top); g.lineTo(X, margin.top + plotH);
      g.stroke(); g.setLineDash([]);
      g.fillStyle = ink; g.globalAlpha = 0.8;
      g.font = '10px sans-serif'; g.textAlign = 'left';
      g.fillText(f.toFixed(2), X + 3,
                 margin.top + 11 + 11 * (index % 3));
      g.globalAlpha = 1;
    });
    g.restore();
    g.strokeStyle = ink; g.globalAlpha = 0.6;
    g.strokeRect(margin.left, margin.top, plotW, plotH);
    g.globalAlpha = 1; g.fillStyle = ink; g.font = '11px sans-serif';
    ticks(view.x0, view.x1, block.logx).forEach(t => {
      const X = px(t);
      if (X < margin.left - 1 || X > width - margin.right + 1) return;
      g.globalAlpha = 0.15; g.beginPath(); g.moveTo(X, margin.top);
      g.lineTo(X, margin.top + plotH); g.stroke(); g.globalAlpha = 1;
      g.textAlign = 'center';
      g.fillText(label(t, block.logx), X, height - margin.bottom + 14);
    });
    ticks(view.y0, view.y1, block.logy).forEach(t => {
      const Y = py(t);
      if (Y < margin.top - 1 || Y > height - margin.bottom + 1) return;
      g.globalAlpha = 0.15; g.beginPath(); g.moveTo(margin.left, Y);
      g.lineTo(width - margin.right, Y); g.stroke(); g.globalAlpha = 1;
      g.textAlign = 'right';
      g.fillText(label(t, block.logy), margin.left - 6, Y + 3.5);
    });
    g.textAlign = 'center';
    g.fillText(block.xlabel, margin.left + plotW / 2, height - 6);
    g.save(); g.translate(12, margin.top + plotH / 2);
    g.rotate(-Math.PI / 2); g.fillText(block.ylabel, 0, 0); g.restore();
  }
  // over an axis, the gesture works that axis alone — the GUI's own
  // behavior: the x-axis strip zooms/pans x, the y-axis strip y
  function axisOnly(e) {
    const r = canvas.getBoundingClientRect();
    const X = e.clientX - r.left, Y = e.clientY - r.top;
    const overX = Y > 340 - margin.bottom, overY = X < margin.left;
    return { xOnly: overX && !overY, yOnly: overY && !overX };
  }
  let dragging = null;
  canvas.addEventListener('pointerdown', e => {
    dragging = Object.assign(
      { x: e.clientX, y: e.clientY, view: Object.assign({}, view) },
      axisOnly(e));
    canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const r = canvas.getBoundingClientRect();
    const fx = (view.x1 - view.x0) / (r.width - margin.left - margin.right);
    const fy = (view.y1 - view.y0) / (340 - margin.top - margin.bottom);
    if (!dragging.yOnly) {
      view.x0 = dragging.view.x0 - (e.clientX - dragging.x) * fx;
      view.x1 = dragging.view.x1 - (e.clientX - dragging.x) * fx;
    }
    if (!dragging.xOnly) {
      view.y0 = dragging.view.y0 + (e.clientY - dragging.y) * fy;
      view.y1 = dragging.view.y1 + (e.clientY - dragging.y) * fy;
    }
    clampView(); draw(); });
  canvas.addEventListener('pointerup', () => dragging = null);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const { xOnly, yOnly } = axisOnly(e);
    const cx = view.x0 + (e.clientX - r.left - margin.left)
      / (r.width - margin.left - margin.right) * (view.x1 - view.x0);
    const cy = view.y1 - (e.clientY - r.top - margin.top)
      / (340 - margin.top - margin.bottom) * (view.y1 - view.y0);
    const factor = Math.exp(e.deltaY * 0.002);
    if (!yOnly) {
      view.x0 = cx + (view.x0 - cx) * factor;
      view.x1 = cx + (view.x1 - cx) * factor;
    }
    if (!xOnly) {
      view.y0 = cy + (view.y0 - cy) * factor;
      view.y1 = cy + (view.y1 - cy) * factor;
    }
    clampView(); draw(); }, { passive: false });
  const rehome = () => { view = Object.assign({}, home);
                         clampView(); draw(); };
  reset.onclick = rehome;
  canvas.addEventListener('dblclick', rehome);
  REDRAWS.push(draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}

/* ---- MAC grid: zoomable, never past the grid itself -------------------- */
function macBlock(block) {
  const s = document.createElement('section'); root.appendChild(s);
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  const reset = controlBar(s, block,
    'drag to pan, wheel to zoom; double-click resets');
  const c = captionFor(block); if (c) s.appendChild(c);

  const rows = block.matrix.length, columns = block.matrix[0].length;
  const full = { x0: 0, x1: columns, y0: 0, y1: rows };
  let view = Object.assign({}, full);
  const left = 64, top = 8, bottom = 56, right = 12;
  // the grid's aspect is the mode counts' own: square cells, however
  // rectangular the comparison — the canvas hugs the grid
  function frame() {
    const budget = Math.min(680, Math.max(220, rows * 24 + 70))
      - top - bottom;
    const availW = (canvas.clientWidth || 850) - left - right;
    let h = budget, w = h * columns / rows;
    if (w > availW) { w = availW; h = w * rows / columns; }
    return { w: w, h: h, height: Math.round(h + top + bottom) };
  }
  function clampView() {
    const spanX = Math.min(view.x1 - view.x0, columns);
    view.x0 = Math.max(0, Math.min(view.x0, columns - spanX));
    view.x1 = view.x0 + spanX;
    const spanY = Math.min(view.y1 - view.y0, rows);
    view.y0 = Math.max(0, Math.min(view.y0, rows - spanY));
    view.y1 = view.y0 + spanY;
  }
  function draw() {
    canvas.dataset.view = JSON.stringify(view);
    canvas.dataset.full = JSON.stringify(full);
    const f = frame();
    const [g, width] = sized(canvas, f.height);
    const plotW = f.w;
    const plotH = f.h;
    const sx = plotW / (view.x1 - view.x0);
    const sy = plotH / (view.y1 - view.y0);
    g.clearRect(0, 0, width, f.height);
    g.save(); g.beginPath();
    g.rect(left, top, plotW, plotH); g.clip();
    const i0 = Math.max(0, Math.floor(view.y0));
    const i1 = Math.min(rows, Math.ceil(view.y1));
    const j0 = Math.max(0, Math.floor(view.x0));
    const j1 = Math.min(columns, Math.ceil(view.x1));
    for (let i = i0; i < i1; i++)
      for (let j = j0; j < j1; j++) {
        g.fillStyle = viridis(block.matrix[i][j]);
        g.fillRect(left + (j - view.x0) * sx, top + (i - view.y0) * sy,
                   sx + 0.5, sy + 0.5);
      }
    // committed matched pairs wear the comparison screen's own red
    // checker — CHECKER squares a side, alternating on (a + b), so
    // half of every square keeps the value's own viridis. The app
    // draws exactly this on the flat grid and on the 3-D bars; a
    // report figure that marked them differently would be a third
    // vocabulary for one fact.
    (block.pairs || []).forEach(([i, j]) => {
      const x = left + (j - view.x0) * sx;
      const y = top + (i - view.y0) * sy;
      const n = 4;
      g.fillStyle = '#e5534b';
      for (let a = 0; a < n; a++)
        for (let b = 0; b < n; b++)
          if ((a + b) % 2 === 0)
            g.fillRect(x + a * sx / n, y + b * sy / n,
                       sx / n + 0.5, sy / n + 0.5);
    });
    g.restore();
    const ink = getComputedStyle(document.body).color;
    g.strokeStyle = ink; g.globalAlpha = 0.6;
    g.strokeRect(left, top, plotW, plotH);
    g.globalAlpha = 1; g.fillStyle = ink; g.font = '10px sans-serif';
    // label stride follows what is *visible*: zoom in and every mode
    // gets its frequency back
    const everyRow = Math.ceil((i1 - i0) / 24);
    for (let i = i0; i < i1; i += everyRow) {
      const y = top + (i + 0.6 - view.y0) * sy;
      if (y < top || y > top + plotH + 4) continue;
      g.textAlign = 'right';
      g.fillText(block.rows[i], left - 5, y);
    }
    const everyColumn = Math.ceil((j1 - j0) / 24);
    for (let j = j0; j < j1; j += everyColumn) {
      const x = left + (j + 0.65 - view.x0) * sx;
      if (x < left - 2 || x > left + plotW + 2) continue;
      g.save();
      g.translate(x, top + plotH + 6);
      g.rotate(Math.PI / 2); g.textAlign = 'left';
      g.fillText(block.columns[j], 0, 0); g.restore();
    }
  }
  let dragging = null;
  canvas.addEventListener('pointerdown', e => {
    dragging = { x: e.clientX, y: e.clientY,
                 view: Object.assign({}, view) };
    canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const f = frame();
    const fx = (view.x1 - view.x0) / f.w;
    const fy = (view.y1 - view.y0) / f.h;
    view.x0 = dragging.view.x0 - (e.clientX - dragging.x) * fx;
    view.x1 = dragging.view.x1 - (e.clientX - dragging.x) * fx;
    view.y0 = dragging.view.y0 - (e.clientY - dragging.y) * fy;
    view.y1 = dragging.view.y1 - (e.clientY - dragging.y) * fy;
    clampView(); draw(); });
  canvas.addEventListener('pointerup', () => dragging = null);
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const f = frame();
    const cx = view.x0 + (e.clientX - r.left - left)
      / f.w * (view.x1 - view.x0);
    const cy = view.y0 + (e.clientY - r.top - top)
      / f.h * (view.y1 - view.y0);
    const factor = Math.exp(e.deltaY * 0.002);
    view.x0 = cx + (view.x0 - cx) * factor;
    view.x1 = cx + (view.x1 - cx) * factor;
    view.y0 = cy + (view.y0 - cy) * factor;
    view.y1 = cy + (view.y1 - cy) * factor;
    clampView(); draw(); }, { passive: false });
  const rehome = () => { view = Object.assign({}, full); draw(); };
  reset.onclick = rehome;
  canvas.addEventListener('dblclick', rehome);
  REDRAWS.push(draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}

/* ---- coherence map: frequency across, channel down, pinned 0..1 -------- */
function mapBlock(block) {
  const s = document.createElement('section'); root.appendChild(s);
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  const c = captionFor(block); if (c) s.appendChild(c);
  function draw() {
    const rows = block.rows.length, columns = block.x.length;
    const height = Math.min(560, Math.max(200, rows * 18 + 70));
    const [g, width] = sized(canvas, height);
    const left = 110, right = 56, top = 8, bottom = 46;
    const plotW = width - left - right, plotH = height - top - bottom;
    const cellW = plotW / columns, cellH = plotH / rows;
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < columns; j++) {
        g.fillStyle = viridis(block.rows[i][j]);
        g.fillRect(left + j * cellW, top + i * cellH,
                   cellW + 0.5, cellH + 0.5);
      }
    const ink = getComputedStyle(document.body).color;
    g.fillStyle = ink; g.font = '10px sans-serif';
    const every = Math.ceil(rows / 24);
    g.textAlign = 'right';
    for (let i = 0; i < rows; i += every)
      g.fillText(block.labels[i], left - 5, top + (i + 0.6) * cellH);
    g.textAlign = 'center';
    const x0 = block.x[0], x1 = block.x[columns - 1];
    for (let n = 0; n <= 6; n++) {
      const v = x0 + (x1 - x0) * n / 6;
      g.fillText(String(Math.round(v)), left + plotW * n / 6,
                 height - bottom + 14);
    }
    g.fillText(block.xlabel, left + plotW / 2, height - 4);
    // the color bar that makes the map readable as numbers
    const barX = width - right + 18, barW = 12;
    for (let i = 0; i < plotH; i++) {
      g.fillStyle = viridis(1 - i / plotH);
      g.fillRect(barX, top + i, barW, 1.5);
    }
    g.textAlign = 'left';
    g.fillText('1', barX + barW + 3, top + 8);
    g.fillText('0', barX + barW + 3, top + plotH);
  }
  REDRAWS.push(draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}

/* ---- rotatable, animating scene ---------------------------------------- */
function sceneBlock(block) {
  const s = document.createElement('section'); root.appendChild(s);
  s.style.position = 'relative';
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  const controls = document.createElement('div');
  controls.className = 'controls'; s.appendChild(controls);
  let mode = -1, playing = false, phase = 0;
  if (block.modes.length) {
    // the corner table over the animation: what is moving, by the
    // numbers — description text stays text, never markup
    const modeinfo = document.createElement('div');
    modeinfo.className = 'modeinfo'; s.appendChild(modeinfo);
    const showInfo = () => {
      modeinfo.textContent = '';
      const raw = mode >= 0 ? block.modes[mode].info : null;
      modeinfo.style.display = raw ? 'block' : 'none';
      if (!raw) return;
      // an overlaid pair carries one entry per set
      (Array.isArray(raw) ? raw : [raw]).forEach(m => {
        const title = document.createElement('div');
        title.textContent = (m.set ? m.set + ' — ' : '')
          + 'Mode ' + m.mode
          + (m.description ? ' — ' + m.description : '');
        const numbers = document.createElement('div');
        numbers.textContent = m.frequency.toFixed(4) + ' Hz, '
          + m.damping.toFixed(3) + ' % damping';
        modeinfo.append(title, numbers);
      });
    };
    const select = document.createElement('select');
    select.append(new Option('undeformed', -1));
    block.modes.forEach((m, i) => select.append(new Option(m.label, i)));
    select.onchange = () => { mode = Number(select.value);
      playing = mode >= 0; showInfo(); };
    controls.appendChild(select);
    const play = document.createElement('button');
    play.textContent = 'play / pause';
    play.onclick = () => playing = !playing && mode >= 0;
    controls.appendChild(play);
    select.value = '0'; select.onchange();
  }
  const resetView = document.createElement('button');
  resetView.className = 'reset'; resetView.textContent = 'Reset View';
  controls.appendChild(resetView);
  const hint = document.createElement('span'); hint.className = 'hint';
  hint.textContent = 'drag to rotate, wheel to zoom'
    + (block.unit ? ' — coordinates in ' + block.unit : '');
  controls.appendChild(hint);
  const c = captionFor(block); if (c) s.appendChild(c);

  const P = block.points;
  const center = [0, 1, 2].map(a =>
    P.reduce((t, p) => t + p[a], 0) / (P.length || 1));
  let extent = Math.sqrt(Math.max(...P.map(p =>
    (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
    + (p[2] - center[2]) ** 2))) || 1;
  // home view: isometric with Z up, the GUI's own opening view
  const forward = normalized([-1, -1, -1]);
  const iso_right = normalized(cross3([0, 0, 1], forward));
  const view = orbitView(canvas, [iso_right, cross3(forward, iso_right)],
                         () => { if (!playing) draw(); });
  resetView.onclick = () => { view.home(); draw(); };
  // fit to what the home view actually projects, not the bounding
  // sphere — a flat model would otherwise open far too small
  let fitX = 0, fitY = 0;
  P.forEach(p => {
    const d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    fitX = Math.max(fitX, Math.abs(d[0] * view.basis[0][0]
      + d[1] * view.basis[0][1] + d[2] * view.basis[0][2]));
    fitY = Math.max(fitY, Math.abs(d[0] * view.basis[1][0]
      + d[1] * view.basis[1][1] + d[2] * view.basis[1][2]));
  });
  fitX = fitX || 1; fitY = fitY || 1;
  const peak = m => Math.sqrt(Math.max(...m.real.map((r, i) =>
    (r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    + (m.imag[i][0] ** 2 + m.imag[i][1] ** 2 + m.imag[i][2] ** 2))))
    || 1;

  function draw() {
    const [g, width, height] = sized(canvas, 420);
    // 2.3 half-widths across leaves ~13% margin for deflection swing
    const scale = Math.min(width / (2.3 * fitX),
                           height / (2.3 * fitY)) * view.zoom;
    const deformed = [], depth = [];
    // the view normal, for face depth and flat shading
    const n = [view.basis[0][1] * view.basis[1][2] - view.basis[0][2] * view.basis[1][1],
               view.basis[0][2] * view.basis[1][0] - view.basis[0][0] * view.basis[1][2],
               view.basis[0][0] * view.basis[1][1] - view.basis[0][1] * view.basis[1][0]];
    const m = mode >= 0 ? block.modes[mode] : null;
    const amplitude = m ? 0.1 * extent / peak(m) : 0;
    const cos = Math.cos(phase), sin = Math.sin(phase);
    const world = [];
    for (let i = 0; i < P.length; i++) {
      let p = P[i];
      if (m) p = [0, 1, 2].map(a => p[a]
        + amplitude * (m.real[i][a] * cos - m.imag[i][a] * sin));
      world.push(p);
      const dx = p[0] - center[0], dy = p[1] - center[1],
            dz = p[2] - center[2];
      deformed.push([
        width / 2 + scale * (dx * view.basis[0][0] + dy * view.basis[0][1]
                             + dz * view.basis[0][2]),
        height / 2 - scale * (dx * view.basis[1][0] + dy * view.basis[1][1]
                              + dz * view.basis[1][2])]);
      depth.push(dx * n[0] + dy * n[1] + dz * n[2]);
    }
    g.clearRect(0, 0, width, height);
    // painter's algorithm over faces and lines together: everything
    // sorted far to near, faces flat-shaded by their tilt to the view
    const items = [];
    (block.faces || []).forEach(face => items.push({
      face, z: face.nodes.reduce((s, i) => s + depth[i], 0)
        / face.nodes.length }));
    block.lines.forEach(line => items.push({
      line, z: (depth[line[0]] + depth[line[1]]) / 2 }));
    /* the node dots sort with everything else rather than being laid
       over the top of it. Drawn afterwards in one flat pass they came
       through the near faces, so a solid airframe showed the dots on
       its far side and read as translucent — which is a statement
       about the structure, and a false one (Brandon, 2026-08-25).
       The nudge forward is for the dot's *own* faces: a face lying
       square to the view averages to its corners' depth exactly, and
       the tie broke either way from frame to frame, so surface nodes
       flickered. A fraction of a percent of the model is far less
       than an element, so it cannot lift a dot through anything in
       front of it. */
    const bias = extent * 0.005;
    for (let i = 0; i < deformed.length; i++)
      items.push({ node: i, z: depth[i] + bias });
    // deflection colormap, exactly the GUI's reading: |offset| per node
    // on viridis, the top of the scale pinned to this mode's own peak.
    // A flat scene (the matched-pair overlay) keeps its own colors:
    // they are what tell the two sets apart.
    let t = null;
    if (m && !block.flat) {
      t = new Array(P.length);
      for (let i = 0; i < P.length; i++) {
        const off = [0, 1, 2].map(a =>
          m.real[i][a] * cos - m.imag[i][a] * sin);
        t[i] = Math.hypot(off[0], off[1], off[2]) / m.peak;
      }
    }
    const shadeHex = (hex, factor) => {
      const v = parseInt(hex.slice(1), 16);
      return 'rgb(' + [16, 8, 0].map(s =>
        Math.round(((v >> s) & 255) * factor)).join(',') + ')';
    };
    // the GUI's colormap interpolates node deflection *across* each
    // element. A linear scalar field over a triangle is exactly a
    // linear gradient along its screen-space slope, so each triangle
    // of a face's fan gets one, stopped through viridis.
    const shadeTriangle = (a, b, c) => {
      const [ax, ay] = deformed[a], [bx, by] = deformed[b],
            [cx, cy] = deformed[c];
      const ta = t[a], tb = t[b], tc = t[c];
      const lo = Math.min(ta, tb, tc), hi = Math.max(ta, tb, tc);
      const d1x = bx - ax, d1y = by - ay, d2x = cx - ax, d2y = cy - ay;
      const det = d1x * d2y - d1y * d2x;
      let fill;
      if (!det || hi - lo < 1e-3) {
        fill = viridis((ta + tb + tc) / 3);
      } else {
        const dt1 = tb - ta, dt2 = tc - ta;
        const gx = (dt1 * d2y - dt2 * d1y) / det;
        const gy = (dt2 * d1x - dt1 * d2x) / det;
        const g2 = gx * gx + gy * gy;
        fill = g.createLinearGradient(
          ax + gx * (lo - ta) / g2, ay + gy * (lo - ta) / g2,
          ax + gx * (hi - ta) / g2, ay + gy * (hi - ta) / g2);
        const stops = Math.min(8, Math.max(2,
          Math.ceil((hi - lo) * 16)));
        for (let s = 0; s <= stops; s++)
          fill.addColorStop(s / stops,
                            viridis(lo + (hi - lo) * s / stops));
      }
      // stroked in its own fill too: adjacent fills leave antialiased
      // hairline seams, and the hairline must not read as a wireframe
      g.fillStyle = fill; g.strokeStyle = fill; g.lineWidth = 1;
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by);
      g.lineTo(cx, cy); g.closePath(); g.fill(); g.stroke();
    };
    items.sort((a, b) => a.z - b.z);
    items.forEach(item => {
      // Per-item opacity, for the matched-pair overlay: the basis is
      // opaque and the other set is drawn *through* it. Far-to-near
      // ordering is what makes that read, and the items are already in
      // that order. Everything else omits it and draws solid.
      g.globalAlpha = (item.face ? item.face.alpha
                       : item.line ? item.line[3]
                       : (block.node_alphas || [])[item.node]) ?? 1;
      if (item.node !== undefined) {
        g.fillStyle = t ? viridis(t[item.node])
                        : block.node_colors[item.node];
        g.fillRect(deformed[item.node][0] - 2,
                   deformed[item.node][1] - 2, 4, 4);
      } else if (item.face) {
        const nodes = item.face.nodes, [a, b, c] = nodes;
        if (t) {
          for (let k = 1; k + 1 < nodes.length; k++)
            shadeTriangle(nodes[0], nodes[k], nodes[k + 1]);
          g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 1;
        } else {
          const u = [0, 1, 2].map(k => world[b][k] - world[a][k]);
          const v = [0, 1, 2].map(k => world[c][k] - world[a][k]);
          const fn = [u[1] * v[2] - u[2] * v[1],
                      u[2] * v[0] - u[0] * v[2],
                      u[0] * v[1] - u[1] * v[0]];
          const size = Math.hypot(fn[0], fn[1], fn[2]) || 1;
          const facing = Math.abs(fn[0] * n[0] + fn[1] * n[1]
                                  + fn[2] * n[2]) / size;
          g.fillStyle = shadeHex(item.face.color, 0.55 + 0.45 * facing);
          g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 1;
        }
        g.beginPath();
        nodes.forEach((i, k) => k ?
          g.lineTo(deformed[i][0], deformed[i][1]) :
          g.moveTo(deformed[i][0], deformed[i][1]));
        g.closePath();
        if (!t) g.fill();
        g.stroke();
      } else {
        const [a, b, color] = item.line;
        if (t) {
          // the traceline reads the same way: endpoint to endpoint
          const grad = g.createLinearGradient(
            deformed[a][0], deformed[a][1],
            deformed[b][0], deformed[b][1]);
          grad.addColorStop(0, viridis(t[a]));
          grad.addColorStop(0.5, viridis((t[a] + t[b]) / 2));
          grad.addColorStop(1, viridis(t[b]));
          g.strokeStyle = grad;
        } else {
          g.strokeStyle = color;
        }
        g.lineWidth = 1.6; g.beginPath();
        g.moveTo(deformed[a][0], deformed[a][1]);
        g.lineTo(deformed[b][0], deformed[b][1]);
        g.stroke();
      }
    });
    g.globalAlpha = 1;
    // DOF arrows, colored by axis like the orientation marker — drawn
    // last, they are the subject. Responses leave the node, label at
    // the tip; forces (arrows_incoming) end on it, label at the base.
    // One shared length, pre-shrunk to the node spacing; names only
    // while they can be read — a few hundred overprint.
    // DOF arrows as the GUI draws them: real 3D solids — a cylinder
    // shaft and a cone head to pyvista's own proportions (shaft
    // radius 0.05 L, head 0.25 L long by 0.1 L wide) — projected
    // through the same camera as the mesh, so they foreshorten,
    // thicken with zoom, and never read as pasted-on sprites
    const named = (block.arrows || []).length <= 40;
    const project = p => {
      const dx = p[0] - center[0], dy = p[1] - center[1],
            dz = p[2] - center[2];
      return [width / 2 + scale * (dx * view.basis[0][0] + dy * view.basis[0][1]
                                   + dz * view.basis[0][2]),
              height / 2 - scale * (dx * view.basis[1][0] + dy * view.basis[1][1]
                                    + dz * view.basis[1][2])];
    };
    (block.arrows || []).forEach(arrow => {
      const v = arrow.vector;
      const L = block.arrow_length;
      const node = world[arrow.node];
      const base3 = block.arrows_incoming
        ? [0, 1, 2].map(a => node[a] - L * v[a]) : node;
      const tip3 = block.arrows_incoming
        ? node : [0, 1, 2].map(a => node[a] + L * v[a]);
      const neck3 = [0, 1, 2].map(a => tip3[a] - 0.25 * L * v[a]);
      // two directions perpendicular to the axis, for the cone's rim
      const seed = Math.abs(v[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      let u = cross3(v, seed); u = normalized(u);
      const w = cross3(v, u);
      const rim = [];
      for (let k = 0; k < 12; k++) {
        const a = 2 * Math.PI * k / 12;
        const c = Math.cos(a) * 0.1 * L, s2 = Math.sin(a) * 0.1 * L;
        rim.push(project([neck3[0] + c * u[0] + s2 * w[0],
                          neck3[1] + c * u[1] + s2 * w[1],
                          neck3[2] + c * u[2] + s2 * w[2]]));
      }
      const base = project(base3), tip = project(tip3),
            neck = project(neck3);
      // the shaft: a cylinder's silhouette is a stroke as wide as its
      // diameter, in world units so zoom thickens it like the GUI's;
      // flat-ended, so the cap never bulges into the cone
      g.strokeStyle = arrow.color; g.lineCap = 'butt';
      g.lineWidth = Math.max(1.5, 0.1 * L * scale);
      g.beginPath(); g.moveTo(base[0], base[1]);
      g.lineTo(neck[0], neck[1]); g.stroke();
      // the head: the cone's exact silhouette — the base ellipse plus
      // the triangle from the apex to the rim's two tangency points
      // (the rim points at extreme angles about the apex). Looking
      // down the axis the triangle degenerates and the disc alone
      // remains, which is what a cone looks like end-on.
      let left = rim[0], right = rim[0], lc = 0, rc = 0;
      rim.forEach(p => {
        const c1 = (p[0] - tip[0]) * (neck[1] - tip[1])
          - (p[1] - tip[1]) * (neck[0] - tip[0]);
        if (c1 < lc) { lc = c1; left = p; }
        if (c1 > rc) { rc = c1; right = p; }
      });
      g.fillStyle = arrow.color;
      g.beginPath();
      rim.forEach((p, k) => k ? g.lineTo(p[0], p[1])
                              : g.moveTo(p[0], p[1]));
      g.closePath(); g.fill();
      g.beginPath(); g.moveTo(tip[0], tip[1]);
      g.lineTo(left[0], left[1]); g.lineTo(right[0], right[1]);
      g.closePath(); g.fill();
      if (named) {
        const dx = tip[0] - base[0], dy = tip[1] - base[1];
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        g.fillStyle = arrow.color;
        g.font = 'bold 11px sans-serif';
        // the label sits past the arrow's free end: the tip going
        // out, the base coming in
        const spot = block.arrows_incoming
          ? [base[0] - ux * 6, base[1] - uy * 6]
          : [tip[0] + ux * 6, tip[1] + uy * 6];
        const outward = block.arrows_incoming ? -ux : ux;
        g.textAlign = outward < 0 ? 'right' : 'left';
        g.fillText(arrow.label, spot[0], spot[1] + 4);
      }
    });
    g.textAlign = 'left';
    // the orientation triad, rotating with the view: X red, Y green,
    // Z blue, exactly the desktop scene's corner axes
    const origin = [40, height - 40];
    [[[1, 0, 0], '#e5534b', 'X'], [[0, 1, 0], '#3fb950', 'Y'],
     [[0, 0, 1], '#4c92d9', 'Z']].forEach(([axis, color, name]) => {
      const sx = axis[0] * view.basis[0][0] + axis[1] * view.basis[0][1]
        + axis[2] * view.basis[0][2];
      const sy = axis[0] * view.basis[1][0] + axis[1] * view.basis[1][1]
        + axis[2] * view.basis[1][2];
      const tip = [origin[0] + 26 * sx, origin[1] - 26 * sy];
      g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
      g.moveTo(origin[0], origin[1]); g.lineTo(tip[0], tip[1]);
      g.stroke();
      g.fillStyle = color; g.font = 'bold 11px sans-serif';
      g.fillText(name, origin[0] + 33 * sx - 3, origin[1] - 33 * sy + 4);
    });
  }
  // wall-clock phase, not per-frame steps: a 120 Hz display must not
  // swing the mode twice as fast as a 60 Hz one. One cycle per two
  // seconds, the desktop animator's own pace.
  const SECONDS_PER_CYCLE = 2.0;
  let last = null;
  (function tick(now) {
    if (playing && last !== null) {
      phase += 2 * Math.PI * (now - last) / (1000 * SECONDS_PER_CYCLE);
      draw();
    }
    last = now;
    requestAnimationFrame(tick);
  })(performance.now());
  REDRAWS.push(draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}

/* ---- the 3-D stage: every record receding, colored by level ----------
   The app's own reading of a data object, in the document. Many
   channels on one 2-D axis hide each other exactly where it matters —
   resonances line up and the tenth curve lands on the first nine — so
   a whole FRF matrix is drawn the way the screen draws it: one curve
   per record at its own station, on one color scale for the scene.

   The stage arrives normalized (`viz.waterfall.stage_curves`, the same
   normalization the app's stage uses) with the real ranges it stands
   in for, so this only has to project, sort and label. */
function stageBlock(block) {
  const s = document.createElement('section'); root.appendChild(s);
  s.style.position = 'relative';
  const canvas = document.createElement('canvas'); s.appendChild(canvas);
  const controls = document.createElement('div');
  controls.className = 'controls'; s.appendChild(controls);
  const resetView = document.createElement('button');
  resetView.className = 'reset'; resetView.textContent = 'Reset View';
  controls.appendChild(resetView);
  const hint = document.createElement('span'); hint.className = 'hint';
  hint.textContent = 'drag to rotate, wheel to zoom';
  controls.appendChild(hint);
  const c = captionFor(block); if (c) s.appendChild(c);

  const [sx, sy, sz] = block.stage;
  const [x0, x1, z0, z1] = block.extents;
  /* the marks stand outside the stage box on purpose — the rail
     above its ceiling, on a wall set back behind the last station —
     so the figure is centered and fitted on everything it draws,
     never on the box alone, or the window weights would be framed
     out of the picture that was asked for */
  let topZ = sz, backY = sy;
  ((block.marks || {}).averaging ? block.marks.averaging.frames : [])
    .forEach(frame => {
      backY = Math.max(backY, block.marks.averaging.wall);
      frame.glyph.forEach(z => topZ = Math.max(topZ, z));
    });
  const center = [sx / 2, backY / 2, topZ / 2];
  const view = orbitView(canvas, block.home, () => draw());
  resetView.onclick = () => { view.home(); draw(); };
  /* how many colors a curve is cut into. The level along a curve is
     what the color says, and canvas strokes one color at a time —
     so consecutive points sharing a bucket share a stroke, which
     turns twenty thousand segments into a few thousand paths. */
  const BUCKETS = 32;
  const corners = [];
  for (const a of [0, sx]) for (const b of [0, backY])
    for (const d of [0, topZ]) corners.push([a, b, d]);

  function draw() {
    const [g, width, height] = sized(canvas, 420);
    const ink = getComputedStyle(document.body).color;
    /* the view, published the way the line plots publish theirs: what
       is being looked at is a fact about the figure, and reading it
       off the pixels is guesswork — two identical draws differ by a
       few hundred antialiased pixels */
    canvas.dataset.view = JSON.stringify(
      {basis: view.basis, zoom: view.zoom});
    let fitX = 0, fitY = 0;
    corners.forEach(p => {
      const at = view.project(p, center, 1, 0, 0);
      fitX = Math.max(fitX, Math.abs(at[0]));
      fitY = Math.max(fitY, Math.abs(at[1]));
    });
    // 2.7 half-widths across: the extra over a scene's 2.3 is the
    // room the tick numbers and channel names need outside the box
    const scale = Math.min(width / (2.7 * (fitX || 1)),
                           height / (2.7 * (fitY || 1))) * view.zoom;
    const at = p => view.project(p, center, scale, width, height);
    const n = view.normal();
    const depth = p => (p[0] - center[0]) * n[0] + (p[1] - center[1]) * n[1]
                     + (p[2] - center[2]) * n[2];
    g.clearRect(0, 0, width, height);
    g.lineJoin = 'round'; g.lineCap = 'round';

    /* everything — box edges, grid lines and curves alike — painted
       far to near out of one list, so a curve in front of a grid line
       covers it and the box behind the data stays behind it */
    const items = [];
    const edge = (a, b, faint) => items.push(
      {edge: [a, b], faint: faint, z: (depth(a) + depth(b)) / 2});
    for (const b of [0, sy]) for (const d of [0, sz])
      edge([0, b, d], [sx, b, d], true);
    for (const a of [0, sx]) for (const d of [0, sz])
      edge([a, 0, d], [a, sy, d], true);
    for (const a of [0, sx]) for (const b of [0, sy])
      edge([a, b, 0], [a, b, sz], true);
    (block.runs || []).forEach(run => {
      const mid = run.xz[Math.floor(run.xz.length / 2)] || [0, 0];
      items.push({run: run, z: depth([mid[0], run.station, mid[1]])});
    });
    /* the scalogram's sheet: a surface, not a fan of lines — a quad
       per cell on the level colormap, each corner standing at its own
       amplitude, so the ridges rise the way the app's stage draws
       them (Brandon, 2026-08-29). Cells inside the cone of influence
       stand dimmed: there the picture is shaped by where the record
       was cut, and undimmed it would read as data. Every quad joins
       the one painter's list, so the box edges and axis numbers sort
       against the surface like everything else. */
    /* the scalogram's surface, one item per ROW STRIP rather than
       per quad — smoothness is the point (Brandon, 2026-08-29: the
       surface stays, the choppiness does not). A strip sorts against
       the box and marks by its own middle, exact for parallel
       strips; inside a strip the painter below walks its columns
       far-to-near and merges same-colored runs of quads into single
       polygons, so a quiet floor costs a handful of fills instead of
       thousands. */
    if (block.sheet) {
      const sheet = block.sheet, rows = sheet.stations;
      for (let r = 0; r + 1 < rows.length; r++)
        items.push({strip: r, sheet: sheet,
                    z: depth([sx / 2, (rows[r] + rows[r + 1]) / 2,
                              0])});
      /* the cone of influence as two translucent walls, exactly the
         app's own reading (Brandon, 2026-08-29 — the report painter
         fills translucent faces perfectly well; the dimmed cells
         were a choice, and the wrong one once the two disagreed):
         a curtain from floor to ceiling at the reach the record's
         ends have at each row, one face per row pair so the painter
         sorts it against the surface like everything else. */
      const cone = sheet.cone || [];
      for (const side of [0, 1])
        for (let r = 0; r + 1 < rows.length; r++) {
          const a = side ? sx - (cone[r] || 0) : (cone[r] || 0);
          const b = side ? sx - (cone[r + 1] || 0) : (cone[r + 1] || 0);
          items.push({
            face: [[a, rows[r], 0], [b, rows[r + 1], 0],
                   [b, rows[r + 1], sz], [a, rows[r], sz]],
            color: '128, 128, 128', alpha: 0.22,
            z: depth([(a + b) / 2, (rows[r] + rows[r + 1]) / 2,
                      sz / 2])});
        }
    }
    /* the marks the flat figure carries, given depth: the analyzed
       span as a slab across every channel — because the analysis
       covers every channel — and, on the back wall, the rail of
       window weights. Sorted in with everything else, so the slab
       tints the curves inside it and not the ones in front. */
    const marks = block.marks || {};
    const slab = (x0, x1, color, alpha) => {
      // the four upright faces of the box, each its own item: a
      // translucent solid read through is what the app draws, and a
      // face at a time is what a painter's algorithm can order
      [[[x0, 0, 0], [x1, 0, 0], [x1, 0, sz], [x0, 0, sz]],
       [[x0, sy, 0], [x1, sy, 0], [x1, sy, sz], [x0, sy, sz]],
       [[x0, 0, 0], [x0, sy, 0], [x0, sy, sz], [x0, 0, sz]],
       [[x1, 0, 0], [x1, sy, 0], [x1, sy, sz], [x1, 0, sz]]].forEach(
        face => items.push({
          face: face, color: color, alpha: alpha,
          z: face.reduce((t, p) => t + depth(p), 0) / face.length}));
      // and the rims that say exactly where it starts and stops
      [x0, x1].forEach(x => items.push({
        rim: [[x, 0, 0], [x, sy, 0], [x, sy, sz], [x, 0, sz], [x, 0, 0]],
        z: depth([x, sy / 2, sz / 2])}));
    };
    if (marks.averaging) slab(marks.averaging.span[0],
                              marks.averaging.span[1], markBlue(), 0.18);
    (marks.shocks ? marks.shocks.windows : []).forEach(w => {
      slab(w.span[0], w.span[1], markBlue(), 0.18);
      // numbered over the slab, as the app numbers them: 'shock 2'
      // points at nothing unless the figure says which one that is
      items.push({number: w.label,
                  at: [(w.span[0] + w.span[1]) / 2, sy / 2, sz * 1.08],
                  z: depth([(w.span[0] + w.span[1]) / 2, sy / 2, sz])});
    });
    (marks.averaging ? marks.averaging.frames : []).forEach(frame => {
      // the band under a window carries that window's own weight per
      // column, so the shading *is* what each moment contributes
      const wall = marks.averaging.wall;
      items.push({frame: frame, wall: wall,
                  fade: marks.averaging.fade,
                  cap: marks.averaging.cap,
                  z: depth([frame.xs[Math.floor(frame.xs.length / 2)],
                            wall, frame.baseline])});
    });
    items.sort((p, q) => p.z - q.z);

    items.forEach(item => {
      if (item.strip !== undefined) {
        /* one row strip of the scalogram's surface. Columns walk
           far-to-near (smaller depth first, matching the global
           sort's own convention), and consecutive quads sharing a
           quantized color and cone state merge into one polygon —
           its top edge the row-r profile, its bottom the row-r+1
           profile walked back. A half-pixel stroke in the fill's own
           color closes the antialiasing seams between spans. */
        const sheet = item.sheet, r = item.strip;
        const xs = sheet.xs, rows = sheet.stations, lv = sheet.levels;
        const last = xs.length - 2;
        const forward = depth([xs[0], rows[r], 0])
                      <= depth([xs[last + 1], rows[r], 0]);
        const flush = (c0, c1, key) => {
          const color = viridis(key / 31);
          g.globalAlpha = 1;
          g.fillStyle = color; g.strokeStyle = color;
          g.lineWidth = 0.5;
          g.beginPath();
          g.moveTo(...at([xs[c0], rows[r], lv[r][c0] * sz]));
          for (let c = c0 + 1; c <= c1 + 1; c++)
            g.lineTo(...at([xs[c], rows[r], lv[r][c] * sz]));
          for (let c = c1 + 1; c >= c0; c--)
            g.lineTo(...at([xs[c], rows[r + 1], lv[r + 1][c] * sz]));
          g.closePath(); g.fill(); g.stroke();
        };
        let open = null, key = null;
        for (let k = 0; k <= last; k++) {
          const c = forward ? k : last - k;
          const level = (lv[r][c] + lv[r][c + 1]
                         + lv[r + 1][c] + lv[r + 1][c + 1]) / 4;
          const wanted = Math.round(
            Math.min(1, Math.max(0, level)) * 31);
          if (open !== null && wanted === key
              && c === (forward ? open.c1 + 1 : open.c0 - 1)) {
            if (forward) open.c1 = c; else open.c0 = c;
          } else {
            if (open !== null) flush(open.c0, open.c1, key);
            open = {c0: c, c1: c}; key = wanted;
          }
        }
        if (open !== null) flush(open.c0, open.c1, key);
        return;
      }
      if (item.face) {
        // the slab, read through: the stretch of the record the
        // analysis covers, across every channel
        const path = item.face.map(at);
        g.fillStyle = 'rgba(' + item.color + ', ' + item.alpha + ')';
        g.beginPath(); g.moveTo(path[0][0], path[0][1]);
        path.slice(1).forEach(q => g.lineTo(q[0], q[1]));
        g.closePath(); g.fill();
        return;
      }
      if (item.rim) {
        const path = item.rim.map(at);
        g.strokeStyle = 'rgba(' + markOrange() + ', 0.55)';
        g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(path[0][0], path[0][1]);
        path.slice(1).forEach(q => g.lineTo(q[0], q[1]));
        g.stroke();
        return;
      }
      if (item.number) {
        const spot = at(item.at);
        g.fillStyle = ink; g.globalAlpha = 0.85;
        g.font = 'bold 11px sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(item.number, spot[0], spot[1]);
        g.globalAlpha = 1;
        return;
      }
      if (item.frame) {
        const frame = item.frame, wall = item.wall;
        const blue = markBlue(), orange = markOrange();
        // one column per sample of the window, each at its own
        // weight: the 2-D gradient brush, given depth
        for (let i = 0; i < frame.xs.length - 1; i++) {
          const w = Math.min(Math.abs(frame.weights[i]), 1);
          const quad = [[frame.xs[i], wall, 0],
                        [frame.xs[i + 1], wall, 0],
                        [frame.xs[i + 1], wall, frame.baseline],
                        [frame.xs[i], wall, frame.baseline]].map(at);
          g.fillStyle = 'rgba(' + blue + ', '
            + (w * item.fade).toFixed(3) + ')';
          g.beginPath(); g.moveTo(quad[0][0], quad[0][1]);
          quad.slice(1).forEach(q => g.lineTo(q[0], q[1]));
          g.closePath(); g.fill();
        }
        // the window itself, and the tick under each end — a
        // tapering window returns to its baseline, so without those
        // nothing says where one frame stopped and the next began
        g.strokeStyle = 'rgb(' + orange + ')'; g.lineWidth = 1.5;
        g.beginPath();
        frame.xs.forEach((x, i) => {
          const spot = at([x, wall, frame.glyph[i]]);
          if (i) g.lineTo(spot[0], spot[1]);
          else g.moveTo(spot[0], spot[1]);
        });
        g.stroke();
        g.beginPath();
        frame.caps.forEach(x => {
          const low = at([x, wall, frame.baseline - item.cap]);
          const high = at([x, wall, frame.baseline + item.cap]);
          g.moveTo(low[0], low[1]); g.lineTo(high[0], high[1]);
        });
        g.stroke();
        return;
      }
      if (item.edge) {
        const a = at(item.edge[0]), b = at(item.edge[1]);
        g.strokeStyle = ink; g.globalAlpha = 0.16; g.lineWidth = 1;
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
        g.stroke(); g.globalAlpha = 1;
        return;
      }
      const run = item.run, xz = run.xz, levels = run.levels;
      if (xz.length < 2) return;
      g.lineWidth = 1.4;
      /* the stood-back object of a pair: the muted gray a compared
         reference wears, off the color scale entirely — present,
         never competing. Drawn from the page's own ink so it follows
         a theme flip, where a baked color would not. */
      if (run.quiet) {
        g.strokeStyle = ink; g.globalAlpha = 0.55;
        g.beginPath();
        let spot = at([xz[0][0], run.station, xz[0][1]]);
        g.moveTo(spot[0], spot[1]);
        for (let i = 1; i < xz.length; i++) {
          spot = at([xz[i][0], run.station, xz[i][1]]);
          g.lineTo(spot[0], spot[1]);
        }
        g.stroke(); g.globalAlpha = 1;
        return;
      }
      /* The color runs *along* each segment, never in blocks. VTK
         interpolates the level between a line's two ends, so a
         segment from a peak down to a trough shades from one to the
         other; painting it in the color of the end it started at
         streaked bright ink clean across the swing of a time history
         (Brandon, 2026-08-24).

         Drawn as VTK draws it — quantized to the color scale, which
         is what a lookup table does anyway. A segment spanning
         several buckets is cut into that many pieces, each flat at
         its own level, and the pieces are collected per bucket so one
         stroke paints all of a curve's ink in that color. The
         honest reading, a true canvas gradient per segment, costs a
         gradient object each: measured at 91 ms a frame on the
         32-FRF stage against 9 for flat, where this is 12. Cut per
         curve rather than per scene, or the batching would paint a
         far curve over a near one and lose the depth order. */
      const bucketOf = v => Math.max(0, Math.min(BUCKETS - 1,
        Math.round(v * (BUCKETS - 1))));
      const strokes = [];
      const strokeFor = b => strokes[b]
        || (strokes[b] = new Path2D());
      let previous = at([xz[0][0], run.station, xz[0][1]]);
      for (let i = 1; i < xz.length; i++) {
        const here = at([xz[i][0], run.station, xz[i][1]]);
        const from = bucketOf(levels[i - 1]), to = bucketOf(levels[i]);
        if (from === to) {
          const path = strokeFor(from);
          path.moveTo(previous[0], previous[1]);
          path.lineTo(here[0], here[1]);
        } else {
          // never more pieces than the segment has pixels: past that
          // the steps are shorter than the line is wide
          const span = Math.hypot(here[0] - previous[0],
                                  here[1] - previous[1]);
          const cuts = Math.max(1, Math.min(Math.abs(to - from),
                                            Math.ceil(span / 2)));
          for (let k = 0; k < cuts; k++) {
            const t0 = k / cuts, t1 = (k + 1) / cuts;
            // the level runs linearly along the segment, so an equal
            // cut in position is an equal cut in level
            const level = levels[i - 1] + (levels[i] - levels[i - 1])
                        * (t0 + t1) / 2;
            const path = strokeFor(bucketOf(level));
            path.moveTo(previous[0] + (here[0] - previous[0]) * t0,
                        previous[1] + (here[1] - previous[1]) * t0);
            path.lineTo(previous[0] + (here[0] - previous[0]) * t1,
                        previous[1] + (here[1] - previous[1]) * t1);
          }
        }
        previous = here;
      }
      strokes.forEach((path, bucket) => {
        g.strokeStyle = viridis(bucket / (BUCKETS - 1));
        g.stroke(path);
      });
    });

    /* the axes' numbers, outside the box: the stage is a unit cube and
       the ranges are what it stands in for, so a reader gets the
       frequencies and levels rather than stage coordinates */
    const middle = at(center);
    const outward = (p, by) => {
      const q = at(p);
      const dx = q[0] - middle[0], dy = q[1] - middle[1];
      const len = Math.hypot(dx, dy) || 1;
      return [q[0] + dx / len * by, q[1] + dy / len * by];
    };
    g.font = '10px sans-serif'; g.fillStyle = ink; g.globalAlpha = 0.75;
    g.textAlign = 'center'; g.textBaseline = 'middle';
    /* along the bottom edge that projects lowest, so the frequencies
       stay under the figure however it is turned */
    const front = [0, sy].reduce((best, b) =>
      at([sx / 2, b, 0])[1] > at([sx / 2, best, 0])[1] ? b : best, 0);
    /* an SRS stage stands in log10 of natural frequency; the ticks
       land on decades and print the real values, exactly as the
       vertical axis does for a log level */
    ticks(x0, x1, block.logx).forEach(v => {
      if (v < x0 || v > x1) return;
      const spot = outward([(v - x0) / ((x1 - x0) || 1) * sx, front, 0], 12);
      g.fillText(label(v, block.logx), spot[0], spot[1]);
    });
    /* and up the vertical edge that projects leftmost */
    const side = [[0, 0], [0, sy], [sx, 0], [sx, sy]].reduce((best, p) =>
      at([p[0], p[1], sz / 2])[0] < at([best[0], best[1], sz / 2])[0]
        ? p : best, [0, 0]);
    ticks(z0, z1, block.log).forEach(v => {
      if (v < z0 || v > z1) return;
      const spot = outward(
        [side[0], side[1], (v - z0) / ((z1 - z0) || 1) * sz], 16);
      g.fillText(label(v, block.log), spot[0], spot[1]);
    });
    /* the depth axis has no numbers — the channel names are its
       labels, which is exactly how the app reads it. Placed on the
       floor at the near end of each station rather than at the
       curve's own first point: the curves start at wildly different
       heights, and hung off those the names landed on top of each
       other and on the level axis. Thinned by what actually
       collides, so a figure keeps every name that fits and a
       nine-hundred-record scene still reads. */
    g.globalAlpha = 0.85;
    /* one list of (station, name) whichever payload this stage
       carries: per-run for curves, per-row for the scalogram's sheet
       — the sheet has no runs, and reaching for them here is what
       crashed the whole edit chrome (Brandon, 2026-08-29: the
       report drew but every control and the Hz labels were gone) */
    const nameRows = [];
    if (block.sheet) {
      block.sheet.stations.forEach((station, k) => {
        if (block.labels[k]) nameRows.push(
          {station: station, label: block.labels[k]});
      });
    } else {
      const seen = new Set();
      (block.runs || []).forEach(run => {
        // a pair meets at one station: the name belongs to the
        // station, so the stood-back curve does not repeat it
        if (run.quiet || seen.has(run.record)) return;
        seen.add(run.record);
        nameRows.push({station: run.station,
                       label: block.labels[run.record] || ''});
      });
    }
    let lastSpot = null;
    nameRows.forEach(row => {
      const spot = outward([-0.02 * sx, row.station, 0], 10);
      if (lastSpot && Math.hypot(spot[0] - lastSpot[0],
                                 spot[1] - lastSpot[1]) < 12) return;
      lastSpot = spot;
      // away from the figure whichever way it has been turned: a name
      // right-aligned on the far side of the box reads back into the
      // data it is naming
      g.textAlign = spot[0] < middle[0] ? 'right' : 'left';
      g.fillText(row.label, spot[0], spot[1]);
    });
    /* the axis titles, on the same two edges the numbers ride */
    g.globalAlpha = 1; g.font = 'bold 11px sans-serif';
    g.textAlign = 'center';
    const xtitle = outward([sx / 2, front, 0], 30);
    g.fillText(block.xlabel, xtitle[0], xtitle[1]);
    const ztitle = outward([side[0], side[1], sz / 2], 46);
    g.save(); g.translate(ztitle[0], ztitle[1]); g.rotate(-Math.PI / 2);
    g.fillText(block.zlabel, 0, 0); g.restore();
  }
  REDRAWS.push(draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}

/* A grid figure: many control channels in one figure, a row per node
   and a column per direction, each cell the channel's own plot drawn
   by plotBlock into the cell (Brandon, 2026-09-19: above four control
   channels a sequence of figures stops reading). The column headings
   and row labels are the page's; the cells' captions are the channels,
   with how far off its global axis a channel sits as the cell's note. */
function gridBlock(block) {
  const s = section(block);
  const grid = document.createElement('div'); grid.className = 'grid';
  grid.style.gridTemplateColumns =
    'auto repeat(' + block.columns.length + ', minmax(0, 1fr))';
  grid.appendChild(document.createElement('div'));
  block.columns.forEach(name => {
    const head = document.createElement('div'); head.className = 'gridhead';
    head.textContent = name; grid.appendChild(head); });
  block.rows.forEach(row => {
    const label = document.createElement('div'); label.className = 'gridrow';
    label.textContent = row.label; grid.appendChild(label);
    row.cells.forEach(cell => {
      const holder = document.createElement('div'); holder.className = 'gridcell';
      cell.forEach(figure => plotBlock(figure, holder));
      grid.appendChild(holder); }); });
  s.insertBefore(grid, s.firstChild);
}

/* The pass/fail box under the title (Brandon, 2026-09-20): whether
   the environment passed, in one word and one color, with the two
   readings it was decided on — so a reader opening the report knows
   before reading anything else. */
function verdictBlock(block) {
  const s = section(null);
  s.className = 'verdict ' + (block.passed ? 'pass' : 'fail');
  const word = document.createElement('div'); word.className = 'word';
  word.textContent = block.passed ? 'PASS' : 'FAIL';
  s.appendChild(word);
  const lines = document.createElement('div');
  lines.textContent = block.lines_percent + '% of ' + block.channels
    + ' control channels had more than ' + block.lines_limit
    + '% of their band outside the abort limits (fails at '
    + block.lines_fail_percent + '%)';
  s.appendChild(lines);
  const rms = document.createElement('div');
  rms.textContent = block.rms_percent + '% of ' + block.channels
    + ' control channels were more than ' + block.rms_limit
    + ' dB off in RMS (fails at ' + block.rms_fail_percent + '%)';
  s.appendChild(rms);
  const how = document.createElement('div'); how.className = 'hint';
  how.textContent = 'Read off the octave-band comparison.';
  s.appendChild(how);
}

/* ---- assembly ---------------------------------------------------------- */
DATA.blocks.forEach(block => {
  if (block.kind === 'text') {
    const s = section(null); const d = document.createElement('div');
    d.className = 'text'; d.innerHTML = block.html; s.appendChild(d);
  } else if (block.kind === 'unbound') {
    const s = section(null); s.className = 'unbound';
    // an 'empty' block is bound fine — its objects just hold nothing
    // to draw (say, an FRF with no drive-point records); telling the
    // user to pick a source would send them chasing the wrong thing
    s.textContent = block.empty
      ? 'Empty ' + block.was + ' block — the bound objects have no '
        + 'matching data to draw (for a drive point plot, the FRF '
        + 'needs a response measured at the excitation DOF).'
      : 'Unbound ' + block.was + ' block — pick a source to fill it in.';
  } else if (block.kind === 'image') {
    const s = section(block);
    const img = document.createElement('img');
    img.className = 'photo'; img.src = block.src;
    img.alt = block.caption || 'photo';
    s.insertBefore(img, s.firstChild);
  } else if (block.kind === 'plot') plotBlock(block);
  else if (block.kind === 'grid') gridBlock(block);
  else if (block.kind === 'verdict') verdictBlock(block);
  else if (block.kind === 'bars') barsBlock(block);
  else if (block.kind === 'mac') macBlock(block);
  else if (block.kind === 'map') mapBlock(block);
  else if (block.kind === 'scene') sceneBlock(block);
  else if (block.kind === 'stage') stageBlock(block);
  else if (block.kind === 'table') {
    const s = section(block);
    const table = document.createElement('table');
    const head = table.createTHead().insertRow();
    block.headers.forEach(h => { const th = document.createElement('th');
      th.textContent = h; head.appendChild(th); });
    const body = table.createTBody();
    /* a cell the app marks is marked here too, in the same two colors
       it shades an exceedance with: a channel red on screen is red on
       the page */
    block.rows.forEach((r, ri) => { const row = body.insertRow();
      r.forEach((v, ci) => {
        const cell = row.insertCell();
        cell.textContent = v;
        const mark = block.marks && block.marks[ri] && block.marks[ri][ci];
        if (mark) cell.className = 'mark-' + mark;
      }); });
    const wrap = document.createElement('div');
    wrap.className = 'tablewrap';
    wrap.appendChild(table);
    s.insertBefore(wrap, s.firstChild);
  }
});

// dark / light toggle: starts from the OS preference (or the reader's
// last choice), and repaints every canvas in the new ink
(() => {
  const toggle = document.createElement('button');
  toggle.className = 'themetoggle';
  const apply = theme => {
    document.documentElement.dataset.theme = theme;
    toggle.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    REDRAWS.forEach(redraw => redraw());
    try { localStorage.setItem('visualdynamics-report-theme', theme); }
    catch (e) { /* file: pages may refuse storage; the toggle still works */ }
  };
  let theme = null;
  try { theme = localStorage.getItem('visualdynamics-report-theme'); } catch (e) {}
  if (theme !== 'dark' && theme !== 'light')
    theme = window.matchMedia
      && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark' : 'light';
  apply(theme);
  toggle.onclick = () => apply(
    document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  document.body.appendChild(toggle);
})();
"""


_EDIT_CSS = """
.unbound { border: 1px dashed var(--faint); border-radius: 6px;
           padding: .8rem; color: var(--faint); }
/* the one piece of chrome the editor's page keeps: which block the
   bar and the pane are acting on. A state of the view, not an act. */
section { border-radius: 6px; }
section.selected { outline: 2px solid var(--accent); outline-offset: 8px; }
"""

_EDIT_JS = r"""
/* The editor's page does two things the export's does not: it frames
   the selected block, and it says which block was clicked. Every act
   — insert, move, delete, a caption, the text — is on the application's
   bar and pane (Brandon, 2026-09-08), so nothing else crosses the
   channel and the exported file never carried any of this. */
new QWebChannel(qt.webChannelTransport, channel => {
  const bridge = channel.objects.bridge;
  const sections = Array.from(document.querySelectorAll('section'));
  window.__select = index => {
    sections.forEach((s, position) => {
      const block = DATA.blocks[position];
      s.classList.toggle('selected', !!block && block.index === index);
    });
  };
  sections.forEach((s, position) => {
    s.addEventListener('click', () => {
      const block = DATA.blocks[position];
      if (!block || block.index === undefined) return;
      window.__select(block.index);
      bridge.apply(JSON.stringify({ op: 'select', at: block.index,
                                    scroll: window.scrollY }));
    });
  });
  // a click on the page outside every block clears the selection —
  // the bar and the pane go back to the report's own (Brandon,
  // 2026-09-08: "I am not able to de-select it")
  document.addEventListener('click', event => {
    if (event.target.closest('section')) return;
    window.__select(-1);
    bridge.apply(JSON.stringify({ op: 'select', at: null,
                                  scroll: window.scrollY }));
  });
  window.__select(DATA.selected === undefined ? -1 : DATA.selected);
});
"""
