// Aftermarket leaderboard: a static page (no server). Reads ONLY the aggregate metrics CSVs in ./data and the
// report cards in ./cards (copied there by `python -m aftermarket.space_push`). No prompts, completions or weights.
// Pure functions first (unit-tested with node from tests/test_space.py), then the DOM code.

const ATTACK_ORDER = ["A0", "A1", "A2", "A3", "A4q3", "A5raw", "A5bon", "A4q4", "A4q2", "A3max"];
const RUN_ORDER = ["BASE", "SDT", "GC", "IL", "RL", "RL_noguard", "ST", "SP", "AM", "AM_rand0", "AM_rand1", "AM_all"];
const METRICS = [
  { id: "asr_lg", label: "Attack success, Llama Guard (primary)", lower: true },
  { id: "asr_wg", label: "Attack success, WildGuard (second judge)", lower: true },
  { id: "over_refusal_wg", label: "Over-refusal on XSTest (WildGuard)", lower: true },
  { id: "mmlu", label: "MMLU accuracy", lower: false },
];
const ATTACK_NAMES = {
  A0: "as released", A1: "benign fine-tune", A2: "harmful fine-tune", A3: "abliteration", A4q3: "3-bit (A4)",
  A4q4: "4-bit (extra, partial)", A4q2: "2-bit (extra, partial)", A5raw: "template removed",
  A5bon: "best-of-8 perturbations", A3max: "abliteration, strongest layer (post-hoc)",
};
// sequential blue ramp (dataviz reference palette): light mode light->dark; dark mode dark->light
const RAMP = {
  light: ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  dark: ["#262624", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
};

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) return [];
  const head = rows[0];
  return rows.slice(1).map(r => Object.fromEntries(head.map((h, k) => [h, r[k] === undefined ? "" : r[k]])));
}

function num(v) {
  const x = parseFloat(v);
  return Number.isFinite(x) ? x : NaN;
}

function ordered(values, order) {
  const rank = v => (order.indexOf(v) >= 0 ? order.indexOf(v) : order.length);
  return [...new Set(values)].sort((a, b) => rank(a) - rank(b) || (a < b ? -1 : a > b ? 1 : 0));
}

// round half to even, like Python's f"{v:.0%}" (so the page shows exactly the README's numbers: 0.085 -> 8%)
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}

function pct(v) {
  return Number.isFinite(v) ? `${roundHalfEven(v * 100)}%` : "-";
}

function fmtCi(v, lo, hi) {
  if (!Number.isFinite(v)) return "-";
  return Number.isFinite(lo) ? `${pct(v)} [${pct(lo)}–${pct(hi)}]` : pct(v);
}

function select(ci, metric, runs, attacks) {
  return ci.filter(r => r.metric === metric && (!runs || !runs.length || runs.includes(r.run))
    && (!attacks || !attacks.length || attacks.includes(r.attack)));
}

// model x attack matrix of one metric: { runs, attacks, cell(run, attack) -> {value, lo, hi, n} | undefined }
function cellMatrix(ci, metric, runs, attacks) {
  const sub = select(ci, metric, runs, attacks);
  const map = new Map(sub.map(r => [`${r.run}|${r.attack}`,
    { value: num(r.value), lo: num(r.lo), hi: num(r.hi), n: num(r.n) }]));
  return {
    runs: ordered(sub.map(r => r.run), RUN_ORDER),
    attacks: ordered(sub.map(r => r.attack), ATTACK_ORDER),
    cell: (run, attack) => map.get(`${run}|${attack}`),
  };
}

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// colour of a 0-1 value on the fixed 0-100% scale (linear interpolation between ramp stops)
function rampColor(v, mode = "light") {
  const ramp = RAMP[mode];
  const t = Math.min(Math.max(v, 0), 1) * (ramp.length - 1);
  const i = Math.min(Math.floor(t), ramp.length - 2), f = t - i;
  const a = hexToRgb(ramp[i]), b = hexToRgb(ramp[i + 1]);
  const c = a.map((x, k) => Math.round(x + (b[k] - x) * f));
  return "#" + c.map(x => x.toString(16).padStart(2, "0")).join("");
}

// text on a heatmap cell: light text on the dark end of the ramp
function inkOn(v, mode = "light") {
  const darkCell = mode === "light" ? v > 0.55 : v < 0.55;
  return darkCell ? "#ffffff" : "#0b0b0b";
}

// retention rows in CSV order (ranked first, best first), with the paired test vs BASE when there is one
function retentionRows(ret, rpw) {
  const vs = new Map((rpw || []).filter(r => r.vs === "BASE").map(r => [r.run, r]));
  return ret.map(r => {
    const p = vs.get(r.run);
    return {
      run: r.run, retention: num(r.retention), lo: num(r.lo), hi: num(r.hi), n: num(r.n),
      families: r.families, partial: r.partial === "True" || r.partial === "true",
      ranked: r.ranked === undefined ? !(r.partial === "True") : r.ranked === "True",
      vsBase: p ? `${num(p.delta) >= 0 ? "+" : ""}${num(p.delta).toFixed(2)} (p_holm ${num(p.p_holm).toFixed(3)})` : "",
    };
  });
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function inline(s) {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

// a local Stage 2 figure (stage2/<name>.png) is the only image source ever rendered
const STAGE2_IMG = /^stage2\/[\w.-]+\.png$/;

// the small markdown subset report cards and the Stage 2 page use: headings, one-line paragraphs, pipe tables,
// numbered lists, "> " notes, bold, italics, code, links. Images are dropped unless opts.images, and then only
// local stage2/*.png.
function mdToHtml(md, opts = {}) {
  const out = [];
  const lines = md.replace(/\r/g, "").split("\n");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) continue;
    const img = l.match(/^!\[([^\]]*)\]\(([^)\s]+)\)\s*$/);
    if (img) {
      if (opts.images && STAGE2_IMG.test(img[2])) out.push(`<img src="${escapeHtml(img[2])}" alt="${escapeHtml(img[1])}">`);
      continue;
    }
    if (/^!\[/.test(l)) continue;
    const h = l.match(/^(#{1,3})\s+(.*)$/);
    if (h) { out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`); continue; }
    if (/^\d+\.\s/.test(l)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) items.push(lines[i++].replace(/^\d+\.\s+/, ""));
      i--;
      out.push("<ol>" + items.map(t => `<li>${inline(t)}</li>`).join("") + "</ol>");
      continue;
    }
    if (l.startsWith("> ")) { out.push(`<p class="note">${inline(l.slice(2))}</p>`); continue; }
    if (l.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      i--;
      const cells = r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const body = rows.filter((_, k) => k !== 1);
      out.push("<table><thead><tr>" + cells(body[0]).map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>"
        + body.slice(1).map(r => "<tr>" + cells(r).map(c => `<td>${inline(c)}</td>`).join("") + "</tr>").join("")
        + "</tbody></table>");
      continue;
    }
    out.push(`<p>${inline(l)}</p>`);
  }
  return out.join("\n");
}

if (typeof module !== "undefined") {
  module.exports = { ATTACK_ORDER, RUN_ORDER, METRICS, parseCSV, ordered, pct, fmtCi, select, cellMatrix, rampColor,
    inkOn, retentionRows, mdToHtml, escapeHtml };
}

// ----------------------------------------------------------------------------- DOM

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const mode = () => (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const state = { data: {}, cards: [] };

  async function fetchText(path) {
    try {
      const r = await fetch(path, { cache: "no-cache" });
      return r.ok ? await r.text() : "";
    } catch (e) { return ""; }
  }

  async function load() {
    for (const f of ["cell_ci", "retention", "retention_pairwise", "agreement", "human_agreement"]) {
      state.data[f] = parseCSV(await fetchText(`data/${f}.csv`));
    }
    const idx = await fetchText("cards/index.json");
    state.cards = idx ? JSON.parse(idx) : [];
  }

  // tooltip: one shared element, positioned at the pointer
  const tip = () => $("tooltip");
  function showTip(ev, html) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
    t.style.left = `${x}px`;
    t.style.top = `${ev.clientY + 14}px`;
  }
  const hideTip = () => { tip().hidden = true; };

  function checkboxes(container, values, labels) {
    container.innerHTML = values.map(v => `<label class="chip"><input type="checkbox" value="${escapeHtml(v)}" checked>`
      + `${escapeHtml(labels ? labels(v) : v)}</label>`).join("");
  }
  const checked = container => [...container.querySelectorAll("input:checked")].map(i => i.value);

  function renderHeatmap() {
    const ci = state.data.cell_ci;
    const metric = METRICS.find(m => m.id === $("metric").value);
    const m = cellMatrix(ci, metric.id, checked($("f-runs")), checked($("f-attacks")));
    const box = $("heatmap");
    if (!m.runs.length) { box.innerHTML = '<p class="muted">No results for this selection.</p>'; return; }
    const md = mode();
    let html = '<table class="heat"><thead><tr><th scope="col">model</th>'
      + m.attacks.map(a => `<th scope="col" title="${escapeHtml(ATTACK_NAMES[a] || a)}">${a}${a === "A3max" ? " †" : ""}</th>`).join("")
      + "</tr></thead><tbody>";
    for (const r of m.runs) {
      html += `<tr><th scope="row">${r}</th>`;
      for (const a of m.attacks) {
        const c = m.cell(r, a);
        if (!c) { html += '<td class="empty">–</td>'; continue; }
        html += `<td data-run="${r}" data-attack="${a}" style="background:${rampColor(c.value, md)};color:${inkOn(c.value, md)}">`
          + `<span class="v">${pct(c.value)}</span><span class="ci">${pct(c.lo)}–${pct(c.hi)}</span></td>`;
      }
      html += "</tr>";
    }
    box.innerHTML = html + "</tbody></table>";
    box.querySelectorAll("td[data-run]").forEach(td => {
      const c = m.cell(td.dataset.run, td.dataset.attack);
      const a = td.dataset.attack;
      td.addEventListener("mousemove", ev => showTip(ev, `<strong>${td.dataset.run} · ${a}</strong> ${escapeHtml(ATTACK_NAMES[a] || "")}`
        + `<br>${escapeHtml(metric.label)}: <strong>${fmtCi(c.value, c.lo, c.hi)}</strong><br>n = ${c.n}`));
      td.addEventListener("mouseleave", hideTip);
    });
    $("heat-legend").innerHTML = `<span>0%</span><span class="bar" style="background:linear-gradient(90deg,${RAMP[md].join(",")})"></span>`
      + `<span>100%</span><span class="muted">${metric.lower ? "lower is better" : "higher is better"} · value [95% CI] · † post-hoc</span>`;
  }

  function renderRetention() {
    const rows = retentionRows(state.data.retention, state.data.retention_pairwise);
    const box = $("retention");
    if (!rows.length) { box.innerHTML = '<p class="muted">No retention scores yet.</p>'; return; }
    const W = 720, rowH = 30, left = 190, right = 60, top = 8, H = top + rows.length * rowH + 34;
    const x = v => left + v * (W - left - right);
    const dark = mode() === "dark";
    const blue = dark ? "#3987e5" : "#2a78d6", orange = dark ? "#d95926" : "#eb6834";
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Retention ranking with 95% confidence intervals">`;
    for (let t = 0; t <= 1.0001; t += 0.2) {
      svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H - 26}" class="grid"/>`
        + `<text x="${x(t)}" y="${H - 10}" class="tick" text-anchor="middle">${t.toFixed(1)}</text>`;
    }
    rows.forEach((r, k) => {
      const y = top + k * rowH, mid = y + rowH / 2;
      const fill = r.run === "AM" ? orange : blue;
      svg += `<g class="bar" data-k="${k}"><rect x="${left}" y="${y + 3}" width="${W - left}" height="${rowH - 6}" fill="transparent"/>`
        + `<text x="${left - 8}" y="${mid + 4}" text-anchor="end" class="label">${r.run}${r.ranked ? "" : " (partial battery)"}</text>`
        + `<rect x="${left}" y="${y + 8}" width="${Math.max(x(r.retention) - left, 0)}" height="${rowH - 16}" rx="4" fill="${fill}" opacity="${r.ranked ? 1 : 0.45}"/>`
        + `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${mid}" y2="${mid}" class="whisker"/>`
        + `<text x="${x(Math.max(r.hi, r.retention)) + 6}" y="${mid + 4}" class="value">${r.retention.toFixed(2)}</text></g>`;
    });
    box.innerHTML = svg + "</svg>";
    box.querySelectorAll("g.bar").forEach(g => {
      const r = rows[+g.dataset.k];
      g.addEventListener("mousemove", ev => showTip(ev, `<strong>${r.run}</strong> retention <strong>${r.retention.toFixed(2)}</strong> `
        + `[${r.lo.toFixed(2)}–${r.hi.toFixed(2)}]<br>families ${escapeHtml(r.families)}${r.vsBase ? `<br>vs BASE ${r.vsBase}` : ""}`));
      g.addEventListener("mouseleave", hideTip);
    });
    let rank = 0;
    $("retention-table").innerHTML = "<table><thead><tr><th>rank</th><th>model</th><th>retention [95% CI]</th>"
      + "<th>vs BASE (paired)</th><th>attack families</th></tr></thead><tbody>"
      + rows.map(r => `<tr><td>${r.ranked ? ++rank : "not ranked"}</td><td>${r.run}</td>`
        + `<td>${r.retention.toFixed(2)} [${r.lo.toFixed(2)}–${r.hi.toFixed(2)}]</td><td>${r.vsBase}</td>`
        + `<td>${escapeHtml(r.families)}${r.partial ? " (partial)" : ""}</td></tr>`).join("") + "</tbody></table>";
  }

  function renderJudges() {
    const rows = state.data.agreement.map(r => [r.pair, r.n, `${(num(r.agree) * 100).toFixed(1)}%`,
      Number.isFinite(num(r.kappa)) ? num(r.kappa).toFixed(2) : "-"]);
    for (const r of state.data.human_agreement) {
      const scope = r.scope && r.scope !== "all" ? `, ${r.scope}` : "";
      rows.push([`${r.pair} (hand labels${scope}; accuracy reweighted)`, r.n, `${(num(r.weighted_accuracy) * 100).toFixed(1)}%`,
        Number.isFinite(num(r.kappa)) ? num(r.kappa).toFixed(2) : "-"]);
    }
    $("judges").innerHTML = rows.length ? "<table><thead><tr><th>labels compared</th><th>n</th><th>agreement</th>"
      + "<th>Cohen's κ</th></tr></thead><tbody>" + rows.map(r => "<tr>" + r.map(c => `<td>${escapeHtml(c)}</td>`).join("")
      + "</tr>").join("") + "</tbody></table>" : '<p class="muted">No judge results yet.</p>';
  }

  async function renderCard(run) {
    if (!run) { $("card").innerHTML = '<p class="muted">No report cards yet.</p>'; return; }
    const md = await fetchText(`cards/${run}.md`);
    $("card").innerHTML = `<img src="cards/${encodeURIComponent(run)}.png" alt="Report card for ${escapeHtml(run)}: `
      + `attack success per attack with 95% CIs, retention, over-refusal and MMLU">` + mdToHtml(md);
  }

  async function renderStage2() {
    const md = await fetchText("stage2/stage2.md");
    $("stage2").innerHTML = md ? mdToHtml(md, { images: true }) : '<p class="muted">No Stage 2 results yet.</p>';
  }

  function tabs() {
    document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll("nav button").forEach(x => x.setAttribute("aria-selected", x === b));
      document.querySelectorAll("main > section").forEach(s => { s.hidden = s.id !== `tab-${b.dataset.tab}`; });
    }));
  }

  async function main() {
    tabs();
    await load();
    const ci = state.data.cell_ci;
    $("metric").innerHTML = METRICS.map(m => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join("");
    checkboxes($("f-attacks"), ordered(ci.map(r => r.attack), ATTACK_ORDER), a => `${a} · ${ATTACK_NAMES[a] || a}`);
    checkboxes($("f-runs"), ordered(ci.map(r => r.run), RUN_ORDER));
    ["metric", "f-attacks", "f-runs"].forEach(id => $(id).addEventListener("change", renderHeatmap));
    $("card-run").innerHTML = state.cards.map(r => `<option>${escapeHtml(r)}</option>`).join("");
    $("card-run").addEventListener("change", e => renderCard(e.target.value));
    const redraw = () => { renderHeatmap(); renderRetention(); };
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    redraw();
    renderJudges();
    renderCard(state.cards[0]);
    renderStage2();
  }

  main();
}
