const fmt = (v, digits = 2) => v == null ? '—' : Number(v).toFixed(digits);
const compact = v => v == null ? '—' : Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(v);
let payload;

function sourceClass() {
  return 'downloaded';
}

function renderCards(items) {
  document.querySelector('#replication-cards').innerHTML = items.map((x, i) => `
    <article class="score-card ${i === 2 ? 'primary' : ''}">
      <span class="label">${x.label}</span><strong>${fmt(x.f1)}</strong>
      <small>F1 · ${x.note || ''}</small>
    </article>`).join('');
}

function renderModels(filter = 'all') {
  const rows = payload.models.filter(m => filter === 'all' || sourceClass(m.source) === filter);
  const min = Math.min(75, ...rows.map(x => x.f1));
  document.querySelector('#f1-chart').innerHTML = rows.slice().sort((a,b) => b.f1-a.f1).map(m => {
    const cls = sourceClass(m.source);
    const width = Math.max(1, (m.f1 - min) / (100 - min) * 100);
    return `<div class="bar-row"><b>${m.id}</b><div class="track"><div class="bar ${cls}" style="width:${width}%"></div></div><span>${fmt(m.f1)}</span></div>`;
  }).join('');
  document.querySelector('#results-body').innerHTML = rows.map(m => `<tr>
    <td><strong>${m.id}</strong> · ${m.model}</td><td>${m.source}</td><td>${fmt(m.precision)}</td>
    <td>${fmt(m.recall)}</td><td><strong>${fmt(m.f1)}</strong></td><td>${fmt(m.iou)}</td>
    <td>${compact(m.params_used)}</td><td>${m.latency_ms_bs1 == null ? '—' : fmt(m.latency_ms_bs1,1)+' ms'}</td></tr>`).join('');
}

async function init() {
  try {
    const res = await fetch('data/results.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`results HTTP ${res.status}`);
    payload = await res.json();
    renderCards(payload.replication);
    renderModels();
    document.querySelector('#findings').innerHTML = payload.findings.map(x => `<div class="finding ${x.kind || ''}">${x.text}</div>`).join('');
    document.querySelector('#generated-at').textContent = `Generated ${payload.generated_at}`;
    document.querySelector('#status span').textContent = payload.status;
  } catch (err) {
    document.querySelector('#status span').textContent = 'Results pending';
    document.querySelector('#replication-cards').innerHTML = '<article class="score-card"><span class="label">Evaluation pipeline</span><strong>…</strong><small>Final results have not been exported yet.</small></article>';
  }
}

document.querySelectorAll('[data-filter]').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('[data-filter]').forEach(x => x.classList.toggle('active', x === btn));
  if (payload) renderModels(btn.dataset.filter);
}));
init();
