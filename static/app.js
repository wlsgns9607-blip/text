const F = ['꽃받침길이','꽃받침폭','꽃잎길이','꽃잎폭'];
const K = ['setosa','versicolor','virginica'];
const COL = ['var(--c0)','var(--c1)','var(--c2)'];
const $ = id => document.getElementById(id);
let rows = [], mu = [], sd = [], W, B, hist = [], raf, x = [5.8, 3.0, 4.3, 1.3], train = [], test = [];

const softmax = z => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); };
const norm = r => r.map((v, j) => (v - mu[j]) / sd[j]);
const fwd = n => softmax([0, 1, 2].map(k => B[k] + n.reduce((s, v, j) => s + v * W[j][k], 0)));
const resetW = () => { W = [0, 1, 2, 3].map(() => [0, 0, 0]); B = [0, 0, 0]; hist = []; };

function step(lr) {
  const gW = [0, 1, 2, 3].map(() => [0, 0, 0]), gB = [0, 0, 0], n = train.length;
  for (const r of train) {
    const p = fwd(r.x);
    for (let k = 0; k < 3; k++) {
      const d = (p[k] - (r.y === k ? 1 : 0)) / n;
      gB[k] += d; for (let j = 0; j < 4; j++) gW[j][k] += r.x[j] * d;
    }
  }
  for (let k = 0; k < 3; k++) { B[k] -= lr * gB[k]; for (let j = 0; j < 4; j++) W[j][k] -= lr * gW[j][k]; }
}
function evalSet(s) {
  let l = 0, a = 0;
  for (const r of s) { const p = fwd(r.x); l -= Math.log(p[r.y] + 1e-12); if (p.indexOf(Math.max(...p)) === r.y) a++; }
  return [l / s.length, a / s.length];
}

function drawPredict() {
  const p = fwd(norm(x));
  $('bars').innerHTML = K.map((k, i) => `<div class="bar"><span>${k}</span><div class="track"><div class="fill" style="width:${p[i] * 100}%;background:${COL[i]}"></div></div><span>${(p[i] * 100).toFixed(1)}%</span></div>`).join('');
  const w = p.indexOf(Math.max(...p));
  $('winner').innerHTML = `예측: <span style="color:${COL[w]}">${K[w]}</span>`;
  drawScatter();
}
function drawScatter() {
  const sx = v => 35 + (v - 0.5) / 6.9 * 350, sy = v => 230 - v / 2.7 * 210;
  let s = '';
  for (let v = 1; v <= 7; v++) s += `<text x="${sx(v)}" y="248" text-anchor="middle">${v}</text>`;
  for (let v = 0; v <= 2; v++) s += `<text x="28" y="${sy(v) + 3}" text-anchor="end">${v}</text>`;
  s += rows.map(r => `<circle cx="${sx(r.petal_length)}" cy="${sy(r.petal_width)}" r="${r.is_test ? 4.5 : 3}" fill="${COL[K.indexOf(r.species)]}" fill-opacity=".7" ${r.is_test ? 'stroke="var(--fg)" stroke-width="1"' : ''}/>`).join('');
  s += `<text x="${sx(x[2])}" y="${sy(x[3]) + 8}" text-anchor="middle" style="font-size:22px;fill:var(--fg)">★</text>`;
  s += K.map((k, i) => `<circle cx="${230 + i * 55}" cy="12" r="4" fill="${COL[i]}"/><text x="${238 + i * 55}" y="15">${k.slice(0, 5)}</text>`).join('');
  $('scatter').innerHTML = s;
}
function drawCurve() {
  const E = hist.length || 1, px = e => 30 + e / Math.max(E, 2) * 360, py = v => 160 - v * 140;
  const line = (key, c) => `<polyline fill="none" stroke="${c}" stroke-width="2" points="${hist.map(h => px(h.epoch) + ',' + py(Math.min(h[key], 1.1))).join(' ')}"/>`;
  $('curve').innerHTML = [0, .5, 1].map(v => `<line x1="30" x2="390" y1="${py(v)}" y2="${py(v)}" stroke="var(--line)"/><text x="26" y="${py(v) + 3}" text-anchor="end">${v}</text>`).join('')
    + line('loss', '#ef4444') + line('acc', 'var(--c2)') + line('val_acc', 'var(--c0)')
    + `<text x="40" y="12" style="fill:#ef4444">loss</text><text x="75" y="12" style="fill:var(--c2)">train acc</text><text x="130" y="12" style="fill:var(--c0)">test acc</text>`;
  if (hist.length) { const h = hist[hist.length - 1]; $('stat').textContent = `에포크 ${h.epoch} · loss ${h.loss.toFixed(3)} · 학습 정확도 ${(h.acc * 100).toFixed(1)}% · 테스트 정확도 ${(h.val_acc * 100).toFixed(1)}%`; }
  else $('stat').textContent = '에포크 0';
}
function drawHeat() {
  const m = Math.max(0.01, ...W.flat().map(Math.abs));
  const c = v => v >= 0 ? `rgba(239,68,68,${Math.abs(v) / m})` : `rgba(59,130,246,${Math.abs(v) / m})`;
  $('heat').innerHTML = `<div class="hm"><div class="h"></div>${K.map(k => `<div class="h">${k}</div>`).join('')}`
    + F.map((f, j) => `<div class="h">${f}</div>` + W[j].map(v => `<div style="background:${c(v)}">${v.toFixed(2)}</div>`).join('')).join('') + '</div>';
}
const drawAll = () => { drawPredict(); drawCurve(); drawHeat(); };

function startTrain() {
  cancelAnimationFrame(raf); resetW();
  const E = +$('ep').value, lr = +$('lr').value; let e = 0;
  $('go').disabled = true;
  (function loop() {
    for (let i = 0; i < 2 && e < E; i++) {
      step(lr); e++;
      const [l, a] = evalSet(train), [vl, va] = evalSet(test);
      hist.push({ epoch: e, loss: l, acc: a, val_loss: vl, val_acc: va });
    }
    drawAll();
    if (e < E) raf = requestAnimationFrame(loop);
    else {
      $('go').disabled = false;
      fetch('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lr, epochs: E, history: hist }) }).then(loadRuns);
    }
  })();
}
async function loadRuns() {
  const r = await (await fetch('/api/runs')).json();
  $('runs').innerHTML = r.length ? r.map(v => `<div>#${v.id} · lr ${v.lr} · ${v.epochs}에포크 · 학습 ${(v.train_acc * 100).toFixed(1)}% · 테스트 ${(v.test_acc * 100).toFixed(1)}%</div>`).join('') : '없음';
}
(async function init() {
  rows = await (await fetch('/api/data')).json();
  const cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];
  cols.forEach((c, j) => {
    const v = rows.map(r => r[c]); mu[j] = v.reduce((a, b) => a + b) / v.length;
    sd[j] = Math.sqrt(v.reduce((a, b) => a + (b - mu[j]) ** 2, 0) / v.length);
  });
  const all = rows.map(r => ({ x: norm(cols.map(c => r[c])), y: K.indexOf(r.species), t: r.is_test }));
  train = all.filter(r => !r.t); test = all.filter(r => r.t);
  $('sliders').innerHTML = cols.map((c, j) => {
    const v = rows.map(r => r[c]);
    return `<div class="sl"><span>${F[j]}</span><input type="range" min="${Math.min(...v)}" max="${Math.max(...v)}" step="0.1" value="${x[j]}" data-j="${j}"><span id="v${j}">${x[j]}</span></div>`;
  }).join('');
  document.querySelectorAll('#sliders input').forEach(el => el.oninput = () => { x[el.dataset.j] = +el.value; $('v' + el.dataset.j).textContent = el.value; drawPredict(); });
  $('go').onclick = startTrain;
  $('reset').onclick = () => { cancelAnimationFrame(raf); $('go').disabled = false; resetW(); drawAll(); };
  resetW(); drawAll(); loadRuns();
})();
