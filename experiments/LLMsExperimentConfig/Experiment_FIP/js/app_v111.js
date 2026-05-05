// Financial Investment Task — v1.1.1
document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);
  const status = $('status'), errBox = $('error');
  const humanBtn = $('humanBtn'), llmBtn = $('llmBtn'), startBtn = $('startBtn'), dlBtn = $('downloadBtn');
  const configApiBtn = $('configApiBtn');

  // API Config State
  let API_KEY = localStorage.getItem('fip_api_key') || '';
  const API_BASE_URL = 'https://openrouter.ai/api/v1';
  let USE_API = localStorage.getItem('fip_use_api') === 'true';
  let API_MODEL = localStorage.getItem('fip_api_model') || 'openai/gpt-4o-mini';
  let API_TEMPERATURE = parseFloat(localStorage.getItem('fip_api_temperature') || '0.7');

  // Fallback defaults if config.json fails
  const FALLBACK = {
    session: 'InvestTaskWebV1_1_1',
    ui_defaults: {
      T_trials: 24, W_window: 80,
      p_calm_to_turb: 0.12, p_turb_to_calm: 0.18,
      mu_L: 0.0008, mu_M: 0.0008, mu_H: 0.0008,
      sigma_calm_L: 0.006, sigma_calm_M: 0.010, sigma_calm_H: 0.016,
      sigma_turb_L: 0.014, sigma_turb_M: 0.022, sigma_turb_H: 0.034,
      rho_calm: 0.15, rho_turb: 0.65,
      trend_thresh: 0.02, alloc_step: 5, seed: 42
    }
  };

  (async function boot() {
    try {
      let cfg;
      try {
        const res = await fetch('./js/config.json', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        cfg = await res.json();
      } catch (e) {
        cfg = FALLBACK;
        showError(`config.json load failed (${e.message}). Using built-in defaults.`);
      }
      await appMain(cfg);
      status.textContent = 'Ready. Pick mode, adjust sliders, then Start.';
      [humanBtn, llmBtn, startBtn, configApiBtn].forEach(b => b.disabled = false);
    } catch (e) {
      showError('Fatal init error: ' + e.message);
    }
  })();

  function showError(msg) {
    errBox.style.display = 'block';
    errBox.textContent = msg;
    console.error(msg);
  }

  // Math helpers
  function matmul(A, B) {
    const n = A.length, m = B[0].length, p = B.length;
    const C = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++) { for (let k = 0; k < p; k++) { const aik = A[i][k]; for (let j = 0; j < m; j++) C[i][j] += aik * B[k][j]; } }
    return C;
  }
  function chol(A) {
    const n = A.length, L = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(s, 1e-12)); else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  function randn() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function mulnorm(mean, cov) {
    const L = chol(cov);
    const z = [randn(), randn(), randn()], y = [0, 0, 0];
    for (let i = 0; i < 3; i++) { let s = 0; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; y[i] = s + mean[i]; }
    return y;
  }
  function covFrom(sigL, sigM, sigH, rho) {
    const S = [[sigL, 0, 0], [0, sigM, 0], [0, 0, sigH]];
    const R = [[1, rho, rho], [rho, 1, rho], [rho, rho, 1]];
    return matmul(matmul(S, R), S);
  }
  function slope(y) {
    const n = y.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const xi = i; const yi = y[i]; sx += xi; sy += yi; sxx += xi * xi; sxy += xi * yi; }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12);
  }
  function trendLabelPct(yPct, thresh) {
    const beta = slope(yPct);
    if (beta > thresh * 100) return 1;
    if (beta < -thresh * 100) return -1;
    return 0;
  }
  function normalizeAlloc(xL, xM, xH) {
    let s = xL + xM + xH;
    if (s <= 0) return [33, 33, 34];
    return [100 * xL / s, 100 * xM / s, 100 * xH / s];
  }

  async function appMain(cfg) {
    const UI = cfg.ui_defaults;
    // Bind all other DOM after we know we're good
    const submitTrial = $('submitTrial');
    const ctxRisk = $('ctxRisk'), ctxLabel = $('ctxLabel');
    const cL = $('cL'), cM = $('cM'), cH = $('cH');
    const ticksL = $('ticksL'), ticksM = $('ticksM'), ticksH = $('ticksH');
    const wL = $('wL'), wM = $('wM'), wH = $('wH'), vwL = $('vwL'), vwM = $('vwM'), vwH = $('vwH');
    const llmPanel = $('llmPanel'), llmPrompt = $('llmPrompt'), llmResponse = $('llmResponse');
    const copyPrompt = $('copyPrompt'), applyLLM = $('applyLLM');
    const autoRunBtn = $('autoRunBtn'), apiStatusEl = $('apiStatus');

    // API UI Elements
    const apiConfigModal = $('apiConfigModal'), apiKeyInput = $('apiKeyInput'), useApiCheckbox = $('useApiCheckbox');
    const apiModelSelect = $('apiModelSelect'), apiTempSlider = $('apiTempSlider'), tempVal = $('tempVal');
    const customModelInput = $('customModelInput'), toggleCustomModel = $('toggleCustomModel');
    const saveApiConfig = $('saveApiConfig'), testApiBtn = $('testApiBtn'), closeApiConfig = $('closeApiConfig');
    const apiTestResult = $('apiTestResult');
    const c_trials = $('T_trials'), c_W = $('W_window'), c_thr = $('trend_thresh');
    const c_ct = $('p_calm_to_turb'), c_tc = $('p_turb_to_calm');
    const c_rhoc = $('rho_calm'), c_rhot = $('rho_turb');
    const sigma_calm_L = $('sigma_calm_L'), sigma_calm_M = $('sigma_calm_M'), sigma_calm_H = $('sigma_calm_H');
    const sigma_turb_L = $('sigma_turb_L'), sigma_turb_M = $('sigma_turb_M'), sigma_turb_H = $('sigma_turb_H');
    const v_trials = $('v_trials'), v_W = $('v_W'), v_thr = $('v_thr'), v_ct = $('v_ct'), v_tc = $('v_tc'), v_rhoc = $('v_rhoc'), v_rhot = $('v_rhot');

    function initControls() {
      c_trials.value = UI.T_trials; v_trials.textContent = `(${UI.T_trials})`;
      c_W.value = UI.W_window; v_W.textContent = `(${UI.W_window})`;
      c_thr.value = UI.trend_thresh; v_thr.textContent = `(${UI.trend_thresh})`;
      c_ct.value = UI.p_calm_to_turb; v_ct.textContent = `(${UI.p_calm_to_turb})`;
      c_tc.value = UI.p_turb_to_calm; v_tc.textContent = `(${UI.p_turb_to_calm})`;
      c_rhoc.value = UI.rho_calm; v_rhoc.textContent = `(${UI.rho_calm})`;
      c_rhot.value = UI.rho_turb; v_rhot.textContent = `(${UI.rho_turb})`;
      sigma_calm_L.value = UI.sigma_calm_L; sigma_calm_M.value = UI.sigma_calm_M; sigma_calm_H.value = UI.sigma_calm_H;
      sigma_turb_L.value = UI.sigma_turb_L; sigma_turb_M.value = UI.sigma_turb_M; sigma_turb_H.value = UI.sigma_turb_H;
    }
    initControls();

    function bindVal(input, labelEl, fmt = (x) => x) { input.addEventListener('input', () => { labelEl.textContent = `(${fmt(input.value)})`; }); }
    bindVal(c_trials, v_trials, x => x);
    bindVal(c_W, v_W, x => x);
    bindVal(c_thr, v_thr, x => Number(x).toFixed(3));
    bindVal(c_ct, v_ct, x => Number(x).toFixed(2));
    bindVal(c_tc, v_tc, x => Number(x).toFixed(2));
    bindVal(c_rhoc, v_rhoc, x => Number(x).toFixed(2));
    bindVal(c_rhot, v_rhot, x => Number(x).toFixed(2));

    let mode = 'human';
    let isAutoRunning = false;

    humanBtn.onclick = () => { mode = 'human'; llmPanel.style.display = 'none'; status.textContent = 'Human mode selected.'; isAutoRunning = false; updateAutoBtn(); };
    llmBtn.onclick = () => { mode = 'llm'; llmPanel.style.display = 'block'; status.textContent = 'LLM mode selected.'; updateAutoBtn(); };

    function updateAutoBtn() {
      if (mode === 'llm') {
        autoRunBtn.style.display = 'block';
        autoRunBtn.textContent = '🤖 Auto Run (API)';
        autoRunBtn.style.background = '#10b981';
      } else {
        autoRunBtn.style.display = 'none';
      }
    }

    function drawSparkPct(canvas, arrPct, ticksEl, color) {
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const n = arrPct.length;
      const min = Math.min(...arrPct), max = Math.max(...arrPct);
      const x = (i) => i * (W - 10) / (n - 1) + 5;
      const y = (v) => H - 18 - (v - min) / (max - min + 1e-9) * (H - 28);
      ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
      const y0 = y(0);
      ctx.beginPath(); ctx.moveTo(5, y0); ctx.lineTo(W - 5, y0); ctx.stroke();
      ctx.strokeStyle = color || '#0ea5e9'; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const xx = x(i), yy = y(arrPct[i]); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }
      ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.font = '11px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'left'; ctx.fillText('t=0', 6, H - 4);
      ctx.textAlign = 'right'; ctx.fillText(`t=${n - 1}`, W - 6, H - 4);
      const minStr = `${min.toFixed(1)}%`; const maxStr = `${max.toFixed(1)}%`; const zeroStr = '0%';
      ticksEl.innerHTML = `<div style="position:absolute; left:0; top:0; font-size:11px; color:#64748b;">${maxStr}</div>
                           <div style="position:absolute; left:0; top:${(H - 18) / 2 - 6}px; font-size:11px; color:#64748b;">${zeroStr}</div>
                           <div style="position:absolute; left:0; bottom:0; font-size:11px; color:#64748b;">${minStr}</div>`;
    }

    function runtime() {
      return {
        T_trials: Number(c_trials.value),
        W_window: Number(c_W.value),
        trend_thresh: Number(c_thr.value),
        p_calm_to_turb: Number(c_ct.value),
        p_turb_to_calm: Number(c_tc.value),
        mu: [UI.mu_L, UI.mu_M, UI.mu_H],
        sig_calm: [Number(sigma_calm_L.value), Number(sigma_calm_M.value), Number(sigma_calm_H.value)],
        sig_turb: [Number(sigma_turb_L.value), Number(sigma_turb_M.value), Number(sigma_turb_H.value)],
        rho_calm: Number(c_rhoc.value),
        rho_turb: Number(c_rhot.value),
        seed: UI.seed
      };
    }

    function nextState(prev, p_ct, p_tc) {
      if (prev === 'calm') { return Math.random() < p_ct ? 'turb' : 'calm'; }
      return Math.random() < p_tc ? 'calm' : 'turb';
    }

    function simulateTrial(R) {
      // ... (reusing Headless logic, but kept here for UI mode)
      let state = Math.random() < 0.5 ? 'calm' : 'turb';
      const prices = { L: [100.0], M: [100.0], H: [100.0] };
      const pct = { L: [0], M: [0], H: [0] };
      const states = [];
      for (let t = 0; t < R.W_window; t++) {
        state = nextState(state, R.p_calm_to_turb, R.p_turb_to_calm);
        states.push(state);
        const cov = (state === 'calm')
          ? covFrom(R.sig_calm[0], R.sig_calm[1], R.sig_calm[2], R.rho_calm)
          : covFrom(R.sig_turb[0], R.sig_turb[1], R.sig_turb[2], R.rho_turb);
        const ret = mulnorm(R.mu, cov);
        const keys = ['L', 'M', 'H'];
        for (let i = 0; i < 3; i++) {
          const k = keys[i];
          const last = prices[k][prices[k].length - 1];
          const nextp = last * Math.exp(ret[i]);
          prices[k].push(nextp);
        }
      }
      for (const k of ['L', 'M', 'H']) { const base = prices[k][0]; pct[k] = prices[k].map(v => (v / base - 1) * 100); }
      const gt_trend = {
        L: trendLabelPct(pct.L, runtime().trend_thresh),
        M: trendLabelPct(pct.M, runtime().trend_thresh),
        H: trendLabelPct(pct.H, runtime().trend_thresh)
      };
      const ctx_true_final = states[states.length - 1];
      return { prices, pct, states, ctx_true_final, gt_trend };
    }

    let RUN = null, trialIdx = 0, trialObj = null, LOG = null;

    function renderTrial() {
      drawSparkPct(cL, trialObj.pct.L, ticksL, '#10b981');
      drawSparkPct(cM, trialObj.pct.M, ticksM, '#0ea5e9');
      drawSparkPct(cH, trialObj.pct.H, ticksH, '#ef4444');
      ctxLabel.textContent = `reported: ${ctxRisk.value}/100`;
      vwL.textContent = wL.value + '%'; vwM.textContent = wM.value + '%'; vwH.textContent = wH.value + '%';
    }


    function normalizeTriple(active, vA, vB, vC) {
      const R = 100 - vA;
      const S = vB + vC;
      let b, c;
      if (S <= 0) { b = R / 2; c = R / 2; } else { b = R * (vB / S); c = R * (vC / S); }
      return [Math.round(vA), Math.round(b), Math.round(c)];
    }

    function bindAlloc() {
      wL.addEventListener('input', () => {
        let [a, b, c] = normalizeTriple('L', Number(wL.value), Number(wM.value), Number(wH.value));
        wL.value = a; wM.value = b; wH.value = c; renderTrial();
      });
      wM.addEventListener('input', () => {
        let [b, a, c] = normalizeTriple('M', Number(wM.value), Number(wL.value), Number(wH.value));
        wM.value = b; wL.value = a; wH.value = c; renderTrial();
      });
      wH.addEventListener('input', () => {
        let [c, a, b] = normalizeTriple('H', Number(wH.value), Number(wL.value), Number(wM.value));
        wH.value = c; wL.value = a; wM.value = b; renderTrial();
      });
    }
    bindAlloc();

    function buildLLMPrompt() {
      const R = RUN;
      function serialize(arr) { const n = arr.length, k = Math.min(60, n); const step = Math.max(1, Math.floor(n / k)); const out = []; for (let i = 0; i < n; i += step) out.push(Number(arr[i].toFixed(2))); return out.slice(-60); }
      const payload = {
        instruction: "You are an investment analyst. Perform three steps: factual trends, contextual risk, and portfolio allocation.",
        units: "Series are percentage change (Δ%) relative to the first point; time axis is in steps.",
        window_len: R.W_window,
        assets: { L: serialize(trialObj.pct.L), M: serialize(trialObj.pct.M), H: serialize(trialObj.pct.H) },
        task: {
          factual: "For each asset L/M/H, output trend in {'up','flat','down'} and confidence 0-100.",
          contextual: "Estimate overall market risk 0-100 (0=calm,100=turbulent).",
          schematic: "Allocate weights (%) among L/M/H that sum to 100."
        },
        respond_in_strict_json: {
          factual: { L: { trend: 'up|flat|down', conf: '0..100' }, M: {}, H: {} },
          context: { risk: '0..100' },
          alloc: { L: '%', M: '%', H: '%' }
        }
      };
      return JSON.stringify(payload, null, 2);
    }

    async function callAPI(promptText) {
      if (!API_KEY) throw new Error('No API Key configured.');
      apiStatusEl.style.display = 'block';
      apiStatusEl.textContent = `Calling ${API_MODEL}...`;

      try {
        const res = await fetch(`${API_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_KEY}`,
            'HTTP-Referer': window.location.href,
            'X-Title': 'FIP Experiment'
          },
          body: JSON.stringify({
            model: API_MODEL,
            messages: [
              { role: 'system', content: 'You are an investment analyst. Respond strictly in JSON.' },
              { role: 'user', content: promptText }
            ],
            temperature: API_TEMPERATURE,
            response_format: { type: "json_object" }
          })
        });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`API Error ${res.status}: ${errText}`);
        }
        const data = await res.json();
        const content = data.choices[0].message.content;
        apiStatusEl.textContent = '✓ Response received.';
        return content;
      } catch (e) {
        apiStatusEl.textContent = `✗ Error: ${e.message}`;
        throw e;
      }
    }

    function factualFromUI() {
      function one(asset) {
        const sel = document.querySelector(`input[name="trend${asset}"]:checked`).value;
        const conf = Number(document.getElementById(`conf${asset}`).value);
        return { trend: sel, conf };
      }
      return { L: one('L'), M: one('M'), H: one('H') };
    }

    function applyLLMResponse(suppressAlert = false) {
      try {
        const obj = JSON.parse(llmResponse.value);
        const F = obj.factual || {};
        const ok = v => ['up', 'flat', 'down'].includes(String(v || '').toLowerCase());
        if (!ok(F.L?.trend) || !ok(F.M?.trend) || !ok(F.H?.trend)) throw new Error('Bad factual');
        const risk = Number((obj.context || {}).risk);
        if (!Number.isFinite(risk)) throw new Error('Bad risk');
        let aL = Number((obj.alloc || {}).L), aM = Number((obj.alloc || {}).M), aH = Number((obj.alloc || {}).H);
        if (![aL, aM, aH].every(Number.isFinite)) throw new Error('Bad alloc');
        let [nL, nM, nH] = normalizeAlloc(aL, aM, aH);
        document.getElementById('confL').value = Math.round(Math.max(0, Math.min(100, Number(F.L.conf || 50))));
        document.getElementById('confM').value = Math.round(Math.max(0, Math.min(100, Number(F.M.conf || 50))));
        document.getElementById('confH').value = Math.round(Math.max(0, Math.min(100, Number(F.H.conf || 50))));
        document.querySelector(`input[name="trendL"][value="${F.L.trend}"]`).checked = true;
        document.querySelector(`input[name="trendM"][value="${F.M.trend}"]`).checked = true;
        document.querySelector(`input[name="trendH"][value="${F.H.trend}"]`).checked = true;
        wL.value = Math.round(nL); wM.value = Math.round(nM); wH.value = Math.round(nH); renderTrial();
        ctxRisk.value = Math.round(Math.max(0, Math.min(100, risk))); ctxLabel.textContent = `reported: ${ctxRisk.value}/100`;
        if (!suppressAlert) alert('LLM response applied. You can Submit Trial.');
      } catch (e) {
        if (!suppressAlert) alert('Invalid LLM JSON.');
      }
    }

    function nextTrial() {
      if (trialIdx >= RUN.T_trials) {
        status.textContent = 'All trials complete. Download the log.';
        dlBtn.disabled = false;
        isAutoRunning = false;
        updateAutoBtn();
        return;
      }
      trialObj = simulateTrial(RUN);
      // reset UI
      document.getElementById('confL').value = 50; document.getElementById('confM').value = 50; document.getElementById('confH').value = 50;
      document.querySelector('input[name="trendL"][value="flat"]').checked = true;
      document.querySelector('input[name="trendM"][value="flat"]').checked = true;
      document.querySelector('input[name="trendH"][value="flat"]').checked = true;
      ctxRisk.value = 50; ctxLabel.textContent = 'reported: 50/100';
      wL.value = 33; wM.value = 33; wH.value = 34; renderTrial();

      if (mode === 'llm') { llmPrompt.value = buildLLMPrompt(); }
      status.textContent = `Trial ${trialIdx + 1} / ${RUN.T_trials}`;
    }

    function endTrial() {
      const F = factualFromUI();
      const risk = Number(ctxRisk.value);
      let aL = Number(wL.value), aM = Number(wM.value), aH = Number(wH.value);
      let [nL, nM, nH] = normalizeAlloc(aL, aM, aH); // enforce 100%
      const entry = {
        trial: trialIdx,
        params: runtime(),
        prices: trialObj.prices,
        pct: trialObj.pct,
        state_final: trialObj.ctx_true_final,
        gt_trend_pct: trialObj.gt_trend,
        report: { factual: F, contextual: { risk }, alloc: { L: nL, M: nM, H: nH } }
      };
      LOG.trials.push(entry);
      trialIdx += 1;
      nextTrial();
    }

    function initRun() {
      RUN = runtime();
      LOG = { session: 'InvestTaskWebV1_1_1', mode, params: RUN, trials: [] };
      trialIdx = 0; dlBtn.disabled = true;
      status.textContent = `Running… ${RUN.T_trials} trials.`;
      nextTrial();
    }

    startBtn.onclick = () => { try { initRun(); } catch (e) { showError('Start failed: ' + e.message); } };
    dlBtn.onclick = () => {
      const blob = new Blob([JSON.stringify(LOG, null, 2)], { type: 'application/json' });
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = `invest_task_web_v111_${Date.now()}.json`; a.click();
    };
    document.getElementById('submitTrial').onclick = endTrial;
    ctxRisk.oninput = () => { ctxLabel.textContent = `reported: ${ctxRisk.value}/100`; };
    ;[wL, wM, wH].forEach(x => x.addEventListener('input', () => { vwL.textContent = wL.value + '%'; vwM.textContent = wM.value + '%'; vwH.textContent = wH.value + '%'; }));
    document.getElementById('copyPrompt').onclick = async () => { try { await navigator.clipboard.writeText(llmPrompt.value); } catch (e) { } };
    document.getElementById('applyLLM').onclick = () => applyLLMResponse(false);

    // API Bindings
    configApiBtn.onclick = () => {
      apiKeyInput.value = API_KEY;
      useApiCheckbox.checked = USE_API;
      apiTempSlider.value = API_TEMPERATURE;
      tempVal.textContent = API_TEMPERATURE;

      // Initial model state
      const isCustom = !Array.from(apiModelSelect.options).some(o => o.value === API_MODEL);
      if (isCustom) {
        customModelInput.value = API_MODEL;
        customModelInput.style.display = 'block';
        apiModelSelect.style.display = 'none';
        toggleCustomModel.textContent = '🔄 Use dropdown list';
      } else {
        apiModelSelect.value = API_MODEL;
        customModelInput.value = '';
        customModelInput.style.display = 'none';
        apiModelSelect.style.display = 'block';
        toggleCustomModel.textContent = '⚙️ Use custom model ID';
      }

      apiConfigModal.style.display = 'flex';
    };

    toggleCustomModel.onclick = () => {
      const isHidden = customModelInput.style.display === 'none';
      if (isHidden) {
        customModelInput.style.display = 'block';
        apiModelSelect.style.display = 'none';
        toggleCustomModel.textContent = '🔄 Use dropdown list';
      } else {
        customModelInput.style.display = 'none';
        apiModelSelect.style.display = 'block';
        toggleCustomModel.textContent = '⚙️ Use custom model ID';
      }
    };

    closeApiConfig.onclick = () => apiConfigModal.style.display = 'none';

    apiTempSlider.oninput = () => tempVal.textContent = apiTempSlider.value;

    saveApiConfig.onclick = () => {
      API_KEY = apiKeyInput.value.trim();
      USE_API = useApiCheckbox.checked;
      const isCustom = customModelInput.style.display !== 'none';
      API_MODEL = isCustom ? (customModelInput.value.trim() || apiModelSelect.value) : apiModelSelect.value;
      API_TEMPERATURE = parseFloat(apiTempSlider.value);

      localStorage.setItem('fip_api_key', API_KEY);
      localStorage.setItem('fip_use_api', USE_API);
      localStorage.setItem('fip_api_model', API_MODEL);
      localStorage.setItem('fip_api_temperature', API_TEMPERATURE);

      apiConfigModal.style.display = 'none';
    };

    testApiBtn.onclick = async () => {
      const key = apiKeyInput.value.trim();
      const isCustom = customModelInput.style.display !== 'none';
      const model = isCustom ? (customModelInput.value.trim() || apiModelSelect.value) : apiModelSelect.value;

      if (!key) { apiTestResult.style.display = 'block'; apiTestResult.textContent = 'Please enter a key.'; return; }
      apiTestResult.style.display = 'block';
      apiTestResult.textContent = `Testing ${model}...`;
      apiTestResult.style.background = '#e0f2fe';

      try {
        const res = await fetch(`${API_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${key}`,
            'HTTP-Referer': window.location.href,
            'X-Title': 'FIP Experiment Test'
          },
          body: JSON.stringify({
            model: model,
            messages: [{ role: 'user', content: 'Hi' }],
            max_tokens: 5000
          })
        });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`${res.status} ${res.statusText}: ${errText}`);
        }
        const d = await res.json();
        apiTestResult.textContent = '✓ Success! Connected.';
        apiTestResult.style.background = '#dcfce7';
      } catch (e) {
        apiTestResult.textContent = '✗ Error: ' + e.message;
        apiTestResult.style.background = '#fee2e2';
      }
    };

    async function autoRunLoop() {
      if (!API_KEY) { alert('Configure API Key first.'); return; }
      if (!RUN) { showError('Press Start first.'); return; }

      isAutoRunning = true;
      autoRunBtn.textContent = '⏹ Stop Auto Run';
      autoRunBtn.style.background = '#ef4444';

      while (isAutoRunning && trialIdx < RUN.T_trials) {
        const prompt = buildLLMPrompt();
        llmPrompt.value = prompt;
        apiStatusEl.style.display = 'block';
        apiStatusEl.textContent = `Trial ${trialIdx + 1}/${RUN.T_trials}: calling API…`;

        try {
          const content = await callAPI(prompt);
          if (!isAutoRunning) break;
          llmResponse.value = content;
          applyLLMResponse(true);
          await new Promise(r => setTimeout(r, 300));
          endTrial();
          await new Promise(r => setTimeout(r, 200));
        } catch (e) {
          showError('Auto run error: ' + e.message);
          isAutoRunning = false;
          break;
        }
      }

      isAutoRunning = false;
      updateAutoBtn();
      autoRunBtn.textContent = '🤖 Auto Run (API)';
      autoRunBtn.style.background = '#10b981';
    }

    autoRunBtn.onclick = () => {
      if (isAutoRunning) { isAutoRunning = false; return; }
      autoRunLoop();
    };

  }
});
