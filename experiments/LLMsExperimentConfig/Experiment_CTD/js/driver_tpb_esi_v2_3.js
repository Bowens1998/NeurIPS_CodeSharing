// ER‑TPB v2.3: reliable start, active mode buttons, diagnosis-grounded vitals, LLM mode
// Includes Headless Parallel Execution support

(async function () {
  const cfg = await (await fetch('./js/config.json', { cache: 'no-store' })).json();
  const R = (id) => document.getElementById(id);
  const log = (msg) => { R('log').textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + R('log').textContent; };
  const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];
  function addClass(el, c) { el.classList.add(c); }
  function rmClass(el, c) { el.classList.remove(c); }

  // Parallel UI
  const parallelStartBtn = R('parallelStartBtn'), parallelStopBtn = R('parallelStopBtn');
  const parallelMonitorSection = R('parallelMonitorSection'), closeParallelMonitor = R('closeParallelMonitor');
  const parallelStats = R('parallelStats'), parallelWorkers = R('parallelWorkers'), parallelDownloadBtn = R('parallelDownloadBtn');

  // Logic Helpers (Pure)
  const Logic = {
    samplePatient: (cfg) => {
      const ages = cfg.patient_profiles.ages;
      const sets = cfg.patient_profiles.comorbid_sets;
      return { age: choice(ages), comorbid: choice(sets) };
    },
    sampleTrueDx: (cfg, highPrev) => {
      const pool = cfg.dx_options;
      const w = pool.map(dx => {
        if (dx === 'Stable / no acute condition') return highPrev ? 0.8 : 1.2;
        if (dx === 'Respiratory failure') return highPrev ? 1.3 : 1.0;
        if (dx === 'Cardiac event') return 1.1;
        if (dx === 'Massive hemorrhage') return 0.9;
        if (dx === 'Infection / sepsis') return 1.2;
        if (dx === 'Neurological event (stroke-like)') return 1.0;
        return 1.0;
      });
      const sum = w.reduce((a, b) => a + b, 0);
      let u = Math.random() * sum;
      for (let i = 0; i < w.length; i++) { u -= w[i]; if (u <= 0) return pool[i]; }
      return pool[0];
    },
    sampleSeverity: (cfg, highPrev) => {
      const prev = highPrev ? cfg.ui_defaults.prevalence_high : cfg.ui_defaults.prevalence_low;
      const u = Math.random();
      if (u < prev * 0.35) return 'LSI';
      if (u < prev) return 'HR';
      return 'STABLE';
    },
    vitalsFromDx: (trueDx, severity, comorbid, highNoise, cfg) => {
      const base = {
        'Respiratory failure': { HR: 110, SBP: 105, RR: 28, SpO2: 86, Temp: 37.5, AVPU: 'V' },
        'Cardiac event': { HR: 102, SBP: 110, RR: 20, SpO2: 94, Temp: 37.0, AVPU: 'A' },
        'Massive hemorrhage': { HR: 122, SBP: 85, RR: 24, SpO2: 93, Temp: 36.8, AVPU: 'A' },
        'Infection / sepsis': { HR: 110, SBP: 95, RR: 22, SpO2: 94, Temp: 38.8, AVPU: 'A' },
        'Neurological event (stroke-like)': { HR: 90, SBP: 160, RR: 18, SpO2: 96, Temp: 37.0, AVPU: 'V' },
        'Stable / no acute condition': { HR: 80, SBP: 120, RR: 16, SpO2: 98, Temp: 36.9, AVPU: 'A' },
      }[trueDx];
      const sev = {
        'STABLE': { dHR: 0, dSBP: 0, dRR: 0, dSpO2: 0, dTemp: 0, AVPU: null },
        'HR': { dHR: +10, dSBP: -10, dRR: +4, dSpO2: -2, dTemp: +0.2, AVPU: null },
        'LSI': { dHR: +20, dSBP: -20, dRR: +8, dSpO2: -6, dTemp: +0.3, AVPU: 'U' }
      }[severity];
      let cHR = 0, cSBP = 0, cRR = 0, cSpO2 = 0, cTemp = 0;
      if (comorbid) {
        if (comorbid.includes('Hypertension')) cSBP += 10;
        if (comorbid.includes('COPD')) { cRR += 2; cSpO2 -= 2; }
        if (comorbid.includes('Diabetes')) cTemp += 0.1;
        if (comorbid.includes('CKD')) cSBP -= 5;
        if (comorbid.includes('CAD')) cHR += 4;
        if (comorbid.includes('Immunosuppressed')) cTemp += 0.2;
        if (comorbid.includes('Anticoagulant use')) cSBP -= 3;
      }
      const noise = highNoise ? cfg.ui_defaults.noise_high : cfg.ui_defaults.noise_low;
      const jitter = (v, s) => (v + (Math.random() * 2 - 1) * s * 5);
      let HR = Math.round(jitter(base.HR + sev.dHR + cHR, noise));
      let SBP = Math.round(jitter(base.SBP + sev.dSBP + cSBP, noise));
      let RR = Math.round(jitter(base.RR + sev.dRR + cRR, noise));
      let SpO2 = Math.max(70, Math.min(100, Math.round(jitter(base.SpO2 + sev.dSpO2 + cSpO2, noise))));
      let Temp = (base.Temp + sev.dTemp + cTemp + (Math.random() * 2 - 1) * 0.1 * noise).toFixed(1);
      let AVPU = sev.AVPU ? sev.AVPU : base.AVPU;
      const DBP = Math.max(40, Math.round(SBP * 0.6));
      return { HR, BP: `${SBP}/${DBP}`, RR, SpO2, Temp, AVPU };
    },
    flagsFromDx: (trueDx, severity) => {
      const f = {
        'Respiratory failure': ['cyanosis', 'accessory muscles', 'tachypnea'],
        'Cardiac event': ['diaphoresis', 'pressure-like chest pain', 'ECG changes?'],
        'Massive hemorrhage': ['pallor', 'weak pulses', 'cool extremities'],
        'Infection / sepsis': ['fever', 'rigors', 'warm flushed skin?'],
        'Neurological event (stroke-like)': ['facial droop?', 'slurred speech?', 'arm drift?'],
        'Stable / no acute condition': ['anxious', 'mild discomfort']
      }[trueDx];
      if (severity === 'LSI') return f.slice(0, 3);
      if (severity === 'HR') return f.slice(0, 2);
      return f.slice(1, 3);
    },
    maybeFlipSeverity: (currSev, highVol, cfg) => {
      const vol = highVol ? cfg.ui_defaults.volatility_high : cfg.ui_defaults.volatility_low;
      if (Math.random() < vol) {
        if (currSev === 'STABLE') return (Math.random() < 0.7) ? 'HR' : 'LSI';
        else if (currSev === 'HR') return (Math.random() < 0.7) ? 'STABLE' : 'LSI';
        else return (Math.random() < 0.8) ? 'HR' : 'LSI';
      }
      return currSev;
    },
    buildPrompt: (state) => {
      const { tick, T, vitals, flags, patient, cc, dx_options } = state;
      return [
        `tick=${tick}/${T}`,
        `patient={age:${patient.age}, comorbid:${JSON.stringify(patient.comorbid)}}`,
        `vitals={HR:${vitals.HR}, BP:"${vitals.BP}", RR:${vitals.RR}, SpO2:${vitals.SpO2}, Temp:${vitals.Temp}, AVPU:"${vitals.AVPU}"}`,
        `complaint="${cc}", flags=${JSON.stringify(flags)}`,
        `INSTRUCTIONS:`,
        `1. Respond STRICT JSON.`,
        `2. Do NOT finalize early unless the patient is unstable/critical.`,
        `3. For stable cases, observe for at least 6-8 ticks before finalizing.`,
        `4. Output format (single line):`,
        `{"action":{"finalize_ESI":null}, "BC":{"ctx":0..100}, "BF":{"dx":"${dx_options[0]}|..."}, "rationale":"max 20 words"}`
      ].join('\n');
    }
  };


  // --- Headless Experiment Class ---
  class HeadlessExperiment {
    constructor(controls, apiConfig, workerId, onProgress) {
      this.controls = controls; // UI controls snapshot
      this.apiConfig = apiConfig;
      this.workerId = workerId;
      this.onProgress = onProgress || (() => { });
      this.log = { session: cfg.session + '_Parallel', workerId, trials: [], controls };
      this.aborted = false;
    }

    async run() {
      const targetTrials = this.controls.trials;
      for (let i = 0; i < targetTrials; i++) {
        if (this.aborted) break;
        this.onProgress(i, targetTrials, 'Running...');
        try {
          const trialResult = await this.runOneTrial(i);
          this.log.trials.push(trialResult);
        } catch (e) {
          console.error(`Worker ${this.workerId} Trial ${i} failed`, e);
          this.log.trials.push({ trialProto: i, error: e.message, timestamp: new Date().toISOString() });
        }
        this.onProgress(i + 1, targetTrials, 'Done');
      }
      return this.log;
    }

    abort() { this.aborted = true; }

    async runOneTrial(trialIdx) {
      const [minT, maxT] = cfg.tick_range;
      const drawnT = Math.floor(Math.random() * (maxT - minT + 1)) + minT;
      const cap = this.controls.max_ticks;
      const T = Math.min(drawnT, cap);

      const patient = Logic.samplePatient(cfg);
      const CCs = ["chest pain", "dyspnea", "abdominal pain", "syncope", "fever + cough", "headache + neuro deficit?", "trauma (fall)", "palpitations"];
      const cc = choice(CCs);
      const highPrev = this.controls.prevalence === 'high';
      const trueDx = Logic.sampleTrueDx(cfg, highPrev);
      Math.random(); // burn one
      let severity = Logic.sampleSeverity(cfg, highPrev);

      const trial = {
        T, maxT: cap, cc, patient, trueDx,
        severity: severity,
        steps: [], finalized: null,
        ctxUpdates: 0
      };

      let tick = 0;
      let done = false;
      let currentCtx = 0; // State for contextual belief

      while (!done && tick < T) {
        if (this.aborted) break;
        tick++;

        const highVol = this.controls.volatility === 'high';
        severity = Logic.maybeFlipSeverity(severity, highVol, cfg);
        trial.severity = severity; // Update current severity

        const highNoise = this.controls.noise === 'high';
        const vitals = Logic.vitalsFromDx(trueDx, severity, patient.comorbid, highNoise, cfg);
        const flags = Logic.flagsFromDx(trueDx, severity);

        const prompt = Logic.buildPrompt({
          tick, T, vitals, flags, patient, cc, dx_options: cfg.dx_options
        });

        // Call API
        let resp;
        try {
          resp = await this.callAPIWithRetry(prompt);
        } catch (e) {
          // Fail safe: if API fails, end trial as failure but don't crash worker
          throw new Error(`API Fail at tick ${tick}: ${e.message}`);
        }

        // Parse
        try {
          const parsed = this.parseResponse(resp);
          // Update ctx
          if (parsed.ctx != null) currentCtx = parsed.ctx;

          const step = {
            tick, vitals, cc, patient, flags, severity, trueDx,
            BF_dx: parsed.dx,
            BC_updates: [{ tick, ctx: currentCtx }], // Approximation for headless
            llm_raw: parsed
          };
          trial.steps.push(step);

          if (parsed.esi != null) {
            trial.finalized = { ESI: parsed.esi, ctxUpdates: -1 }; // -1 indicates headless
            done = true;
          }
        } catch (e) {
          throw new Error(`Parse Error at tick ${tick}: ${e.message}`);
        }

        // Force finalize if time up and not done
        if (!done && tick >= T) {
          trial.finalized = { ESI: 3, forced: true }; // Default to 3
          done = true;
        }
      }

      return trial;
    }

    parseResponse(obj) {
      // Robust parsing logic similar to UI version
      // Expected obj is already JSON object from callAPIWithRetry
      const dx = obj.BF?.dx || obj.dx;
      const ctx = obj.BC?.ctx ?? obj.ctx;
      const esi = obj.action?.finalize_ESI ?? obj.esi;
      return { dx, ctx, esi };
    }

    async callAPIWithRetry(promptStr) {
      let attempts = 0;
      const maxRetries = this.apiConfig.maxRetries;
      const timeoutMs = this.apiConfig.timeout * 1000;

      while (attempts < maxRetries) {
        attempts++;
        try {
          const controller = new AbortController();
          const id = setTimeout(() => controller.abort(), timeoutMs);

          const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${this.apiConfig.key}`,
              "Content-Type": "application/json",
              "HTTP-Referer": window.location.href,
              "X-Title": "CTD Experiment Parallel"
            },
            body: JSON.stringify({
              model: this.apiConfig.model,
              messages: [
                { role: "system", content: "You are an expert Triage Nurse. Output ONLY valid JSON." },
                { role: "user", content: promptStr }
              ],
              temperature: this.apiConfig.temperature,
              max_tokens: 5000,
              response_format: { type: "json_object" }
            }),
            signal: controller.signal
          });
          clearTimeout(id);

          if (!res.ok) {
            const txt = await res.text();
            throw new Error(`${res.status} ${txt}`);
          }

          const data = await res.json();
          const content = data.choices[0].message.content;

          // Quick cleaning based on old driver
          let clean = content.replace(/```json/gi, '').replace(/```/g, '').trim();
          const firstBrace = clean.indexOf('{');
          if (firstBrace >= 0) clean = clean.substring(firstBrace);

          try {
            return JSON.parse(clean);
          } catch (e) {
            // Simple repair attempt
            if (clean.lastIndexOf('}') === -1) clean += '}';
            return JSON.parse(clean);
          }

        } catch (e) {
          if (e.name === 'AbortError') throw new Error('Timeout');
          if (attempts >= maxRetries) throw e;
          await new Promise(r => setTimeout(r, 1000 * attempts));
        }
      }
    }
  }


  // Progress indicator
  const progressIndicator = R('progressIndicator');
  function updateProgress() {
    const completed = runLog ? runLog.trials.length : 0;
    const target = Number(controls.trials?.value) || cfg.trials_per_block;
    progressIndicator.textContent = `${completed}/${target} finished`;
  }

  // Mode selection
  let mode = 'human';
  const humanBtn = R('humanBtn'), llmBtn = R('llmBtn');
  function setMode(m) {
    mode = m;
    if (m === 'human') {
      addClass(humanBtn, 'active'); rmClass(llmBtn, 'active');
      R('llmPanel').style.display = 'none';
      R('autoRunBtn').style.display = 'none';
    }
    else {
      addClass(llmBtn, 'active'); rmClass(humanBtn, 'active');
      R('llmPanel').style.display = 'block';
      R('autoRunBtn').style.display = 'inline-block';
    }
    R('status').innerText = (m === 'llm' ? 'LLM mode selected — paste model JSON into the panel.' : 'Human mode selected — use the UI to respond.');
    log(`Mode set to ${m}`);
  }
  humanBtn.onclick = () => setMode('human');
  llmBtn.onclick = () => setMode('llm');

  // Controls & Buttons
  const controls = {
    prevalence: R('ctl_prevalence'),
    noise: R('ctl_noise'),
    vol: R('ctl_vol'),
    maxticks: R('ctl_maxticks'),
    trials: R('ctl_trials'),
  };
  const btn = {
    start: R('startBtn'),
    next: R('advanceBtn'),
    final: R('finalizeBtn'),
    restart: R('restartBtn'),
    dl: R('downloadBtn'),
    auto: R('autoRunBtn'),
    analyze: R('analyzeBtn'),
    configApi: R('configApiBtn'),
  };

  // API Config State & Elements
  let API_KEY = localStorage.getItem('tpb_api_key') || '';
  let API_MODEL = localStorage.getItem('tpb_api_model') || 'openai/gpt-4o-mini';
  let API_TEMP = parseFloat(localStorage.getItem('tpb_api_temp') || '0.7');

  const apiConfigModal = R('apiConfigModal');
  const apiKeyInput = R('apiKeyInput'); // in modal
  const apiModelSelect = R('apiModelSelect');
  const apiTempSlider = R('apiTempSlider');
  const tempVal = R('tempVal');
  const customModelInput = R('customModelInput');
  const toggleCustomBtn = R('toggleCustomModel');
  const saveApiBtn = R('saveApiConfig');
  const testApiBtn = R('testApiBtn');
  const closeApiBtn = R('closeApiConfig');
  const apiTestResult = R('apiTestResult');

  const analysisPanel = R('analysisPanel');
  const analysisOutput = R('analysisOutput');

  // Load saved config into UI
  apiKeyInput.value = API_KEY;
  apiTempSlider.value = API_TEMP;
  tempVal.innerText = API_TEMP;

  // Set model select
  if (Array.from(apiModelSelect.options).some(o => o.value === API_MODEL)) {
    apiModelSelect.value = API_MODEL;
  } else {
    // Custom model
    customModelInput.value = API_MODEL;
    customModelInput.style.display = 'block';
  }

  // Bind Config Modal Events
  btn.configApi.onclick = () => {
    apiConfigModal.style.display = 'flex';
    apiKeyInput.value = API_KEY;
  };
  closeApiBtn.onclick = () => { apiConfigModal.style.display = 'none'; };

  apiTempSlider.oninput = () => {
    tempVal.innerText = apiTempSlider.value;
  };

  toggleCustomBtn.onclick = () => {
    if (customModelInput.style.display === 'none') {
      customModelInput.style.display = 'block';
      customModelInput.focus();
    } else {
      customModelInput.style.display = 'none';
      customModelInput.value = '';
    }
  };

  saveApiBtn.onclick = () => {
    API_KEY = apiKeyInput.value.trim();
    API_TEMP = parseFloat(apiTempSlider.value);

    const custom = customModelInput.value.trim();
    API_MODEL = (custom && customModelInput.style.display !== 'none') ? custom : apiModelSelect.value;

    localStorage.setItem('tpb_api_key', API_KEY);
    localStorage.setItem('tpb_api_model', API_MODEL);
    localStorage.setItem('tpb_api_temp', API_TEMP);

    alert('Settings Saved!');
    apiConfigModal.style.display = 'none';
  };

  testApiBtn.onclick = async () => {
    const k = apiKeyInput.value.trim();
    if (!k) { alert('Enter API Key first.'); return; }

    apiTestResult.style.display = 'block';
    apiTestResult.innerText = 'Testing connection...';
    apiTestResult.style.background = '#f1f5f9';
    apiTestResult.style.color = '#334155';

    try {
      const res = await fetch("https://openrouter.ai/api/v1/auth/key", {
        method: "GET",
        headers: { "Authorization": `Bearer ${k}` }
      });
      if (res.ok) {
        const d = await res.json();
        apiTestResult.innerText = `Success! Key valid.\nUser/Credit limit info might be available in console.`;
        apiTestResult.style.background = '#dcfce7';
        apiTestResult.style.color = '#166534';
        console.log('Auth check:', d);
      } else {
        throw new Error(res.statusText + ' (' + res.status + ')');
      }
    } catch (e) {
      apiTestResult.innerText = `Error: ${e.message}`;
      apiTestResult.style.background = '#fee2e2';
      apiTestResult.style.color = '#991b1b';
    }
  };

  // Modal
  const modalBg = document.getElementById('modalBg');
  const esiSelect = document.getElementById('esiSelect');
  const confirmESI = document.getElementById('confirmESI');
  const ctxReqHint = document.getElementById('ctxReqHint');
  const openModal = () => { modalBg.style.display = 'flex'; };
  const closeModal = () => { modalBg.style.display = 'none'; };

  // Build BF diagnosis options
  const dxList = R('dxList');
  const dxOptions = (cfg.dx_options || []);
  function rebuildDx() {
    dxList.innerHTML = '';
    dxOptions.forEach((label, i) => {
      const id = `dx_${i}`;
      const row = document.createElement('label');
      row.style.display = 'flex'; row.style.alignItems = 'center'; row.style.gap = '8px';
      row.innerHTML = `<input type="radio" name="dx" value="${label}" id="${id}"><span>${label}</span>`;
      dxList.appendChild(row);
    });
  }
  rebuildDx();

  // Context slider
  const ctxSlider = R('ctxSlider');
  const ctxVal = R('ctxVal');
  const ctxBanner = R('ctxBanner');
  let ctxUpdateCount = 0;
  ctxSlider.addEventListener('input', () => {
    ctxVal.innerText = ctxSlider.value;
    if (trial && !done) {
      const s = trial.steps[trial.steps.length - 1];
      s.BC_updates.push({ tick: s.tick, ctx: Number(ctxSlider.value) });
      ctxUpdateCount += 1;
    }
  });

  // LLM panel
  const llmPrompt = R('llmPrompt');
  const llmResp = R('llmResp');
  R('copyPrompt').onclick = async () => {
    try { await navigator.clipboard.writeText(llmPrompt.value); log('Prompt copied.'); } catch (e) { log('Clipboard blocked.'); }
  };
  R('applyLLM').onclick = () => {
    try {
      const obj = JSON.parse(llmResp.value.trim());
      applyLLMResponse(obj);
      llmResp.value = '';
    } catch (e) { alert('Invalid JSON'); }
  };

  function renderPatient(p, cc, trueDx) {
    R('Age').innerText = p.age;
    R('Comorbid').innerText = (p.comorbid && p.comorbid.length) ? p.comorbid.join(', ') : 'None';
    R('CC').innerText = cc;
    R('TrueDx').innerText = trueDx;
  }

  function updateVitalsPanel(v, flags, t, T) {
    R('HR').innerText = v.HR;
    R('BP').innerText = v.BP;
    R('RR').innerText = v.RR;
    R('SpO2').innerText = v.SpO2;
    R('Temp').innerText = v.Temp;
    R('AVPU').innerText = v.AVPU;
    R('Trend').innerText = (t === 1 ? "—" : "mixed");
    R('Flags').innerText = flags.join(', ');
    R('tickLbl').innerText = `tick ${t}/${T} (cap ${trial.maxT})`;
  }

  function getSelectedDx() {
    const el = dxList.querySelector('input[name="dx"]:checked');
    return el ? el.value : null;
  }

  // State
  let runLog = null;
  let trial = null;
  let tickIdx = 0;
  let done = false;

  const CCs = ["chest pain", "dyspnea", "abdominal pain", "syncope", "fever + cough", "headache + neuro deficit?", "trauma (fall)", "palpitations"];

  // Buttons
  btn.start.onclick = () => start(true);
  btn.next.onclick = () => stepTick();
  btn.final.onclick = () => finalizeESIModal();
  btn.restart.onclick = () => restartTrial();
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); if (!btn.next.disabled) stepTick(); }
    else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); if (!btn.final.disabled) finalizeESIModal(); }
    else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); if (!btn.restart.disabled) restartTrial(); }
  });
  btn.dl.onclick = () => downloadRun();

  function enableRun(on) {
    btn.next.disabled = !on;
    btn.final.disabled = !on;
    btn.restart.disabled = !on;
  }

  function start(forceReset = false) {
    if (forceReset || !runLog) {
      runLog = {
        session: cfg.session, mode, trials: [], controls: {
          prevalence: controls.prevalence.value, noise: controls.noise.value,
          volatility: controls.vol.value, max_ticks: Number(controls.maxticks.value) || cfg.max_ticks,
          trials: Number(controls.trials.value) || cfg.trials_per_block
        }
      };
      R('log').textContent = ''; // Clear log UI
      log('Current session reset.');
      analysisPanel.style.display = 'none'; // Hide analysis
    } else {
      runLog.mode = mode;
    }
    updateProgress();
    btn.dl.disabled = true;
    enableRun(true);
    R('status').innerText = (mode === 'llm' ? 'LLM mode. Space=Next, F=Finalize (modal), R=Restart.' : 'Human mode. Space=Next, F=Finalize (modal), R=Restart.');
    newTrial();
  }

  function restartTrial() {
    if (!runLog) { start(); return; }
    log('Trial restarted.');
    R('status').innerText = (mode === 'llm' ? 'LLM mode.' : 'Human mode.') + ' Space=Next, F=Finalize, R=Restart.';
    enableRun(true);
    newTrial();
  }

  function newTrial() {
    const [minT, maxT] = cfg.tick_range;
    const drawnT = Math.floor(Math.random() * (maxT - minT + 1)) + minT;
    const cap = Number(controls.maxticks.value) || cfg.max_ticks;
    const T = Math.min(drawnT, cap);

    const patient = Logic.samplePatient(cfg);
    const cc = choice(CCs);
    const highPrev = controls.prevalence.value === 'high';
    const trueDx = Logic.sampleTrueDx(cfg, highPrev);
    Math.random(); // Sync random burn
    const severity0 = Logic.sampleSeverity(cfg, highPrev);

    trial = {
      T, maxT: cap, cc, patient, trueDx,
      severity: severity0,
      steps: [], finalized: null,
      ctxUpdates: 0
    };
    tickIdx = 0; done = false;

    ctxSlider.value = 0; ctxVal.innerText = '0'; ctxUpdateCount = 0;
    dxList.querySelectorAll('input[name="dx"]').forEach(e => e.checked = false);
    renderPatient(patient, cc, trueDx);

    log(`New trial: T=${T} (cap ${cap}), Dx=${trueDx}, Sev=${severity0}, CC=${cc}, age=${patient.age}, comorbid=${JSON.stringify(patient.comorbid)}`);
    stepTick(true);
  }

  function stepTick(first = false) {
    if (done) return;
    if (!first) {
      const highVol = controls.vol.value === 'high';
      trial.severity = Logic.maybeFlipSeverity(trial.severity, highVol, cfg);
    }
    const highNoise = controls.noise.value === 'high';
    const vitals = Logic.vitalsFromDx(trial.trueDx, trial.severity, trial.patient.comorbid, highNoise, cfg);
    const t = Math.min(tickIdx + 1, trial.T);
    const flags = Logic.flagsFromDx(trial.trueDx, trial.severity);
    updateVitalsPanel(vitals, flags, t, trial.T);

    const rt = cfg.ctx.reminder_ticks;
    const lastMinusOne = Math.max(1, trial.T - 1);
    const remA = (rt[0] === -1 ? lastMinusOne : rt[0]);
    const remB = (rt[1] === -1 ? lastMinusOne : rt[1]);
    const showReminder = (t === remA) || (t === remB);
    R('ctxBanner').style.display = showReminder ? 'block' : 'none';

    const step = {
      tick: t, vitals, cc: trial.cc, patient: trial.patient, flags,
      BF_dx: getSelectedDx(),
      BC_updates: [],
      severity: trial.severity,
      trueDx: trial.trueDx
    };
    trial.steps.push(step);
    tickIdx = t;
    log(`Tick advanced to ${t}/${trial.T}`);

    if (mode === 'llm') {
      llmPrompt.value = Logic.buildPrompt({ tick: t, T: trial.T, vitals, flags, patient: trial.patient, cc: trial.cc, dx_options: cfg.dx_options });
    }

    if (t === trial.T) {
      R('status').innerText = 'Time limit reached for this case. Please finalize ESI.';
      btn.next.disabled = true;
      finalizeESIModal();
    }
  }

  function finalizeESIModal() {
    if (done) return;
    ctxReqHint.style.display = (ctxUpdateCount < cfg.ctx.min_required_updates) ? 'block' : 'none';
    openModal();
  }

  function applyLLMResponse(obj) {
    const s = trial.steps[trial.steps.length - 1];

    // Support both formats: prefer nested (BF, BC, action), fallback to flat (dx, ctx, esi)
    const dx = obj.BF?.dx || obj.dx;
    const ctx = obj.BC?.ctx ?? obj.ctx;
    const esi = obj.action?.finalize_ESI ?? obj.esi;

    if (dx) {
      s.BF_dx = String(dx);
      const radios = dxList.querySelectorAll('input[name="dx"]');
      radios.forEach(r => { if (r.value === s.BF_dx) r.checked = true; });
      log(`LLM BF dx: ${s.BF_dx}`);
    }

    if (typeof ctx === 'number') {
      const v = Math.max(0, Math.min(100, Math.round(ctx)));
      ctxSlider.value = v; ctxVal.innerText = String(v);
      s.BC_updates.push({ tick: s.tick, ctx: v });
      ctxUpdateCount += 1;
      log(`LLM BC ctx: ${v}`);
    }

    if (esi != null) {
      const lvl = Math.max(1, Math.min(5, Number(esi)));
      esiSelect.value = String(lvl);
      finalizeESIModal();
      log(`LLM finalize ESI=${lvl}`);
    }
  }

  confirmESI.onclick = () => {
    if (done) return;
    const curr = trial.steps[trial.steps.length - 1];
    curr.BF_dx = getSelectedDx() || curr.BF_dx;
    curr.BC_updates.push({ tick: curr.tick, ctx: Number(ctxSlider.value) });

    const lvl = Math.max(1, Math.min(5, Number(esiSelect.value) || 3));
    trial.finalized = { ESI: lvl, ctxUpdates: ctxUpdateCount };
    log(`Finalized ESI=${lvl} (ctx updates=${ctxUpdateCount})`);
    R('status').innerText = `Finalized ESI=${lvl}. Press R to restart or Download to save.`;

    if (!runLog) runLog = { session: cfg.session, mode, trials: [], controls: {} };
    runLog.trials.push(trial);
    updateProgress();
    done = true;
    btn.dl.disabled = false;
    btn.analyze.disabled = false;
    btn.next.disabled = true;
    btn.final.disabled = true;
    btn.restart.disabled = false;
    closeModal();
  };

  function downloadRun() {
    const blob = new Blob([JSON.stringify(runLog || {}, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `er_tpb_v2_3_run_${Date.now()}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    log('Run JSON downloaded.');
  }

  // --- Parallel Execution Handlers ---
  let activeWorkers = [];

  parallelStartBtn.onclick = async () => {
    if (!API_KEY) { alert('Configure API Key first!'); return; }

    const groupCount = parseInt(R('parallelGroupCount').value);
    const trialsPerGroup = parseInt(R('parallelTrialsPerGroup').value);
    const timeout = parseInt(R('parallelTimeout').value);
    const retries = parseInt(R('parallelMaxRetries').value);

    parallelMonitorSection.style.display = 'block';
    parallelStartBtn.disabled = true;
    parallelStopBtn.disabled = false;
    parallelDownloadBtn.disabled = true;
    parallelWorkers.innerHTML = '';
    activeWorkers = [];

    // Snapshot config
    const controlsSnap = {
      prevalence: controls.prevalence.value,
      noise: controls.noise.value,
      volatility: controls.vol.value,
      max_ticks: Number(controls.maxticks.value) || cfg.max_ticks,
      trials: trialsPerGroup
    };

    const apiCfg = { key: API_KEY, model: API_MODEL, temperature: API_TEMP, timeout, maxRetries: retries };

    // Initialize GUI Control
    for (let i = 0; i < groupCount; i++) {
      const wId = `G${i + 1}`;
      const card = document.createElement('div');
      card.className = 'worker-card';
      card.innerHTML = `
              <div class="worker-id">${wId}</div>
              <div class="worker-progress"><progress id="prog_${wId}" value="0" max="${trialsPerGroup}"></progress></div>
              <div class="worker-status" id="status_${wId}">Waiting...</div>
          `;
      parallelWorkers.appendChild(card);

      const onProgress = (curr, total, msg) => {
        const p = document.getElementById(`prog_${wId}`);
        const s = document.getElementById(`status_${wId}`);
        if (p) p.value = curr;
        if (s) s.textContent = `${curr}/${total} ${msg}`;
      };

      activeWorkers.push(new HeadlessExperiment(controlsSnap, apiCfg, wId, onProgress));
    }

    parallelStats.textContent = `Running ${groupCount} groups concurrently...`;

    try {
      const logResults = await Promise.all(activeWorkers.map(w => w.run()));

      const finalAgg = {
        session: 'CTD_Parallel_Run',
        timestamp: new Date().toISOString(),
        config: controlsSnap,
        groups: logResults
      };

      parallelStats.textContent = 'All groups completed.';
      parallelDownloadBtn.disabled = false;
      parallelDownloadBtn.onclick = () => {
        const blob = new Blob([JSON.stringify(finalAgg, null, 2)], { type: 'application/json' });
        const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
        a.download = `tpb_parallel_${Date.now()}.json`; a.click();
      };

    } catch (e) {
      parallelStats.textContent = `Error: ${e.message}`;
    } finally {
      parallelStartBtn.disabled = false;
      parallelStopBtn.disabled = true;
    }
  };

  parallelStopBtn.onclick = () => {
    activeWorkers.forEach(w => w.abort());
    parallelStats.textContent = 'Stopping...';
    parallelStopBtn.disabled = true;
  };

  closeParallelMonitor.onclick = () => { parallelMonitorSection.style.display = 'none'; };


  // Legacy Auto Run Logic (UI Driven) - simplified to call same logic or just exist
  // We keep it for backward compat but user likely wants Headless mainly.
  let autoRunning = false;
  btn.auto.onclick = () => { alert('Use Parallel Execution Panel instead for robust headless runs.'); };

})();