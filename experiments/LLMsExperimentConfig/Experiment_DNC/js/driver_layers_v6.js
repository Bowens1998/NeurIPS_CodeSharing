// DSB v6 with Live Controls + OpenAI API Integration
(async function () {
  const defaults = await (await fetch('./js/config.json', { cache: 'no-store' })).json();
  let config = JSON.parse(JSON.stringify(defaults)); // shallow clone for mutation by UI

  // --- API Configuration (Unified via OpenRouter) ---
  let API_KEY = localStorage.getItem('api_key') || '';
  const API_BASE_URL = 'https://openrouter.ai/api/v1';
  let USE_API = localStorage.getItem('use_api') === 'true';
  let API_MODEL = localStorage.getItem('api_model') || 'openai/gpt-4o-mini';
  let API_TEMPERATURE = parseFloat(localStorage.getItem('api_temperature') || '0.7');

  // Clean up old config (v6.0 migration)
  if (localStorage.getItem('openai_api_key')) {
    API_KEY = localStorage.getItem('openai_api_key');
    localStorage.removeItem('openai_api_key');
    localStorage.setItem('api_key', API_KEY);
  }
  if (localStorage.getItem('api_base_url')) localStorage.removeItem('api_base_url');

  // --- UI bindings ---
  const EL = (id) => document.getElementById(id);
  const getControls = () => {
    const drift_range = parseInt(EL('ctl_drift_range').value, 10);
    const drift_values = Array.from({ length: drift_range * 2 + 1 }, (_, i) => i - drift_range);

    return {
      max_steps: parseInt(EL('ctl_max_steps').value, 10),
      cols: parseInt(EL('ctl_cols').value, 10),
      drain_step: parseFloat(EL('ctl_drain_step').value),
      drain_coll: parseFloat(EL('ctl_drain_coll').value),
      urgency_extra: parseFloat(EL('ctl_urgency').value),
      vol_low: parseFloat(EL('ctl_vol_low').value),
      vol_high: parseFloat(EL('ctl_vol_high').value),
      gust_rate: parseFloat(EL('ctl_gust_rate').value),
      bias: parseFloat(EL('ctl_bias').value),
      dense: parseFloat(EL('ctl_dense').value),
      sparse: parseFloat(EL('ctl_sparse').value),
      corr_keep: parseFloat(EL('ctl_corr_keep').value),
      drift_values,
      drift_probability: parseFloat(EL('ctl_drift_prob').value),
      trials_per_block: parseInt(EL('ctl_trials').value, 10) || 5
    };
  };

  // --- Elements ---
  const gridEl = document.getElementById('grid');
  const hud = document.getElementById('hud');
  const status = document.getElementById('status');
  const startBtn = document.getElementById('startBtn');
  const dlBtn = document.getElementById('downloadBtn');
  const humanBtn = document.getElementById('humanBtn');
  const llmBtn = document.getElementById('llmBtn');

  const llmPanel = document.getElementById('llmPanel');
  const llmPrompt = document.getElementById('llmPrompt');
  const llmResponse = document.getElementById('llmResponse');
  const copyPrompt = document.getElementById('copyPrompt');
  const nextStep = document.getElementById('nextStep');
  const autoRunBtn = document.getElementById('autoRunBtn');
  const apiStatusEl = document.getElementById('apiStatus');
  const beliefSlider = document.getElementById('beliefSlider');
  const beliefVal = document.getElementById('beliefVal');

  const surveyModal = document.getElementById('surveyModal');
  const okSurvey = document.getElementById('okSurvey');
  const skipSurvey = document.getElementById('skipSurvey');

  const analyzeBtn = document.getElementById('analyzeBtn');
  const analysisModal = document.getElementById('analysisModal');
  const closeAnalysis = document.getElementById('closeAnalysis');
  const analysisContent = document.getElementById('analysisContent');

  analyzeBtn.onclick = () => {
    if (!runLog) return;
    const results = performAnalysis(runLog);
    analysisContent.innerHTML = renderAnalysis(results);
    analysisModal.style.display = 'flex';
  };

  closeAnalysis.onclick = () => {
    analysisModal.style.display = 'none';
  };

  let mode = 'human';
  humanBtn.onclick = () => { mode = 'human'; llmPanel.style.display = 'none'; status.innerText = 'Human mode selected.'; };
  llmBtn.onclick = () => { mode = 'llm'; llmPanel.style.display = 'block'; status.innerText = 'LLM mode selected.'; };

  let runLog = null;

  function applyUI() {
    const u = getControls();
    config.trial.max_steps = u.max_steps;
    config.grid.cols = u.cols;
    config.battery.drain_per_step = u.drain_step;
    config.battery.drain_collision_penalty = u.drain_coll;
    config.battery.urgency_extra = u.urgency_extra; // custom field
    config._vol_low = u.vol_low;
    config._vol_high = u.vol_high;
    config._gust_rate = u.gust_rate;
    config._bias = u.bias;
    config._dense = u.dense;
    config._sparse = u.sparse;
    config._corr_keep = u.corr_keep;
    config.drift_values = u.drift_values;
    config.drift_probability = u.drift_probability;

    // reflow grid
    gridEl.style.gridTemplateRows = `repeat(${config.grid.rows}, ${config.grid.cell}px)`;
    gridEl.style.gridTemplateColumns = `repeat(${config.grid.cols}, ${config.grid.cell}px)`;
  }

  function rand(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function newMap(density) {
    const p = density === 'dense' ? config._dense : config._sparse;
    const walls = new Set();
    const rows = config.grid.rows, cols = config.grid.cols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if ((r === Math.floor(rows / 2) && c === 0) || (r === Math.floor(rows / 2) && c === cols - 1)) continue;
        if (Math.random() < p) walls.add(`${r},${c}`);
      }
    }
    for (let c = 0; c < cols; c++) {
      if (Math.random() < config._corr_keep) continue; // keep some obstacles
      walls.delete(`${Math.floor(config.grid.rows / 2)},${c}`);
    }
    return walls;
  }

  function driftGenerators(volatility) {
    const rows = config.grid.rows;
    const gens = [];
    const base = (volatility === 'high') ? config._vol_high : config._vol_low;
    for (let r = 0; r < rows; r++) {
      gens.push((function* () {
        let d = rand(config.drift_values);
        while (true) {
          yield d;
          if (Math.random() < base) d = rand(config.drift_values);
        }
      })());
    }
    return gens;
  }

  function render(agent, walls, battery) {
    const rows = config.grid.rows, cols = config.grid.cols, cell = config.grid.cell;
    gridEl.innerHTML = '';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const div = document.createElement('div');
        div.className = 'cell';
        const key = `${r},${c}`;
        if (walls.has(key)) div.classList.add('wall');
        if (r === Math.floor(rows / 2) && c === 0) div.classList.add('start');
        if (r === Math.floor(rows / 2) && c === cols - 1) div.classList.add('goal');
        if (r === agent.r && c === agent.c) div.classList.add('agent');
        gridEl.appendChild(div);
      }
    }
    const batPct = clamp(battery, 0, config.battery.max);
    const color = batPct > 50 ? '#22c55e' : batPct > 20 ? '#f59e0b' : '#ef4444';
    hud.innerHTML = `Battery <span class="battery"><div style="width:${batPct}%; background:${color}"></div></span> <span style="margin-left:6px">${batPct.toFixed(0)}%</span>`;
  }

  function neighbors(agent, walls) {
    const rows = config.grid.rows, cols = config.grid.cols;
    const dirs = [[-1, 0, 'UP'], [1, 0, 'DOWN'], [0, -1, 'LEFT'], [0, 1, 'RIGHT']];
    const out = [];
    for (const [dr, dc, name] of dirs) {
      const rr = agent.r + dr, cc = agent.c + dc;
      const key = `${rr},${cc}`;
      out.push({ dir: name, blocked: walls.has(key) || rr < 0 || rr >= rows || cc < 0 || cc >= cols });
    }
    return out;
  }

  function driftToNudge(drift) {
    const bias = 0.5 + (config._bias || 0.25) * Math.sign(drift);
    const gust = Math.random() < (config._gust_rate || 0.14) ? 2 : 1;
    return (Math.random() < bias ? 1 : -1) * gust;
  }

  // Build local map as 0/1 matrix (0=passable, 1=wall)
  function buildLocalMap(r, c, walls, grid_rows, grid_cols, rowRange = 2, colRange = 3) {
    const matrix = [];

    for (let dr = -rowRange; dr <= rowRange; dr++) {
      const row = [];
      for (let dc = -colRange; dc <= colRange; dc++) {
        const absRow = r + dr;
        const absCol = c + dc;

        // Out of bounds or wall = 1, otherwise = 0
        if (absRow < 0 || absRow >= grid_rows || absCol < 0 || absCol >= grid_cols) {
          row.push(1);
        } else if (walls.has(`${absRow},${absCol}`)) {
          row.push(1);
        } else {
          row.push(0);
        }
      }
      matrix.push(row);
    }

    return matrix;
  }

  function buildPrompt(state) {
    const { step, r, c, battery, time_left, collisions, urgency, walls,
      prev_action, prev_row, prev_col, grid_rows, grid_cols, goal_row, goal_col, current_belief } = state;

    const distToGoalCol = goal_col - c;
    const distToGoalRow = goal_row - r;
    const stepsLeft = time_left;

    // Build local map as 0/1 matrix (3x3 grid centered at current position)
    const localMap = buildLocalMap(r, c, walls, grid_rows, grid_cols, 1, 1);
    const mapStr = localMap.map(row => row.join(' ')).join('\n');

    // Build wind inference section
    let windInference = '';
    let blockedWarning = '';

    if (prev_action && prev_row !== undefined && prev_col !== undefined) {
      const rowChange = r - prev_row;
      const colChange = c - prev_col;

      // Check if action was blocked (e.g. tried RIGHT but didn't move RIGHT)
      // Note: Wind might mask this, but if we tried RIGHT and col didn't increase, we likely hit something or wind fought us hard.
      // Simpler check: if we are at the same spot (or very close) and battery drained.
      // Better: check if the intended move direction resulted in 0 change in that axis.

      let intendedDr = 0, intendedDc = 0;
      if (prev_action === 'UP') intendedDr = -1;
      if (prev_action === 'DOWN') intendedDr = 1;
      if (prev_action === 'LEFT') intendedDc = -1;
      if (prev_action === 'RIGHT') intendedDc = 1;

      // If we tried to move in a direction but didn't move at all in that direction (or moved opposite), we might be blocked.
      // (Ignoring wind for a moment, if I move RIGHT and col doesn't increase, I'm likely blocked).

      const movedInIntendedDir = (intendedDr !== 0 && Math.sign(rowChange) === intendedDr) ||
        (intendedDc !== 0 && Math.sign(colChange) === intendedDc);

      if (!movedInIntendedDir) {
        blockedWarning = `WARNING: Your last action "${prev_action}" did not result in movement in that direction. You are likely BLOCKED by a wall. Do not repeat "${prev_action}". Try to detour around the obstacle.`;
      }

      windInference = [
        ``,
        `Wind Analysis (infer before deciding):`,
        `- Previous position: row=${prev_row}, col=${prev_col}`,
        `- Previous action: ${prev_action}`,
        `- Current position: row=${r}, col=${c}`,
        `- Actual movement: ${rowChange} rows, ${colChange} columns`,
        `- Expected movement (no wind): ${prev_action === 'UP' ? '-1 row' : prev_action === 'DOWN' ? '1 row' : prev_action === 'LEFT' ? '-1 col' : prev_action === 'RIGHT' ? '1 col' : '0'}`,
        ``,
        `Infer: Based on intended vs actual movement, what is the current wind strength and direction?`
      ].join('\n');
    } else {
      windInference = `\n\nWind Analysis: First step, no previous movement to analyze. Wind is unknown.`;
    }

    return [
      `You are a drone pilot in a ${grid_rows}×${grid_cols} grid (rows=0-${grid_rows - 1}, cols=0-${grid_cols - 1}).`,
      `Goal: reach the green cell at row=${goal_row}, col=${goal_col}. Distance: ${distToGoalRow} rows, ${distToGoalCol} columns.`,
      ``,
      `Movement Rules:`,
      `- Without wind: UP/DOWN moves 1 row, LEFT/RIGHT moves 1 column`,
      `- Wind effect: adds lateral (horizontal) drift to your position each step`,
      `- Walls block movement and drain battery faster on collision`,
      ``,
      `Current State:`,
      `- Position: row=${r}, col=${c}`,
      `- Battery: ${battery}%`,
      `- Steps left: ${stepsLeft}`,
      `- Collisions so far: ${collisions}`,
      `- Current Contextual Belief (0=Safe, 100=Dangerous): ${current_belief}`,
      ``,
      `Nearby Map (3 rows × 3 cols, centered at your position, 0=passable, 1=wall/boundary):`,
      mapStr,
      windInference,
      blockedWarning ? `\n${blockedWarning}\n` : '',
      ``,
      `Task: First infer wind strength from movement feedback, then decide next action.`,
      `Strategy:`,
      `- If you are blocked by a wall, you MUST take a detour to go around it.`,
      `- Do not get stuck trying to push through a wall.`,
      `- Minimizing distance is good, but avoiding walls is better.`,
      ``,
      `IMPORTANT: You MUST choose exactly ONE action from this list:`,
      `- "UP" - move up one row`,
      `- "DOWN" - move down one row`,
      `- "LEFT" - move left one column`,
      `- "RIGHT" - move right one column`,
      ``,
      `You can also adjust the "belief" value (0-100) if you think the environment is safer or more dangerous than before.`,
      ``,
      `Response format (STRICT JSON, no markdown):`,
      `{"action":"UP","rationale":"your brief reasoning","belief":50}`,
      ``,
      `The "action" field MUST be one of: UP, DOWN, LEFT, RIGHT (uppercase, exact match).`,
      `The "belief" field (optional) updates the Contextual Belief.`
    ].filter(x => x).join('\n');
  }

  function buildSurveyPrompt(trialLog) {
    const { steps, factors, end_reason } = trialLog;
    const totalSteps = steps.length;
    const collisions = steps.filter(s => s.collision).length;
    const finalBattery = steps[steps.length - 1]?.battery || 0;
    const initialBattery = steps[0]?.battery || 100;

    // Generate concise step history
    const historySummary = steps.map(s => {
      const colStr = s.collision ? ' [COLLISION]' : '';
      return `Step ${s.step}: Action ${s.action}. Rationale: "${s.rationale || ''}". Result: Battery ${s.battery}${colStr}`;
    }).join('\n');

    return [
      `You just completed a drone navigation trial. Review the trial summary and step history below to evaluate the environment difficulty.`,
      ``,
      `Trial Summary:`,
      `- Total steps: ${totalSteps}`,
      `- End reason: ${end_reason}`,
      `- Starting battery: ${initialBattery}%`,
      `- Final battery: ${finalBattery}%`,
      `- Total collisions: ${collisions}`,
      ``,
      `Step History:`,
      historySummary,
      ``,
      `Task: Based on your actual experience (reflected in the history above), classify the overall environment difficulty.`,
      `Review your previous decisions and their outcomes. Consider:`,
      `- Wind variability: Did you experience unpredictable wind changes? Strong resistance?`,
      `- Battery pressure: Was battery draining faster than comfortable?`,
      `- Obstacle encounters: Did you face many collisions or near-misses?`,
      `- Navigation stress: How challenging was the overall experience?`,
      ``,
      `Provide a "belief" score from 0 (Safe) to 100 (Dangerous) based on the overall difficulty.`,
      ``,
      `Respond in STRICT JSON: {"belief":0-100,"reasoning":"brief explanation"}. No extra text.`
    ].filter(x => x).join('\n');
  }

  // --- API Call (OpenRouter) with conversation history support ---
  async function callAPI(prompt, retries = 3, conversationHistory = null, expectAction = true) {
    if (!API_KEY) {
      throw new Error('API key not configured');
    }

    const systemMsg = "You are a precise drone pilot. Respond ONLY with valid JSON, no markdown formatting, no extra text.";

    // Build messages array
    let messages;
    if (conversationHistory && conversationHistory.length > 0) {
      // History is only used for survey (contains complete trial)
      // Apply a safety limit for very long trials
      const MAX_HISTORY_PAIRS = 50;
      let limitedHistory = conversationHistory;

      if (conversationHistory.length > MAX_HISTORY_PAIRS * 2) {
        // Keep the most recent messages
        limitedHistory = conversationHistory.slice(-MAX_HISTORY_PAIRS * 2);
        console.log(`⚠️ History truncated (safety): ${conversationHistory.length} → ${limitedHistory.length} messages`);
      }

      // Add context note for survey (history is only used for survey)
      const historyPairs = limitedHistory.length / 2;
      const contextNote = `\n\n[Note: The following conversation shows your complete trial with ${historyPairs} navigation steps. Review all decisions and outcomes to evaluate the overall difficulty.]`;

      const systemMsgWithContext = systemMsg + contextNote;

      // Use conversation history (for navigation or survey)
      // Prepend system message if not already present
      const hasSystem = limitedHistory[0]?.role === 'system';
      if (hasSystem) {
        messages = limitedHistory.concat([{
          role: "user",
          content: prompt
        }]);
      } else {
        messages = [{ role: "system", content: systemMsgWithContext }].concat(limitedHistory).concat([{
          role: "user",
          content: prompt
        }]);
      }
    } else {
      // Single prompt (for navigation steps)
      messages = [
        { role: "system", content: systemMsg },
        { role: "user", content: prompt }
      ];
    }

    // Validate all messages have valid content
    messages = messages.map((msg, idx) => {
      if (!msg.content || typeof msg.content !== 'string') {
        console.warn(`Message ${idx} has invalid content:`, msg);
        return { ...msg, content: msg.content || "[empty]" };
      }
      return msg;
    });

    console.log('=== API Request Details ===');
    console.log('Model:', API_MODEL);
    console.log('Total messages:', messages.length);
    console.log('Conversation history length:', conversationHistory ? conversationHistory.length : 0);

    // Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
    const totalChars = messages.reduce((sum, msg) => sum + (msg.content?.length || 0), 0);
    console.log('Approximate input tokens:', Math.ceil(totalChars / 4));

    // Log message structure (without full content to avoid clutter)
    console.log('Message structure:', messages.map((m, i) => ({
      index: i,
      role: m.role,
      contentLength: m.content?.length || 0,
      contentPreview: (m.content || '').substring(0, 50) + '...'
    })));

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        if (apiStatusEl) apiStatusEl.innerText = `Calling OpenRouter (${API_MODEL}) - attempt ${attempt}/${retries}...`;

        const requestBody = {
          model: API_MODEL,
          messages: messages,
          temperature: API_TEMPERATURE,
          max_tokens: 1000  // Increased to 1000 for experimental models
        };

        console.log('API Request:', {
          model: API_MODEL,
          messageCount: messages.length,
          temperature: API_TEMPERATURE
        });

        // OpenAI API call (compatible with OpenRouter)
        const response = await fetch(`${API_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_KEY}`,
            'HTTP-Referer': window.location.href, // For OpenRouter rankings
            'X-Title': 'DNC Experiment'           // For OpenRouter rankings
          },
          body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
          const responseText = await response.text();
          console.error('API Error Response (raw):', responseText);

          let errorMsg;
          try {
            const error = JSON.parse(responseText);
            errorMsg = error.error?.message || error.message || JSON.stringify(error);
          } catch {
            errorMsg = `HTTP ${response.status}: ${responseText.substring(0, 200)}`;
          }
          throw new Error(errorMsg);
        }

        const responseText = await response.text();
        console.log('=== API Response (raw) ===');
        console.log('Full response:', responseText);

        let data;
        try {
          data = JSON.parse(responseText);
        } catch (e) {
          console.error('Failed to parse JSON:', responseText);
          throw new Error(`Invalid JSON response: ${responseText.substring(0, 200)}`);
        }

        console.log('=== Parsed API Response ===');
        console.log('Full data object:', data);
        console.log('Choices:', data.choices);

        if (!data.choices || !data.choices[0] || !data.choices[0].message) {
          console.error('Unexpected response structure:', data);
          throw new Error(`Unexpected API response format: ${JSON.stringify(data).substring(0, 200)}`);
        }

        // OpenAI response format
        let content = data.choices[0].message.content;

        // Check for empty or null content
        if (!content || content.trim() === '') {
          console.error('Empty response from API. Full response:', data);
          throw new Error(`Model returned empty response. This model (${API_MODEL}) may not be available or compatible. Try a different model.`);
        }

        content = content.trim();
        console.log('LLM content (before cleaning):', content.substring(0, 300));

        // Remove markdown code blocks if present
        content = content.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        console.log('LLM content (after cleaning):', content.substring(0, 300));

        // Check again after cleaning
        if (content === '') {
          console.error('Content became empty after cleaning');
          throw new Error(`Model returned only markdown formatting, no actual content. Model: ${API_MODEL}`);
        }

        let parsed;
        try {
          parsed = JSON.parse(content);
        } catch (e) {
          console.error('Failed to parse LLM response as JSON:', content);
          throw new Error(`LLM returned invalid JSON: ${content.substring(0, 200)}. Parse error: ${e.message}`);
        }

        // Validate response structure
        if (expectAction && !parsed.action) {
          throw new Error(`LLM response missing "action" field: ${JSON.stringify(parsed)}`);
        }

        console.log('Parsed LLM response:', parsed);
        if (apiStatusEl) apiStatusEl.innerText = `✓ API response received`;
        return parsed;

      } catch (error) {
        console.error(`API call attempt ${attempt} failed:`, error);
        if (attempt === retries) {
          // Enhanced error message for common issues
          let errorMsg = error.message;
          if (errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError')) {
            errorMsg = 'Network Error: Cannot reach OpenRouter. Check your internet connection and API key.';
          }
          if (apiStatusEl) apiStatusEl.innerText = `✗ ${errorMsg}`;
          throw new Error(errorMsg);
        }
        // Wait before retry (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  }

  let isRunning = false;

  async function run() {
    if (isRunning) return;
    isRunning = true;

    // Disable controls
    startBtn.disabled = true;
    humanBtn.disabled = true;
    llmBtn.disabled = true;
    startBtn.innerText = 'Running...';

    try {
      applyUI(); // read sliders
      dlBtn.disabled = true;
      runLog = { session: 'BatteryLayersV6', mode, trials: [], ui: getControls() };
      const blocks = defaults.blocks;

      for (let b = 0; b < blocks; b++) {
        const factors = {
          volatility: b % 2 === 0 ? 'low' : 'high',
          urgency: b % 2 === 0 ? 'off' : 'on',
          sensor_noise: b % 2 === 0 ? 'low' : 'high',
          map_density: b % 2 === 0 ? 'sparse' : 'dense'
        };
        const u = getControls();
        const trialsPerBlock = u.trials_per_block;
        for (let t = 0; t < trialsPerBlock; t++) {
          status.innerText = `Block ${b + 1}/${blocks} • Trial ${t + 1}/${trialsPerBlock} (${mode})`;
          await runOneTrial({ b, t, factors });
        }
      }
      status.innerText = 'All trials complete. You can download the data.';
    } catch (e) {
      console.error(e);
      status.innerText = 'Error during run: ' + e.message;
    } finally {
      isRunning = false;
      startBtn.disabled = false;
      humanBtn.disabled = false;
      llmBtn.disabled = false;
      startBtn.innerText = 'Start';
      dlBtn.disabled = false;
      analyzeBtn.disabled = false;
    }
  }

  function currentLayerDrift(drifts, r) {
    const gen = drifts[r];
    const { value } = gen.next();
    return value;
  }

  function showSurvey(reason) {
    return new Promise(resolve => {
      let chosenContext = null;
      document.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      surveyModal.style.display = 'flex';

      const reasonEl = document.getElementById('surveyReason');
      if (reasonEl) {
        reasonEl.innerText = reason ? `Trial Ended: ${reason}` : 'Trial Ended';
        reasonEl.style.color = reason === 'goal' ? '#16a34a' : '#dc2626';
      }

      if (beliefSlider) {
        beliefSlider.value = "50";
        if (beliefVal) beliefVal.innerText = "50";
      }

      const cleanup = () => {
        surveyModal.style.display = 'none';
      };

      okSurvey.onclick = () => {
        const val = Number(beliefSlider.value);
        const result = { context_belief: val };
        cleanup(); resolve(result);
      };
      skipSurvey.onclick = () => {
        const result = { context_belief: null };
        cleanup(); resolve(result);
      };
    });
  }

  async function runOneTrial({ b, t, factors }) {
    const walls = newMap(factors.map_density);
    const drifts = driftGenerators(factors.volatility);

    // Constants for this trial
    const goalRow = Math.floor(config.grid.rows / 2);
    const goalCol = config.grid.cols - 1;

    let agent = { r: goalRow, c: 0 };
    let step = 0;
    let time_left = config.trial.max_steps;
    let battery = config.battery.max;
    let currentBelief = 50; // default start
    let collisions = 0;
    let prevStep = { action: null, row: null, col: null };

    // History design:
    // - Navigation: no history (each step is independent for speed)
    // - Survey: fullHistory (complete record for evaluation)
    let fullHistory = [];

    const trialLog = { block: b, trial: t, factors, steps: [], mode };

    render(agent, walls, battery);

    // Helper to build prompt state
    const getPromptState = () => ({
      step,
      r: agent.r,
      c: agent.c,
      battery: Math.round(battery),
      time_left,
      collisions,
      urgency: factors.urgency,
      walls: walls,  // Pass the full walls set for local map generation
      prev_action: prevStep.action,
      prev_row: prevStep.row,
      prev_col: prevStep.col,
      grid_rows: config.grid.rows,
      grid_cols: config.grid.cols,
      goal_row: goalRow,
      goal_col: goalCol,
      current_belief: currentBelief
    });

    let awaiting = true;
    const onKey = (e) => {
      if (mode !== 'human' || !awaiting) return;
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      e.preventDefault();
      const mapKey = { 'ArrowUp': 'UP', 'ArrowDown': 'DOWN', 'ArrowLeft': 'LEFT', 'ArrowRight': 'RIGHT' };
      stepOnce(mapKey[e.key], '');
    };
    document.addEventListener('keydown', onKey);

    function applyDrain(collided) {
      let d = config.battery.drain_per_step + (collided ? config.battery.drain_collision_penalty : 0);
      if (factors.urgency === 'on') d += (config.battery.urgency_extra || 0);
      battery = Math.max(0, battery - d);
    }

    function atGoal() {
      return (agent.r === goalRow && agent.c === goalCol);
    }

    function endCondition() {
      if (atGoal()) return 'goal';
      if (battery <= 0) return 'battery';
      if (time_left <= 0) return 'timeout';
      return null;
    }

    function stepOnce(actionLabel, rationale) {
      if (!awaiting) return; // Prevent moves after end

      let dr = 0, dc = 0;
      if (actionLabel === 'UP') dr = -1;
      if (actionLabel === 'DOWN') dr = 1;
      if (actionLabel === 'LEFT') dc = -1;
      if (actionLabel === 'RIGHT') dc = 1;

      // 1. Proposed Move
      const nextR = clamp(agent.r + dr, 0, config.grid.rows - 1);
      const nextC = clamp(agent.c + dc, 0, config.grid.cols - 1);

      // 2. Check Wall Collision (Voluntary Move)
      let collision = 0;
      let hitWall = false;

      if (walls.has(`${nextR},${nextC}`)) {
        hitWall = true;
        collision = 1;
        // Agent stays put
      } else {
        agent.r = nextR;
        agent.c = nextC;
      }

      // 3. Apply Wind Drift
      let actualDrift = 0;
      let drift = 0;
      let nudge = 0;

      // Apply drift only based on probability
      if (Math.random() < (config.drift_probability !== undefined ? config.drift_probability : 1.0)) {
        drift = currentLayerDrift(drifts, agent.r);
        nudge = driftToNudge(drift);

        // Iterative drift check to prevent tunneling
        const driftSteps = Math.abs(nudge);
        const driftDir = Math.sign(nudge);

        for (let i = 1; i <= driftSteps; i++) {
          const checkC = agent.c + (driftDir * i);

          // Check bounds
          if (checkC < 0 || checkC >= config.grid.cols) {
            collision = 1; // Hit boundary
            break;
          }

          // Check wall
          if (walls.has(`${agent.r},${checkC}`)) {
            collision = 1; // Hit wall
            break;
          }

          // If safe, we can move here
          actualDrift = driftDir * i;
        }
      }

      agent.c += actualDrift;

      step += 1;
      time_left -= 1;

      collisions += collision;
      applyDrain(collision > 0); // Pass boolean or count? applyDrain takes boolean-ish
      render(agent, walls, battery);

      trialLog.steps.push({
        step, action: actionLabel, rationale: rationale || null,
        row: agent.r, col: agent.c, drift_row: drift, nudge, battery: Math.round(battery),
        pos: `${agent.r},${agent.c}`, collision, belief: currentBelief
      });

      const why = endCondition();
      if (why) {
        finish(why);
      } else {
        if (mode === 'llm') {
          llmPrompt.value = buildPrompt(getPromptState());
        }
        prevStep = { action: actionLabel, row: agent.r, col: agent.c };
      }
    }

    function finish(why) {
      if (!awaiting) return; // Already finished
      awaiting = false;
      document.removeEventListener('keydown', onKey);
      trialLog.end_reason = why;
      runLog.trials.push(trialLog);

      // Auto-save to localStorage
      try {
        localStorage.setItem('dsb_backup_log', JSON.stringify(runLog));
      } catch (e) { console.warn('Backup failed', e); }

      proceed(why);
    }

    async function proceed(why) {
      if (mode === 'human') {
        showSurvey(why).then(res => {
          trialLog.context_belief = res.context_belief;
          trialLog.human_context_conf = null; // deprecated
          trialLog.human_context_label = null; // deprecated
          dlBtn.disabled = false;
          resolveTrial();
        });
      } else if (mode === 'llm' && USE_API && API_KEY) {
        // LLM evaluates the trial via API (with complete memory!)
        try {
          if (apiStatusEl) apiStatusEl.innerText = 'LLM evaluating trial (with full history)...';
          const surveyPrompt = buildSurveyPrompt(trialLog);
          // Pass null for history (using summary), and false for expectAction (survey returns classification)
          const evaluation = await callAPI(surveyPrompt, 3, null, false);

          trialLog.context_belief = (typeof evaluation.belief === 'number') ? evaluation.belief : 50;
          trialLog.llm_context_label = null;
          trialLog.llm_context_conf = 0;
          trialLog.llm_reasoning = evaluation.reasoning || '';

          if (apiStatusEl) apiStatusEl.innerText = `✓ LLM evaluation: ${evaluation.classification}`;
          console.log('LLM Survey Result (with full history):', evaluation);
          console.log('Full history length:', fullHistory.length, 'messages');
        } catch (error) {
          console.error('LLM survey failed:', error);
          trialLog.llm_context_label = null;
          trialLog.llm_context_conf = 0;
          trialLog.llm_reasoning = 'Error: ' + error.message;
          if (apiStatusEl) apiStatusEl.innerText = '✗ LLM evaluation failed';
        }
        dlBtn.disabled = false;
        resolveTrial();
      } else {
        // LLM manual mode - no survey
        dlBtn.disabled = false;
        resolveTrial();
      }
    }

    let resolveTrial;
    const done = new Promise(res => { resolveTrial = res; });

    // Auto Run Logic
    let isAutoRunning = false;

    async function startAutoLoop() {
      if (isAutoRunning) return; // Already running
      isAutoRunning = true;

      // Update UI
      autoRunBtn.innerHTML = '⏹ Stop Auto Run';
      const originalBg = autoRunBtn.style.background;
      autoRunBtn.style.background = '#ef4444';

      console.log('Starting Auto Run Loop');

      while (isAutoRunning && awaiting) {
        // Clear previous response
        llmResponse.value = '';
        if (apiStatusEl) apiStatusEl.innerText = `🔄 Auto-running (Step ${step})...`;

        try {
          const currentPrompt = llmPrompt.value;

          // API Call
          const response = await callAPI(currentPrompt);

          if (!isAutoRunning || !awaiting) break;

          llmResponse.value = JSON.stringify(response, null, 2);
          const a = String(response.action || '').trim().toUpperCase();

          if (!['UP', 'DOWN', 'LEFT', 'RIGHT'].includes(a)) {
            throw new Error(`Invalid action: "${response.action}"`);
          }

          fullHistory.push({ role: "user", content: currentPrompt });
          fullHistory.push({ role: "assistant", content: JSON.stringify(response) || "{}" });

          await new Promise(resolve => setTimeout(resolve, 500));
          if (!isAutoRunning || !awaiting) break;

          if (typeof response.belief === 'number') {
            currentBelief = response.belief;
          }

          stepOnce(a, (response.rationale || '').slice(0, 200));
          await new Promise(resolve => setTimeout(resolve, 200));

        } catch (e) {
          console.error('Auto Run Error:', e);
          isAutoRunning = false;
          window.globalAutoRun = false; // Stop globally on error
          alert(`Auto Run Error: ${e.message}`);
        }
      }

      // Loop exited (either trial ended or user stopped)
      isAutoRunning = false;

      // Restore UI only if we stopped globally or trial is done
      // Actually, if trial is done, we want to keep the button state "Running" for next trial
      // But here we are inside runOneTrial scope.
      // We should restore UI if globalAutoRun is turned off.
      if (!window.globalAutoRun) {
        autoRunBtn.innerHTML = '🤖 Auto Run (API)';
        autoRunBtn.style.background = '#10b981'; // Restore green
      }
    }

    if (mode === 'llm') {
      llmPanel.style.display = 'block';
      llmPrompt.value = buildPrompt(getPromptState());

      copyPrompt.onclick = async () => { /* ... */ };

      nextStep.onclick = () => { /* ... */ };

      if (USE_API && API_KEY) {
        autoRunBtn.style.display = 'inline-block';

        // If global auto-run is on, start immediately
        if (window.globalAutoRun) {
          startAutoLoop();
        } else {
          autoRunBtn.innerHTML = '🤖 Auto Run (API)';
          autoRunBtn.style.background = '#10b981';
        }

        autoRunBtn.onclick = () => {
          if (window.globalAutoRun) {
            // Stop
            window.globalAutoRun = false;
            isAutoRunning = false; // Break local loop
            autoRunBtn.innerHTML = 'Stopping...';
          } else {
            // Start
            window.globalAutoRun = true;
            startAutoLoop();
          }
        };
      } else {
        autoRunBtn.style.display = 'none';
      }
    } else {
      llmPanel.style.display = 'none';
    }

    await done;
  }

  // Initial grid CSS sizing
  (function initGrid() {
    const r = defaults.grid.rows, c = defaults.grid.cols, cell = defaults.grid.cell;
    gridEl.style.gridTemplateRows = `repeat(${r}, ${cell}px)`;
    gridEl.style.gridTemplateColumns = `repeat(${c}, ${cell}px)`;
  })();

  // Buttons
  startBtn.onclick = () => run();
  dlBtn.onclick = () => {
    const blob = new Blob([JSON.stringify(runLog, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dsb_layers_v6_run_${Date.now()}.json`;
    document.body.appendChild(a); a.click(); a.remove();
  };

  // --- API Configuration Modal ---
  const apiConfigModal = document.getElementById('apiConfigModal');
  const configApiBtn = document.getElementById('configApiBtn');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const useApiCheckbox = document.getElementById('useApiCheckbox');
  const apiModelSelect = document.getElementById('apiModelSelect');
  const customModelInput = document.getElementById('customModelInput');
  const toggleCustomModel = document.getElementById('toggleCustomModel');
  const apiTempSlider = document.getElementById('apiTempSlider');
  const tempVal = document.getElementById('tempVal');
  const saveApiConfig = document.getElementById('saveApiConfig');
  const closeApiConfig = document.getElementById('closeApiConfig');
  const testApiBtn = document.getElementById('testApiBtn');
  const apiTestResult = document.getElementById('apiTestResult');


  // Load saved config
  if (apiKeyInput) apiKeyInput.value = API_KEY;
  if (useApiCheckbox) useApiCheckbox.checked = USE_API;

  // Helper to set model select logic
  const updateModelUI = (model) => {
    const options = Array.from(apiModelSelect.options).map(o => o.value);
    if (options.includes(model)) {
      apiModelSelect.value = model;
      customModelInput.style.display = 'none';
      apiModelSelect.style.display = 'block';
      toggleCustomModel.innerHTML = '⚙️ Use custom model ID';
    } else {
      customModelInput.style.display = 'block';
      apiModelSelect.style.display = 'none';
      customModelInput.value = model;
      toggleCustomModel.innerHTML = '⬅️ Back to model list';
    }
  };

  updateModelUI(API_MODEL);

  if (apiTempSlider) {
    apiTempSlider.value = API_TEMPERATURE;
    apiTempSlider.oninput = () => {
      if (tempVal) tempVal.innerText = apiTempSlider.value;
    };
  }
  if (tempVal) tempVal.innerText = API_TEMPERATURE;

  // Toggle Custom Model
  if (toggleCustomModel) {
    toggleCustomModel.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();

      if (apiModelSelect.style.display !== 'none' && customModelInput.style.display === 'none') {
        apiModelSelect.style.display = 'none';
        customModelInput.style.display = 'block';
        customModelInput.value = apiModelSelect.value;
        customModelInput.focus();
        toggleCustomModel.innerHTML = '⬅️ Back to model list';
      } else {
        customModelInput.style.display = 'none';
        apiModelSelect.style.display = 'block';
        const options = Array.from(apiModelSelect.options).map(o => o.value);
        if (options.includes(customModelInput.value.trim())) {
          apiModelSelect.value = customModelInput.value.trim();
        }
        toggleCustomModel.innerHTML = '⚙️ Use custom model ID';
      }
    };
  }

  if (configApiBtn) {
    configApiBtn.onclick = () => {
      if (apiConfigModal) {
        apiConfigModal.style.display = 'flex';

        // Refresh all field values from stored config
        if (apiKeyInput) apiKeyInput.value = API_KEY;
        if (useApiCheckbox) useApiCheckbox.checked = USE_API;
        if (apiTempSlider) {
          apiTempSlider.value = API_TEMPERATURE;
          if (tempVal) tempVal.innerText = API_TEMPERATURE;
        }

        // Refresh UI state for model selection
        updateModelUI(API_MODEL);
      }
    };
  }

  if (closeApiConfig) {
    closeApiConfig.onclick = () => {
      if (apiConfigModal) apiConfigModal.style.display = 'none';
    };
  }

  if (saveApiConfig) {
    saveApiConfig.onclick = () => {
      API_KEY = apiKeyInput ? apiKeyInput.value.trim() : '';
      USE_API = useApiCheckbox ? useApiCheckbox.checked : false;
      API_MODEL = (customModelInput.style.display !== 'none')
        ? customModelInput.value.trim()
        : (apiModelSelect ? apiModelSelect.value : 'openai/gpt-4o-mini');
      API_TEMPERATURE = apiTempSlider ? parseFloat(apiTempSlider.value) : 0.7;

      localStorage.setItem('api_key', API_KEY);
      localStorage.setItem('use_api', USE_API);
      localStorage.setItem('api_model', API_MODEL);
      localStorage.setItem('api_temperature', API_TEMPERATURE);

      alert('✓ API configuration saved! All models will be accessed via OpenRouter.');
      if (apiConfigModal) apiConfigModal.style.display = 'none';
    };
  }

  if (testApiBtn) {
    testApiBtn.onclick = async () => {
      testApiBtn.disabled = true;
      if (apiTestResult) {
        apiTestResult.style.display = 'block';
        apiTestResult.style.background = '#fef3c7';
        apiTestResult.style.color = '#92400e';
        apiTestResult.innerText = 'Testing OpenRouter connection...';
      }

      const tempKey = apiKeyInput ? apiKeyInput.value.trim() : '';

      if (!tempKey) {
        if (apiTestResult) {
          apiTestResult.style.background = '#fee2e2';
          apiTestResult.style.color = '#991b1b';
          apiTestResult.innerText = '✗ Please enter an OpenRouter API key';
        }
        testApiBtn.disabled = false;
        return;
      }

      try {
        // Determine model for test
        let selectedModel = (customModelInput.style.display !== 'none')
          ? customModelInput.value.trim()
          : (apiModelSelect ? apiModelSelect.value : 'openai/gpt-4o-mini');

        // Test API via OpenRouter
        const response = await fetch(`${API_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${tempKey}`,
            'HTTP-Referer': window.location.href,
            'X-Title': 'DNC Experiment Test'
          },
          body: JSON.stringify({
            model: selectedModel,
            messages: [{ role: "user", content: "Say 'test ok' in JSON: {\"status\":\"ok\"}" }],
            max_tokens: 50
          })
        });

        if (!response.ok) {
          const responseText = await response.text();
          console.error('API Test Error Response (raw):', responseText);

          let errorMsg;
          try {
            const error = JSON.parse(responseText);
            errorMsg = error.error?.message || error.message || JSON.stringify(error);
          } catch {
            errorMsg = `HTTP ${response.status}: ${responseText.substring(0, 200)}`;
          }
          throw new Error(errorMsg);
        }

        const responseText = await response.text();
        console.log('API Test Response (raw):', responseText.substring(0, 500));

        let data;
        try {
          data = JSON.parse(responseText);
        } catch (e) {
          console.error('Failed to parse JSON:', responseText);
          throw new Error(`Invalid JSON response: ${responseText.substring(0, 200)}`);
        }

        if (apiTestResult) {
          apiTestResult.style.background = '#d1fae5';
          apiTestResult.style.color = '#065f46';
          const modelName = data.model || 'unknown';
          const tokens = data.usage?.total_tokens || 0;

          apiTestResult.innerText = `✓ OpenRouter connection successful! Model: ${modelName}, Usage: ${tokens} tokens`;
        }
      } catch (error) {
        if (apiTestResult) {
          apiTestResult.style.background = '#fee2e2';
          apiTestResult.style.color = '#991b1b';

          let errorMsg = error.message;
          if (errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError')) {
            errorMsg = 'Network Error: Cannot reach OpenRouter. Check your internet connection and API key.';
          }

          apiTestResult.innerText = `✗ ${errorMsg}`;
        }
      }

      testApiBtn.disabled = false;
    };
  }

})();
