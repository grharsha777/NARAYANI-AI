const socket = io();

const CLASSIFICATION_VISIBLE_THRESHOLD = 0.25;
const ROUTE_REFRESH_MS = 10000;
const RESPONDER_LOCATION_INTERVAL_MS = 10000;

function createStore(initialState) {
  let state = initialState;
  const listeners = new Set();
  return {
    getState() { return state; },
    setState(nextState) {
      state = nextState;
      listeners.forEach((listener) => listener(state));
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

const store = createStore({
  calls: [],
  selectedCallId: null,
  stats: null,
  feedbackPending: false,
  feedbackMessage: '',
  lastAlertedCriticalCallId: null,
});

const ui = {
  transcriptPinnedToBottom: true,
  transcriptOffsetFromBottom: 0,
  transcriptUnread: 0,
  selectedTranscriptCallId: null,
  map: null,
  mapContainer: null,
  callerMarker: null,
  responderMarker: null,
  accuracyCircle: null,
  routeLine: null,
  routeDistance: '--',
  routeEta: '--',
  lastRouteFetchAt: 0,
  responderLocationInterval: null,
  responderTrackingCallId: null,
  mapCallId: null,
};

const panelCache = new Map();

const elements = {
  callsList: document.getElementById('calls-list'),
  callsCount: document.getElementById('calls-count'),
  clearEndedButton: document.getElementById('clear-ended-btn'),
  statCalls: document.getElementById('stat-calls'),
  statSeverity: document.getElementById('stat-severity'),
  statResponse: document.getElementById('stat-response'),
  emptyState: document.getElementById('empty-state'),
  detailGrid: document.getElementById('detail-grid'),
};

// Broken before: the dispatcher view kept partial per-panel state, so severity, preview, transcript, and history drifted out of sync.
// Fixed now: every render reads from one atomic call snapshot and only keeps UI-only scroll/map state separately.
function defaultSnapshot(callId) {
  return {
    callId,
    status: 'ASSESSING',
    confidence: 0,
    latencyMs: null,
    classification: { severity: 'LOW', category: 'UNKNOWN', confidence: 0, triage_color: 'GREEN' },
    grounding: {
      persons_visible: 0,
      subject_positions: ['unknown'],
      visible_injuries: ['none'],
      environment: 'unknown',
      hazards_visible: ['none'],
      caller_speech_transcript: '',
      caller_emotional_state: 'unknown',
      frame_quality: 'obstructed',
      literal_summary: 'Awaiting grounded video analysis.',
    },
    analysis: {
      scene_summary: 'Awaiting grounded visual evidence.',
      what_happened: { incident_type: 'Awaiting evidence', confidence: 0, timeline: 'No grounded timeline yet.', mechanism: 'No grounded mechanism yet.' },
      patient_status: { avpu: 'UNKNOWN', breathing: 'UNKNOWN', hemorrhage: 'UNKNOWN', shock_signs: 'UNKNOWN', priority: 'UNKNOWN' },
      caller_instructions: ['Awaiting grounded assessment.'],
      do_not_do: ['Do not move the patient unless there is an immediate danger.'],
      dispatch_recommendation: { units: ['BLS_AMBULANCE'], lights_and_siren: false, hospital_prealert: 'Awaiting more evidence.', additional_info_to_gather: ['Confirm exact location and immediate symptoms.'] },
      risk_flags: ['Live assessment pending.'],
      overall_confidence: 0,
    },
    contradictions: [],
    analysisRevision: null,
    debug: { lastError: null },
  };
}

function classificationVisible(classification) {
  return Number(classification?.confidence || 0) >= CLASSIFICATION_VISIBLE_THRESHOLD;
}

function getSelectedCall(state = store.getState()) {
  return state.calls.find((call) => call.callId === state.selectedCallId) || null;
}

function withSnapshot(call) {
  if (!call) return null;
  return {
    ...call,
    analysisSnapshot: call.analysisSnapshot || defaultSnapshot(call.callId),
    transcript: Array.isArray(call.transcript) ? call.transcript : [],
    frameHistory: Array.isArray(call.frameHistory) ? call.frameHistory : [],
  };
}

function priorityScore(call) {
  const snapshot = call.analysisSnapshot || defaultSnapshot(call.callId);
  if (classificationVisible(snapshot.classification)) {
    return { CRITICAL: 100, HIGH: 80, MODERATE: 60, LOW: 40 }[snapshot.classification.severity] || 0;
  }
  return Number(call.severityScore || 0);
}

function activeCallsOnly(calls) {
  return calls.filter((call) => call.isActive !== false && !['ENDED', 'RESOLVED'].includes(call.callStatus));
}

function sortActiveCalls(calls) {
  return [...activeCallsOnly(calls)].sort((left, right) => {
    const scoreDelta = priorityScore(right) - priorityScore(left);
    if (scoreDelta !== 0) return scoreDelta;
    return new Date(right.startTime || 0).getTime() - new Date(left.startTime || 0).getTime();
  });
}

function formatTime(value, includeSeconds = false) {
  if (!value) return '--';
  return new Date(value).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: includeSeconds ? '2-digit' : undefined,
  });
}

function severityColor(triageColor = 'GREEN') {
  return { RED: '#dc2626', ORANGE: '#ea580c', YELLOW: '#ca8a04', GREEN: '#15803d' }[triageColor] || '#64748b';
}

function triageClass(triageColor = 'GREEN') {
  return `triage-${String(triageColor).toLowerCase()}`;
}

function statusAccent(call) {
  const snapshot = call.analysisSnapshot || defaultSnapshot(call.callId);
  if (classificationVisible(snapshot.classification) && snapshot.classification.severity === 'CRITICAL') return '#dc2626';
  if (call.callStatus === 'ENDED' || call.callStatus === 'RESOLVED' || call.isActive === false) return '#94a3b8';
  if (call.callStatus === 'ASSESSING' || !classificationVisible(snapshot.classification)) return '#ea580c';
  return '#16a34a';
}

function callSeverityText(call) {
  const snapshot = call.analysisSnapshot || defaultSnapshot(call.callId);
  return classificationVisible(snapshot.classification) ? snapshot.classification.severity : 'ASSESSING';
}

function renderSafePanel(id, renderer) {
  const element = document.getElementById(id);
  if (!element) return;
  try {
    const html = renderer();
    if (panelCache.get(id) !== html) {
      panelCache.set(id, html);
      element.innerHTML = html;
    }
    element.classList.remove('panel-failed');
  } catch (_error) {
    const cached = panelCache.get(id);
    element.innerHTML = cached ? `${cached}<div class="panel-overlay">Updating...</div>` : '<div class="empty-copy">Updating...</div>';
    element.classList.add('panel-failed');
  }
}

function playCriticalAlert() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  const audioContext = new AudioContextClass();
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = 'triangle';
  oscillator.frequency.value = 880;
  gain.gain.value = 0.0001;
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start();
  gain.gain.exponentialRampToValueAtTime(0.18, audioContext.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioContext.currentTime + 0.35);
  oscillator.stop(audioContext.currentTime + 0.4);
  oscillator.onended = () => audioContext.close().catch(() => {});
}

function maybePromoteCritical(prevState, nextState) {
  const topCritical = sortActiveCalls(nextState.calls).find((call) => {
    const snapshot = call.analysisSnapshot || defaultSnapshot(call.callId);
    return classificationVisible(snapshot.classification) && snapshot.classification.severity === 'CRITICAL';
  });
  if (!topCritical || nextState.lastAlertedCriticalCallId === topCritical.callId) return nextState;
  playCriticalAlert();
  return { ...nextState, selectedCallId: topCritical.callId, lastAlertedCriticalCallId: topCritical.callId };
}

function setAppState(updater) {
  const currentState = store.getState();
  const nextState = typeof updater === 'function' ? updater(currentState) : updater;
  store.setState(maybePromoteCritical(currentState, nextState));
}

function updateCall(callId, updater) {
  setAppState((state) => {
    const exists = state.calls.some((call) => call.callId === callId);
    const calls = exists
      ? state.calls.map((call) => {
          if (call.callId !== callId) return call;
          const nextCall = typeof updater === 'function' ? updater(call) : { ...call, ...updater };
          return withSnapshot(nextCall);
        })
      : [
          withSnapshot(typeof updater === 'function'
            ? updater({ callId, startTime: new Date().toISOString(), transcript: [], frameHistory: [], isActive: true })
            : { callId, startTime: new Date().toISOString(), transcript: [], frameHistory: [], isActive: true, ...updater }),
          ...state.calls,
        ];
    const nextSelected = state.selectedCallId && calls.some((call) => call.callId === state.selectedCallId && call.isActive !== false && !['ENDED', 'RESOLVED'].includes(call.callStatus))
      ? state.selectedCallId
      : (sortActiveCalls(calls)[0]?.callId || null);
    return { ...state, calls, selectedCallId: nextSelected };
  });
}

function upsertCall(nextCall) {
  updateCall(nextCall.callId, (existing) => ({ ...(existing || {}), ...nextCall }));
}

function renderCallsList(state) {
  const activeCalls = sortActiveCalls(state.calls);
  const endedCount = state.calls.filter((call) => ['ENDED', 'RESOLVED'].includes(call.callStatus) || call.isActive === false).length;
  elements.callsCount.textContent = String(activeCalls.length);
  elements.clearEndedButton.textContent = endedCount > 0 ? `Clear Ended (${endedCount})` : 'Clear Ended Calls';
  elements.clearEndedButton.disabled = endedCount === 0;

  if (activeCalls.length === 0) {
    elements.callsList.innerHTML = '<div class="empty-copy">No active calls</div>';
    return;
  }

  elements.callsList.innerHTML = activeCalls.map((call) => {
    const snapshot = call.analysisSnapshot || defaultSnapshot(call.callId);
    const active = state.selectedCallId === call.callId ? ' active' : '';
    const classificationReady = classificationVisible(snapshot.classification);
    const sublabel = classificationReady ? `${snapshot.classification.category} · ${snapshot.classification.triage_color}` : 'ASSESSING';
    const firstWords = call.firstCallerMessage || snapshot.grounding.caller_speech_transcript || 'Waiting for first caller words...';
    return `
      <div class="call-card${active}" onclick="selectCall('${call.callId}')">
        <span class="call-accent" style="background:${statusAccent(call)}"></span>
        <button class="call-close" type="button" aria-label="End ${call.callId}" onclick="confirmEndCall(event, '${call.callId}')">&times;</button>
        <div class="call-card-body">
          <div class="call-topline">
            <strong>${callSeverityText(call)}</strong>
            <span>${formatTime(call.startTime)}</span>
          </div>
          <h3>${call.emergencyType || snapshot.classification.category || 'UNKNOWN'}</h3>
          <div class="call-subtitle">${firstWords}</div>
          <p>${sublabel}</p>
          <small>${call.callId}</small>
        </div>
      </div>
    `;
  }).join('');
}

function renderStats(state) {
  elements.statCalls.textContent = state.stats?.callsToday ?? '--';
  elements.statSeverity.textContent = state.stats?.avgSeverity != null ? `${state.stats.avgSeverity}` : '--';
  elements.statResponse.textContent = state.stats?.avgResponseTime != null ? `${state.stats.avgResponseTime}s` : '--';
}

function previewPanelSkeleton() {
  return `
    <div class="skeleton skeleton-video"></div>
    <div class="skeleton skeleton-line"></div>
    <div class="skeleton skeleton-line short"></div>
  `;
}

function renderDetail(state) {
  const selected = withSnapshot(getSelectedCall(state));
  const showSelected = Boolean(selected && selected.isActive !== false && !['ENDED', 'RESOLVED'].includes(selected.callStatus));
  elements.emptyState.hidden = showSelected;
  elements.detailGrid.hidden = !showSelected;
  if (!showSelected) return;

  const snapshot = selected.analysisSnapshot || defaultSnapshot(selected.callId);
  const confidencePct = Math.round((snapshot.confidence || 0) * 100);
  const previewImage = selected.liveFrame || selected.frameHistory?.[0]?.thumbnail || '';
  const updating = snapshot.status !== 'READY';
  const latency = snapshot.latencyMs ?? selected.latestLatencyMs ?? null;
  const classification = snapshot.classification;
  const classificationReady = classificationVisible(classification);
  const lowConfidence = classificationReady && Number(classification.confidence || 0) < 0.5;
  const feedbackLabel = state.feedbackPending ? 'Saving feedback...' : (state.feedbackMessage || 'Model Learning: ON');

  // Optimized live preview: only update img.src when frame changes, avoid full innerHTML re-render
  const livePreviewEl = document.getElementById('panel-live-preview');
  if (livePreviewEl) {
    const existingImg = livePreviewEl.querySelector('.live-preview');
    if (existingImg && previewImage) {
      // Fast path: just update the image src and text, no DOM rebuild
      if (existingImg.src !== previewImage) existingImg.src = previewImage;
      const subhead = livePreviewEl.querySelector('.panel-subhead');
      if (subhead) subhead.textContent = latency != null ? `${Math.round(latency)} ms estimated delay` : 'Waiting for live latency sample';
      const statusChip = livePreviewEl.querySelector('.status-chip');
      if (statusChip) { statusChip.textContent = snapshot.status; statusChip.className = `status-chip ${updating ? 'status-updating' : 'status-ready'}`; }
      const summary = livePreviewEl.querySelector('.preview-meta span');
      if (summary) summary.textContent = snapshot.grounding.literal_summary;
    } else {
      // Full render only when structure changes (first frame or no frame)
      renderSafePanel('panel-live-preview', () => `
        <div class="panel-head">
          <div class="panel-head-left">
            <span>Caller Live Feed</span>
            <span class="panel-subhead">${latency != null ? `${Math.round(latency)} ms estimated delay` : 'Waiting for live latency sample'}</span>
          </div>
          <span class="status-chip ${updating ? 'status-updating' : 'status-ready'}">${snapshot.status}</span>
        </div>
        ${previewImage
          ? `<div class="live-preview-shell"><img class="live-preview" src="${previewImage}" alt="Caller live preview"></div>`
          : previewPanelSkeleton()}
        <div class="preview-meta">
          <span>${snapshot.grounding.literal_summary}</span>
          <span>${snapshot.grounding.frame_quality.toUpperCase()}</span>
        </div>
      `);
    }
  }

  renderSafePanel('panel-severity', () => {
    if (snapshot.status === 'ERROR') {
      return `
        <div class="severity-box triage-yellow" aria-label="Severity updating after analysis error">
          <div class="panel-head"><span>Severity Box</span><span class="status-chip">ERROR</span></div>
          <div class="severity-main">
            <div class="severity-title">UPDATING</div>
            <div class="severity-subtitle">${snapshot.debug?.lastError || 'Last grounded result held while analysis retries.'}</div>
          </div>
        </div>
      `;
    }

    const criticalClass = classification.severity === 'CRITICAL' && classificationReady ? ' critical-pulse' : '';
    const title = classificationReady ? `${classification.severity} · ${classification.category}` : 'ASSESSING...';
    const subtitle = classificationReady
      ? `${lowConfidence ? 'Low confidence' : 'Confidence'} ${Math.round((classification.confidence || 0) * 100)}%`
      : '<span class="spinner"></span> Waiting for the first grounded result.';

    return `
      <div class="severity-box ${triageClass(classification.triage_color)}${criticalClass}" aria-label="Severity ${classification.severity} category ${classification.category} triage ${classification.triage_color}">
        <div class="panel-head"><span>Severity Box</span><span class="status-chip">${classification.triage_color}</span></div>
        <div class="severity-main">
          <div class="severity-title">${title}</div>
          <div class="severity-subtitle">${subtitle}</div>
          ${lowConfidence ? '<span class="pill pill-warn">LOW CONFIDENCE</span>' : ''}
        </div>
      </div>
    `;
  });

  renderSafePanel('panel-map', () => `
    <div class="panel-head">
      <div class="panel-head-left">
        <span>Caller Location</span>
        <span class="panel-subhead">Live responder route and ETA without paid map keys.</span>
      </div>
      <button class="copy-btn" type="button" onclick="copyCallerLocation()">Copy Location</button>
    </div>
    <div class="map-stats">
      <div class="map-stat-copy">
        <strong>${selected.callerLocation ? '📍 Caller location live' : '📍 Caller location pending'}</strong>
        <span class="support-copy" id="map-distance-text">${selected.callerLocation ? 'Distance: --' : 'Distance: --'}</span>
        <span class="support-copy" id="map-eta-text">${(selected.responderLocation || selected.ambulanceLocation) ? 'ETA: --' : 'ETA: --'}</span>
      </div>
      <span class="learning-pill">${selected.callerLocation ? 'LIVE' : 'WAITING'}</span>
    </div>
    <div class="map-shell" id="dispatch-map"></div>
  `);

  renderSafePanel('panel-summary', () => `
    <div class="panel-head"><span>Scene Summary</span></div>
    <p class="body-copy">${snapshot.analysis.scene_summary}</p>
    <div class="support-copy">Grounded: ${snapshot.grounding.literal_summary}</div>
  `);

  renderSafePanel('panel-what-happened', () => `
    <div class="panel-head"><span>What Happened</span></div>
    <div class="metric-grid">
      <div><small>Incident</small><strong>${snapshot.analysis.what_happened.incident_type}</strong></div>
      <div><small>Confidence</small><strong>${Math.round((snapshot.analysis.what_happened.confidence || 0) * 100)}%</strong></div>
    </div>
    <p class="body-copy"><strong>Timeline:</strong> ${snapshot.analysis.what_happened.timeline}</p>
    <p class="body-copy"><strong>Mechanism:</strong> ${snapshot.analysis.what_happened.mechanism}</p>
  `);

  renderSafePanel('panel-patient-status', () => `
    <div class="panel-head"><span>Patient Status</span></div>
    <div class="status-grid">
      <div class="status-tile"><small>AVPU</small><strong>${snapshot.analysis.patient_status.avpu}</strong></div>
      <div class="status-tile"><small>Breathing</small><strong>${snapshot.analysis.patient_status.breathing}</strong></div>
      <div class="status-tile"><small>Hemorrhage</small><strong>${snapshot.analysis.patient_status.hemorrhage}</strong></div>
      <div class="status-tile"><small>Shock</small><strong>${snapshot.analysis.patient_status.shock_signs}</strong></div>
      <div class="status-tile"><small>Priority</small><strong>${snapshot.analysis.patient_status.priority}</strong></div>
      <div class="status-tile"><small>Visible Persons</small><strong>${snapshot.grounding.persons_visible}</strong></div>
    </div>
  `);

  renderSafePanel('panel-caller-instructions', () => `
    <div class="panel-head"><span>Caller Instructions</span></div>
    <ol class="instruction-list">
      ${(snapshot.analysis.caller_instructions || []).slice(0, 6).map((item) => `<li>${item}</li>`).join('')}
    </ol>
    <div class="support-copy"><strong>Do not:</strong> ${(snapshot.analysis.do_not_do || []).join(' ')}</div>
  `);

  renderSafePanel('panel-dispatch-recommendation', () => `
    <div class="panel-head"><span>Dispatch Recommendation</span></div>
    <div class="badge-row">
      ${(snapshot.analysis.dispatch_recommendation.units || []).map((unit) => `<span class="pill">${unit}</span>`).join('')}
      <span class="pill">${snapshot.analysis.dispatch_recommendation.lights_and_siren ? 'LIGHTS & SIREN' : 'STANDARD RESPONSE'}</span>
    </div>
    <p class="body-copy"><strong>Hospital pre-alert:</strong> ${snapshot.analysis.dispatch_recommendation.hospital_prealert}</p>
    <p class="body-copy"><strong>Gather next:</strong> ${(snapshot.analysis.dispatch_recommendation.additional_info_to_gather || []).join(', ')}</p>
  `);

  renderSafePanel('panel-risk-flags', () => `
    <div class="panel-head"><span>Risk Flags</span></div>
    <div class="badge-row">
      ${(snapshot.analysis.risk_flags || []).map((flag) => `<span class="risk-badge">${flag}</span>`).join('')}
      ${snapshot.contradictions?.length ? '<span class="risk-badge risk-badge-alert">Validator intervened</span>' : ''}
    </div>
    <div class="support-copy">Hazards visible: ${(snapshot.grounding.hazards_visible || []).join(', ')}</div>
  `);

  renderSafePanel('panel-confidence', () => `
    <div class="panel-head"><span>Confidence Meter</span></div>
    <div class="meter-shell"><div class="meter-fill" style="width:${confidencePct}%"></div></div>
    <div class="metric-grid">
      <div><small>Overall</small><strong>${confidencePct}%</strong></div>
      <div><small>Frame Quality</small><strong>${snapshot.grounding.frame_quality}</strong></div>
      <div><small>Emotion</small><strong>${snapshot.grounding.caller_emotional_state}</strong></div>
      <div><small>Environment</small><strong>${snapshot.grounding.environment}</strong></div>
    </div>
  `);

  renderSafePanel('panel-frame-history', () => `
    <div class="panel-head"><span>Frame History</span></div>
    <div class="frame-strip">
      ${(selected.frameHistory || []).slice(0, 10).map((item) => `
        <div class="frame-thumb">
          <img src="${item.thumbnail}" alt="Analyzed frame at ${formatTime(item.capturedAt, true)}">
          <span>${formatTime(item.capturedAt, true)}</span>
        </div>
      `).join('') || '<div class="empty-copy">No analyzed frames yet</div>'}
    </div>
  `);

  renderSafePanel('panel-transcript', () => `
    <div class="panel-head"><span>Transcript</span><span class="learning-pill">Model Learning: ON</span></div>
    <div class="transcript-shell">
      <div class="transcript-stack" id="dispatcher-transcript-list">
        ${(selected.transcript || []).slice(-60).map((entry) => `
          <div class="bubble ${entry.role === 'assistant' ? 'bubble-ai' : 'bubble-user'}">
            <small>${entry.role} · ${formatTime(entry.timestamp, true)}</small>
            <p>${entry.content}</p>
          </div>
        `).join('') || '<div class="empty-copy">No transcript yet</div>'}
      </div>
      <button class="new-message-badge" id="dispatcher-new-message" type="button">↓ New message</button>
    </div>
  `);

  renderSafePanel('panel-dispatch-controls', () => `
    <div class="panel-head"><span>Dispatch Controls</span></div>
    <div class="badge-row">
      <span class="pill">${selected.dispatchConfirmed ? 'DISPATCHED' : 'PENDING DISPATCH'}</span>
      <span class="pill">${selected.callerLanguage || 'EN'}</span>
      ${lowConfidence ? '<span class="pill pill-warn">VERIFY MANUALLY</span>' : ''}
    </div>
    <p class="body-copy">${selected.dispatchConfirmed ? `${selected.ambulanceUnit || 'Unit assigned'} · ETA ${selected.eta || '--'} min` : 'No unit confirmed yet.'}</p>
    <button class="primary-btn" onclick="dispatchHelp()">${selected.dispatchConfirmed ? 'Refresh Dispatch' : 'Confirm Dispatch'}</button>
  `);

  renderSafePanel('panel-feedback', () => `
    <div class="panel-head"><span>Dispatcher Feedback</span></div>
    <p class="body-copy">Mark the current AI output so grounded descriptions and corrections feed the learning log.</p>
    <div class="feedback-row">
      <button class="feedback-btn" onclick="submitFeedback('ACCURATE')" ${state.feedbackPending ? 'disabled' : ''}>Accurate</button>
      <button class="feedback-btn" onclick="submitFeedback('PARTIAL')" ${state.feedbackPending ? 'disabled' : ''}>Partial</button>
      <button class="feedback-btn" onclick="submitFeedback('INACCURATE')" ${state.feedbackPending ? 'disabled' : ''}>Inaccurate</button>
    </div>
    <div class="support-copy">${feedbackLabel}</div>
  `);

  setupTranscriptPanel(selected.callId);
  ensureDispatchMap(selected);
  ensureResponderLocationTracking(selected.callId);
}

function render(state) {
  renderStats(state);
  renderCallsList(state);
  renderDetail(state);
}

store.subscribe(render);

function refreshTranscriptBadge() {
  const badge = document.getElementById('dispatcher-new-message');
  if (!badge) return;
  if (ui.transcriptUnread > 0 && !ui.transcriptPinnedToBottom) {
    badge.textContent = `↓ New message${ui.transcriptUnread > 1 ? `s (${ui.transcriptUnread})` : ''}`;
    badge.classList.add('visible');
  } else {
    badge.classList.remove('visible');
  }
}

function setupTranscriptPanel(callId) {
  const container = document.getElementById('dispatcher-transcript-list');
  const badge = document.getElementById('dispatcher-new-message');
  if (!container || !badge) return;

  if (ui.selectedTranscriptCallId !== callId) {
    ui.selectedTranscriptCallId = callId;
    ui.transcriptPinnedToBottom = true;
    ui.transcriptUnread = 0;
    ui.transcriptOffsetFromBottom = 0;
  }

  badge.onclick = () => {
    ui.transcriptPinnedToBottom = true;
    ui.transcriptUnread = 0;
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    refreshTranscriptBadge();
  };

  if (!container.dataset.bound) {
    container.dataset.bound = 'true';
    container.addEventListener('scroll', () => {
      const offset = container.scrollHeight - container.clientHeight - container.scrollTop;
      ui.transcriptPinnedToBottom = offset <= 50;
      ui.transcriptOffsetFromBottom = Math.max(0, offset);
      if (ui.transcriptPinnedToBottom) ui.transcriptUnread = 0;
      refreshTranscriptBadge();
    });
  }

  if (ui.transcriptPinnedToBottom) {
    container.scrollTop = container.scrollHeight;
  } else {
    container.scrollTop = Math.max(0, container.scrollHeight - container.clientHeight - ui.transcriptOffsetFromBottom);
  }
  refreshTranscriptBadge();
}

function ensureMapContainer() {
  const mapElement = document.getElementById('dispatch-map');
  if (!mapElement || !window.L) return null;

  if (!ui.map || ui.mapContainer !== mapElement) {
    if (ui.map) ui.map.remove();
    ui.map = L.map(mapElement).setView([20.5937, 78.9629], 4);
    ui.mapContainer = mapElement;
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(ui.map);
    ui.callerMarker = null;
    ui.responderMarker = null;
    ui.accuracyCircle = null;
    ui.routeLine = null;
    ui.lastRouteFetchAt = 0;
  }

  return ui.map;
}

function updateMapStats(distance, eta) {
  ui.routeDistance = distance;
  ui.routeEta = eta;
  const selected = withSnapshot(getSelectedCall());
  const distanceElement = document.getElementById('map-distance-text');
  const etaElement = document.getElementById('map-eta-text');
  if (distanceElement) distanceElement.textContent = selected?.callerLocation ? `Distance: ${distance} km` : 'Distance: --';
  if (etaElement) etaElement.textContent = (selected?.responderLocation || selected?.ambulanceLocation)
    ? `ETA: ${eta} min${selected.ambulanceUnit ? ` · ${selected.ambulanceUnit}` : ''}`
    : 'ETA: --';
}

function fitDispatchMap(callerLocation, responderLocation) {
  if (!ui.map || !callerLocation) return;
  const points = [[callerLocation.lat, callerLocation.lng]];
  if (responderLocation?.lat != null && responderLocation?.lng != null) {
    points.push([responderLocation.lat, responderLocation.lng]);
  }
  if (points.length === 1) {
    ui.map.setView(points[0], 15);
    return;
  }
  ui.map.fitBounds(L.latLngBounds(points), { padding: [42, 42] });
}

async function updateRouteAndEta(selected) {
  const callerLocation = selected.callerLocation;
  const responderLocation = selected.responderLocation || selected.ambulanceLocation;
  if ((!callerLocation || !responderLocation) && ui.routeLine) {
    map.removeLayer(ui.routeLine);
    ui.routeLine = null;
  }
  if (!callerLocation || !responderLocation) {
    updateMapStats('--', '--');
    return;
  }
  if (Date.now() - ui.lastRouteFetchAt < ROUTE_REFRESH_MS - 1000) return;

  ui.lastRouteFetchAt = Date.now();
  try {
    const url = `https://router.project-osrm.org/route/v1/driving/${responderLocation.lng},${responderLocation.lat};${callerLocation.lng},${callerLocation.lat}?overview=full&geometries=geojson`;
    const response = await fetch(url);
    const data = await response.json();
    const route = data.routes?.[0];
    if (!route) throw new Error('No route available');

    const routeCoords = route.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    if (!ui.routeLine) {
      ui.routeLine = L.polyline(routeCoords, {
        color: '#FF4444',
        weight: 4,
        opacity: 0.8,
        dashArray: '10, 5',
      }).addTo(ui.map);
    } else {
      ui.routeLine.setLatLngs(routeCoords);
    }

    updateMapStats((route.distance / 1000).toFixed(1), String(Math.ceil(route.duration / 60)));
    fitDispatchMap(callerLocation, responderLocation);
  } catch (_error) {
    updateMapStats('--', '--');
  }
}

function ensureDispatchMap(selected) {
  const map = ensureMapContainer();
  if (!map) return;

  if (ui.mapCallId !== selected.callId) {
    ui.mapCallId = selected.callId;
    ui.lastRouteFetchAt = 0;
    updateMapStats('--', '--');
    if (ui.callerMarker) {
      map.removeLayer(ui.callerMarker);
      ui.callerMarker = null;
    }
    if (ui.responderMarker) {
      map.removeLayer(ui.responderMarker);
      ui.responderMarker = null;
    }
    if (ui.accuracyCircle) {
      map.removeLayer(ui.accuracyCircle);
      ui.accuracyCircle = null;
    }
    if (ui.routeLine) {
      map.removeLayer(ui.routeLine);
      ui.routeLine = null;
    }
  }

  const callerLocation = selected.callerLocation;
  const responderLocation = selected.responderLocation || selected.ambulanceLocation;

  if (callerLocation?.lat != null && callerLocation?.lng != null) {
    const callerLatLng = [callerLocation.lat, callerLocation.lng];
    if (!ui.callerMarker) {
      ui.callerMarker = L.marker(callerLatLng, {
        icon: L.divIcon({ className: 'caller-pin', html: '🚨', iconSize: [30, 30] }),
      }).bindPopup('🚨 Caller Location').addTo(map);
    } else {
      ui.callerMarker.setLatLng(callerLatLng);
    }

    if (!ui.accuracyCircle) {
      ui.accuracyCircle = L.circle(callerLatLng, {
        radius: selected.callerLocationAccuracy || callerLocation.accuracy || 0,
        color: '#FF4444',
        fillOpacity: 0.1,
      }).addTo(map);
    } else {
      ui.accuracyCircle.setLatLng(callerLatLng);
      ui.accuracyCircle.setRadius(selected.callerLocationAccuracy || callerLocation.accuracy || 0);
    }
  } else {
    if (ui.callerMarker) {
      map.removeLayer(ui.callerMarker);
      ui.callerMarker = null;
    }
    if (ui.accuracyCircle) {
      map.removeLayer(ui.accuracyCircle);
      ui.accuracyCircle = null;
    }
  }

  if (responderLocation?.lat != null && responderLocation?.lng != null) {
    const responderLatLng = [responderLocation.lat, responderLocation.lng];
    if (!ui.responderMarker) {
      ui.responderMarker = L.marker(responderLatLng, {
        icon: L.divIcon({ className: 'responder-pin', html: '🚑', iconSize: [30, 30] }),
      }).bindPopup('🚑 Responder Location').addTo(map);
    } else {
      ui.responderMarker.setLatLng(responderLatLng);
    }
  } else if (ui.responderMarker) {
    map.removeLayer(ui.responderMarker);
    ui.responderMarker = null;
  }

  fitDispatchMap(callerLocation, responderLocation);
  updateRouteAndEta(selected).catch(() => {
    updateMapStats('--', '--');
  });
}

function ensureResponderLocationTracking(callId) {
  if (!navigator.geolocation || !callId) return;
  if (ui.responderTrackingCallId === callId && ui.responderLocationInterval) return;

  const emitLocation = () => {
    navigator.geolocation.getCurrentPosition((position) => {
      socket.emit('update-responder-location', {
        callId,
        lat: position.coords.latitude,
        lng: position.coords.longitude,
        accuracy: position.coords.accuracy,
      });
    });
  };

  if (ui.responderLocationInterval) clearInterval(ui.responderLocationInterval);
  ui.responderTrackingCallId = callId;
  emitLocation();
  ui.responderLocationInterval = setInterval(emitLocation, RESPONDER_LOCATION_INTERVAL_MS);
}

function sanitizeImageUrl(value) {
  return typeof value === 'string' && value.startsWith('data:image') ? value : '';
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

async function isMeaningfullyDifferent(previousThumbnail, nextThumbnail) {
  if (!previousThumbnail || !nextThumbnail || previousThumbnail === nextThumbnail) return false;

  try {
    const [previousImage, nextImage] = await Promise.all([loadImage(previousThumbnail), loadImage(nextThumbnail)]);
    const canvas = document.createElement('canvas');
    canvas.width = 48;
    canvas.height = 27;
    const context = canvas.getContext('2d', { willReadFrequently: true });

    context.drawImage(previousImage, 0, 0, canvas.width, canvas.height);
    const previousPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(nextImage, 0, 0, canvas.width, canvas.height);
    const nextPixels = context.getImageData(0, 0, canvas.width, canvas.height).data;

    let differentPixels = 0;
    const totalPixels = canvas.width * canvas.height;
    for (let index = 0; index < previousPixels.length; index += 4) {
      const diff = Math.abs(previousPixels[index] - nextPixels[index])
        + Math.abs(previousPixels[index + 1] - nextPixels[index + 1])
        + Math.abs(previousPixels[index + 2] - nextPixels[index + 2]);
      if (diff > 60) differentPixels += 1;
    }
    return (differentPixels / totalPixels) > 0.05;
  } catch (_error) {
    return previousThumbnail !== nextThumbnail;
  }
}

function ensureUniqueTimestamp(capturedAt, history) {
  const candidate = new Date(capturedAt || Date.now()).getTime();
  const existing = new Set(history.map((item) => new Date(item.capturedAt).getTime()));
  let value = candidate;
  while (existing.has(value)) value += 1000;
  return new Date(value).toISOString();
}

async function buildNextFrameHistory(existingHistory, thumbnail, capturedAt) {
  const cleanThumbnail = sanitizeImageUrl(thumbnail);
  if (!cleanThumbnail) return existingHistory || [];
  const currentHistory = Array.isArray(existingHistory) ? existingHistory : [];
  const previous = currentHistory[0];
  const changed = previous ? await isMeaningfullyDifferent(previous.thumbnail, cleanThumbnail) : true;
  if (!changed) return currentHistory;
  const nextEntry = { thumbnail: cleanThumbnail, capturedAt: ensureUniqueTimestamp(capturedAt, currentHistory) };
  return [nextEntry, ...currentHistory].slice(0, 10);
}

window.selectCall = function selectCall(callId) {
  ui.transcriptPinnedToBottom = true;
  ui.transcriptUnread = 0;
  ui.transcriptOffsetFromBottom = 0;
  setAppState((state) => ({ ...state, selectedCallId: callId }));
  socket.emit('request_call_analysis', { callId });
};

window.confirmEndCall = async function confirmEndCall(event, callId) {
  event.stopPropagation();
  if (!window.confirm('End this call?')) return;
  await fetch(`/api/calls/${callId}/end`, { method: 'POST' });
  setAppState((state) => {
    const calls = state.calls.map((call) => call.callId === callId ? withSnapshot({ ...call, isActive: false, callStatus: 'ENDED' }) : call);
    return { ...state, calls, selectedCallId: state.selectedCallId === callId ? null : state.selectedCallId };
  });
};

window.clearEndedCalls = async function clearEndedCalls() {
  await fetch('/api/calls/ended', { method: 'DELETE' });
  setAppState((state) => {
    const calls = state.calls.filter((call) => call.isActive !== false && !['ENDED', 'RESOLVED'].includes(call.callStatus));
    return { ...state, calls, selectedCallId: sortActiveCalls(calls)[0]?.callId || null };
  });
};

window.copyCallerLocation = async function copyCallerLocation() {
  const selected = withSnapshot(getSelectedCall());
  if (!selected?.callerLocation) return;
  const text = `${selected.callerLocation.lat}, ${selected.callerLocation.lng}`;
  try {
    await navigator.clipboard.writeText(text);
    setAppState((state) => ({ ...state, feedbackMessage: 'Caller coordinates copied.' }));
  } catch (_error) {
    setAppState((state) => ({ ...state, feedbackMessage: `Caller coordinates: ${text}` }));
  }
};

window.dispatchHelp = async function dispatchHelp() {
  const selected = withSnapshot(getSelectedCall());
  if (!selected) return;

  await fetch(`/api/calls/${selected.callId}/dispatch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ambulanceUnit: selected.analysisSnapshot?.classification?.severity === 'CRITICAL' ? 'ALS-01' : 'BLS-03',
      paramedicName: 'Dispatcher Assigned',
      eta: selected.analysisSnapshot?.classification?.severity === 'CRITICAL' ? 4 : 7,
      ambulanceLocation: selected.responderLocation || null,
    }),
  });

  await loadData();
};

window.submitFeedback = async function submitFeedback(assessment) {
  const selected = withSnapshot(getSelectedCall());
  if (!selected || !selected.analysisSnapshot?.analysisRevision) {
    setAppState((state) => ({ ...state, feedbackMessage: 'No analysis revision available for feedback.' }));
    return;
  }

  setAppState((state) => ({ ...state, feedbackPending: true, feedbackMessage: 'Saving feedback...' }));
  try {
    await fetch(`/api/calls/${selected.callId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysisRevision: selected.analysisSnapshot.analysisRevision, assessment }),
    });
    setAppState((state) => ({ ...state, feedbackPending: false, feedbackMessage: `Feedback saved: ${assessment}` }));
  } catch (_error) {
    setAppState((state) => ({ ...state, feedbackPending: false, feedbackMessage: 'Unable to save feedback right now.' }));
  }
};

async function loadData() {
  const [callsResponse, statsResponse] = await Promise.all([
    fetch('/api/calls').then((response) => response.json()),
    fetch('/api/stats').then((response) => response.json()),
  ]);

  const previousCalls = store.getState().calls;
  const calls = Array.isArray(callsResponse)
    ? callsResponse.map((serverCall) => {
        const existing = previousCalls.find((call) => call.callId === serverCall.callId);
        return withSnapshot({
          ...(existing || {}),
          ...serverCall,
          liveFrame: existing?.liveFrame || '',
          latestLatencyMs: existing?.latestLatencyMs ?? null,
          frameHistory: existing?.frameHistory || [],
        });
      })
    : [];

  const sortedActive = sortActiveCalls(calls);
  const selectedStillExists = sortedActive.some((call) => call.callId === store.getState().selectedCallId);
  setAppState((state) => ({
    ...state,
    calls,
    stats: statsResponse,
    selectedCallId: selectedStillExists ? state.selectedCallId : (sortedActive[0]?.callId || null),
  }));
}

socket.on('transcript update', ({ callId, entry, transcript, role }) => {
  if (callId === store.getState().selectedCallId && !ui.transcriptPinnedToBottom) ui.transcriptUnread += 1;
  updateCall(callId, (call) => {
    const nextEntry = entry || { role: role || 'caller', content: transcript || '', timestamp: new Date().toISOString() };
    return {
      ...call,
      transcript: [...(call.transcript || []), nextEntry].slice(-80),
      firstCallerMessage: call.firstCallerMessage || (nextEntry.role === 'caller' ? nextEntry.content.slice(0, 110) : call.firstCallerMessage),
    };
  });
});

socket.on('video_frame_update', ({ callId, frame, latencyMs }) => {
  updateCall(callId, (call) => ({
    ...call,
    liveFrame: sanitizeImageUrl(frame) || call.liveFrame,
    latestLatencyMs: latencyMs ?? call.latestLatencyMs ?? null,
  }));
});

socket.on('analysis_snapshot', async ({ callId, snapshot, thumbnail, capturedAt, latencyMs }) => {
  const current = withSnapshot(store.getState().calls.find((call) => call.callId === callId) || { callId, transcript: [], frameHistory: [] });
  const frameHistory = await buildNextFrameHistory(current.frameHistory, thumbnail, capturedAt);
  updateCall(callId, (call) => ({
    ...call,
    analysisSnapshot: snapshot,
    severityScore: classificationVisible(snapshot.classification) ? ({ CRITICAL: 9, HIGH: 7, MODERATE: 5, LOW: 3 }[snapshot.classification.severity] || 0) : 0,
    severityLabel: classificationVisible(snapshot.classification) ? snapshot.classification.severity : 'Assessing',
    emergencyType: classificationVisible(snapshot.classification) ? snapshot.classification.category : (call.emergencyType || 'UNKNOWN'),
    callerEmotion: snapshot.grounding.caller_emotional_state,
    backgroundSound: snapshot.grounding.environment,
    callStatus: snapshot.status === 'READY' ? 'ACTIVE' : (call.callStatus === 'ENDED' ? 'ENDED' : 'ASSESSING'),
    liveFrame: sanitizeImageUrl(thumbnail) || call.liveFrame,
    latestLatencyMs: latencyMs ?? snapshot.latencyMs ?? call.latestLatencyMs ?? null,
    frameHistory,
  }));
});

socket.on('call_meta_update', (payload) => {
  upsertCall({ ...payload, analysisSnapshot: payload.analysisSnapshot || defaultSnapshot(payload.callId) });
});

socket.on('dispatch_status', (payload) => {
  updateCall(payload.callId, (call) => ({
    ...call,
    dispatchConfirmed: payload.dispatchConfirmed,
    ambulanceUnit: payload.ambulanceUnit,
    eta: payload.eta,
    responderLocation: payload.responderLocation || call.responderLocation,
  }));
});

socket.on('combined severity update', ({ callId, combinedSeverity, severityLabel }) => {
  updateCall(callId, (call) => ({
    ...call,
    severityScore: Math.max(Number(call.severityScore || 0), Number(combinedSeverity || 0)),
    severityLabel: call.severityLabel === 'Critical' ? call.severityLabel : severityLabel,
  }));
});

socket.on('feedback_logged', ({ callId, assessment }) => {
  updateCall(callId, (call) => ({ ...call, lastFeedbackAssessment: assessment }));
});

socket.on('call_ended', ({ callId }) => {
  updateCall(callId, (call) => ({ ...call, isActive: false, callStatus: 'ENDED' }));
});

socket.on('caller-location-update', ({ callId, lat, lng, accuracy, responderLocation }) => {
  updateCall(callId, (call) => ({
    ...call,
    callerLocation: { lat, lng, accuracy },
    callerLocationAccuracy: accuracy,
    responderLocation: responderLocation || call.responderLocation,
  }));
});

socket.on('responder-location-update', ({ callId, lat, lng, accuracy, callerLocation }) => {
  updateCall(callId, (call) => ({
    ...call,
    responderLocation: { lat, lng, accuracy },
    callerLocation: callerLocation || call.callerLocation,
  }));
});

elements.clearEndedButton.addEventListener('click', () => {
  window.clearEndedCalls().catch(() => {
    setAppState((state) => ({ ...state, feedbackMessage: 'Unable to clear ended calls.' }));
  });
});

loadData().catch(() => {
  setAppState((state) => ({ ...state, feedbackMessage: 'Unable to load initial data.' }));
});

setInterval(() => {
  loadData().catch(() => {
    // Keep the current state if polling fails.
  });
}, 5000);
