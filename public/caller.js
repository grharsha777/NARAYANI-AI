const socket = io();

const CLASSIFICATION_VISIBLE_THRESHOLD = 0.25;
const PREVIEW_INTERVAL_MS = 100;
const ANALYSIS_INTERVAL_MS = 3000;
const LOCATION_INTERVAL_MS = 5000;
const ROUTE_REFRESH_MS = 10000;
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000];

// Broken before: camera, transcript, and map state were mixed together and the video stream was throttled like a recording.
// Fixed now: preview, AI capture, transcript scroll state, and live location state are tracked separately so the preview stays live.
const state = {
  currentCallId: null,
  sessionId: `session-${Date.now()}`,
  micState: 'idle',
  recognition: null,
  cameraDesired: false,
  videoStream: null,
  previewInterval: null,
  analysisInterval: null,
  locationInterval: null,
  reconnectTimer: null,
  reconnectAttempt: 0,
  intentionalStop: false,
  transcriptPinnedToBottom: true,
  transcriptUnread: 0,
  seenTranscriptKeys: new Set(),
  map: null,
  callerMarker: null,
  responderMarker: null,
  routeLine: null,
  accuracyCircle: null,
  callerLocation: null,
  responderLocation: null,
  lastRouteFetchAt: 0,
  lastSnapshot: null,
  currentLanguage: 'en',
};

const elements = {
  micButton: document.getElementById('mic-btn'),
  micLabel: document.getElementById('mic-label'),
  transcriptList: document.getElementById('transcript-list'),
  newMessageBadge: document.getElementById('caller-new-message'),
  severityDisplay: document.getElementById('severity-display'),
  severityMeta: document.getElementById('severity-meta'),
  severityPill: document.getElementById('severity-pill'),
  liveBadge: document.getElementById('live-badge'),
  liveDot: document.getElementById('live-dot'),
  liveText: document.getElementById('live-text'),
  cameraLabel: document.getElementById('cam-label'),
  video: document.getElementById('local-video'),
  videoShell: document.getElementById('video-shell'),
  videoOverlay: document.getElementById('video-overlay'),
  overlayTitle: document.getElementById('video-overlay-title'),
  overlayText: document.getElementById('video-overlay-text'),
  videoLatency: document.getElementById('video-latency'),
  dispatchBox: document.getElementById('dispatch-box'),
  dispatchInfo: document.getElementById('dispatch-info'),
  langDisplay: document.getElementById('lang-display'),
  locationRequest: document.getElementById('location-request'),
  mapStatus: document.getElementById('map-status'),
  mapDistance: document.getElementById('map-distance'),
  callerMap: document.getElementById('caller-map'),
};

function setMicState(nextState, label) {
  state.micState = nextState;
  elements.micButton.className = `mic-btn mic-${nextState}`;
  elements.micLabel.textContent = label;
}

function transcriptKey(role, text, timestamp) {
  return `${role}:${timestamp}:${text}`;
}

function refreshTranscriptScrollState() {
  const container = elements.transcriptList;
  const distanceFromBottom = container.scrollHeight - container.clientHeight - container.scrollTop;
  state.transcriptPinnedToBottom = distanceFromBottom <= 50;
  if (state.transcriptPinnedToBottom) {
    state.transcriptUnread = 0;
    elements.newMessageBadge.classList.remove('visible');
  }
}

function scrollTranscriptToBottom(smooth = false) {
  elements.transcriptList.scrollTo({
    top: elements.transcriptList.scrollHeight,
    behavior: smooth ? 'smooth' : 'auto',
  });
  state.transcriptPinnedToBottom = true;
  state.transcriptUnread = 0;
  elements.newMessageBadge.classList.remove('visible');
}

function appendTranscriptMessage(role, text, timestamp = new Date().toISOString()) {
  if (!text) {
    return;
  }

  const key = transcriptKey(role, text, timestamp);
  if (state.seenTranscriptKeys.has(key)) {
    return;
  }

  state.seenTranscriptKeys.add(key);
  const message = document.createElement('div');
  message.className = role === 'caller' ? 'msg msg-user' : 'msg msg-ai';
  message.textContent = text;
  elements.transcriptList.appendChild(message);

  if (state.transcriptPinnedToBottom) {
    scrollTranscriptToBottom();
    return;
  }

  state.transcriptUnread += 1;
  elements.newMessageBadge.textContent = `↓ New message${state.transcriptUnread > 1 ? `s (${state.transcriptUnread})` : ''}`;
  elements.newMessageBadge.classList.add('visible');
}

function setVideoLatency(latencyMs) {
  elements.videoLatency.textContent = latencyMs != null ? `${Math.round(latencyMs)} ms live delay` : 'Live preview';
}

function setSeverity(snapshot) {
  state.lastSnapshot = snapshot || state.lastSnapshot;
  if (!snapshot || !snapshot.classification) {
    elements.severityDisplay.textContent = 'ASSESSING...';
    elements.severityMeta.textContent = 'Waiting for grounded video.';
    elements.severityPill.dataset.level = 'assessing';
    elements.severityPill.setAttribute('aria-label', 'Severity assessing');
    return;
  }

  if (snapshot.status === 'ERROR') {
    elements.severityDisplay.textContent = 'UPDATING';
    elements.severityMeta.textContent = snapshot.debug?.lastError || 'Retrying grounded analysis.';
    elements.severityPill.dataset.level = 'assessing';
    elements.severityPill.setAttribute('aria-label', 'Severity updating after analysis error');
    return;
  }

  const { classification } = snapshot;
  const confidence = Number(classification.confidence || 0);
  const visible = confidence >= CLASSIFICATION_VISIBLE_THRESHOLD;

  if (!visible) {
    elements.severityDisplay.textContent = 'ASSESSING...';
    elements.severityMeta.textContent = snapshot.status === 'LOW_CONFIDENCE'
      ? 'Grounded evidence is weak. Staying cautious.'
      : 'First grounded frame is processing.';
    elements.severityPill.dataset.level = 'assessing';
    elements.severityPill.setAttribute('aria-label', 'Severity assessing');
    return;
  }

  const lowConfidence = confidence < 0.5;
  elements.severityDisplay.textContent = `${classification.severity} · ${classification.category}`;
  elements.severityMeta.textContent = lowConfidence
    ? `Low confidence ${Math.round(confidence * 100)}%. Keep the camera steady while we verify.`
    : `Confidence ${Math.round(confidence * 100)}% · ${classification.triage_color}`;
  elements.severityPill.dataset.level = classification.severity.toLowerCase();
  elements.severityPill.setAttribute(
    'aria-label',
    `Severity ${classification.severity}, category ${classification.category}, triage ${classification.triage_color}`
  );
}

function setLiveStatus(status, detail) {
  const live = status === 'live';
  elements.liveBadge.dataset.state = status;
  elements.liveDot.dataset.state = status;
  elements.liveText.textContent = live ? 'LIVE' : 'DISCONNECTED';
  elements.liveBadge.setAttribute('aria-label', live ? 'Camera live' : 'Camera disconnected');
  elements.videoOverlay.classList.toggle('visible', !live);
  elements.overlayTitle.textContent = live ? '' : 'Camera disconnected';
  elements.overlayText.textContent = live ? '' : detail || 'Reconnecting to the live camera feed.';
}

function detectLikelyLanguage(text) {
  if (!text) {
    return 'en';
  }

  if (/[\u0900-\u097F]/.test(text)) {
    return 'hi';
  }

  if (/\b(mera|meri|madad|sahayata|khoon|dard|jal|chot|madhu|naina|haath|ungli|bachao)\b/i.test(text)) {
    return 'hi';
  }

  return 'en';
}

function normalizeAudioBytes(audioBytes) {
  if (!audioBytes) {
    return null;
  }

  if (audioBytes instanceof ArrayBuffer) {
    return audioBytes;
  }

  if (audioBytes.buffer instanceof ArrayBuffer) {
    return audioBytes.buffer;
  }

  if (typeof audioBytes === 'object') {
    const values = Object.values(audioBytes);
    if (values.length > 0) {
      return new Uint8Array(values).buffer;
    }
  }

  return null;
}

function speakWithBrowserVoice(text, language) {
  if (!('speechSynthesis' in window) || !text) {
    setMicState('idle', 'Tap to reply');
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = language === 'hi' ? 'hi-IN' : 'en-IN';
  utterance.onend = () => setMicState('idle', 'Tap to reply');
  window.speechSynthesis.speak(utterance);
}

function clearReconnectTimer() {
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function stopFramePumps() {
  if (state.previewInterval) {
    clearInterval(state.previewInterval);
    state.previewInterval = null;
  }
  if (state.analysisInterval) {
    clearInterval(state.analysisInterval);
    state.analysisInterval = null;
  }
}

function detachStream() {
  if (state.videoStream) {
    state.videoStream.getTracks().forEach((track) => track.stop());
    state.videoStream = null;
  }

  elements.video.srcObject = null;
  stopFramePumps();
}

function scheduleReconnect() {
  if (!state.cameraDesired || state.intentionalStop) {
    return;
  }

  clearReconnectTimer();
  const delay = RECONNECT_DELAYS_MS[Math.min(state.reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)];
  state.reconnectAttempt += 1;
  setLiveStatus('disconnected', `Retrying camera in ${Math.round(delay / 1000)}s.`);
  state.reconnectTimer = setTimeout(() => {
    connectCamera().catch(() => {
      scheduleReconnect();
    });
  }, delay);
}

function handleTrackEnded() {
  detachStream();
  setLiveStatus('disconnected', 'Camera disconnected. Reconnecting now.');
  scheduleReconnect();
}

function buildFrameMetrics() {
  const width = Math.max(elements.video.videoWidth || 0, 160);
  const height = Math.max(elements.video.videoHeight || 0, 90);
  const canvas = document.createElement('canvas');
  canvas.width = 160;
  canvas.height = Math.max(90, Math.round((height / width) * 160));
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(elements.video, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;

  let brightnessTotal = 0;
  let varianceTotal = 0;
  let lastLuma = null;
  for (let index = 0; index < pixels.length; index += 4) {
    const luma = (pixels[index] * 0.299) + (pixels[index + 1] * 0.587) + (pixels[index + 2] * 0.114);
    brightnessTotal += luma;
    if (lastLuma != null) {
      varianceTotal += Math.abs(luma - lastLuma);
    }
    lastLuma = luma;
  }

  const count = pixels.length / 4;
  return {
    brightness: Number((brightnessTotal / count).toFixed(2)),
    variance: Number((varianceTotal / Math.max(1, count - 1)).toFixed(2)),
  };
}

function captureFrameData(quality) {
  if (!state.currentCallId || !elements.video.srcObject || elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return null;
  }

  const canvas = document.createElement('canvas');
  const width = elements.video.videoWidth || 640;
  const height = elements.video.videoHeight || 360;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.drawImage(elements.video, 0, 0, width, height);

  return {
    callId: state.currentCallId,
    frame: canvas.toDataURL('image/jpeg', quality),
    capturedAt: new Date().toISOString(),
    metrics: buildFrameMetrics(),
  };
}

function startFramePumps() {
  stopFramePumps();

  const sendPreviewFrame = () => {
    const payload = captureFrameData(0.85);
    if (!payload) {
      return;
    }
    socket.emit('video_preview', payload);
  };

  const sendAnalysisFrame = () => {
    const payload = captureFrameData(0.8);
    if (!payload) {
      return;
    }
    socket.emit('analysis_frame', payload);
  };

  sendPreviewFrame();
  sendAnalysisFrame();
  state.previewInterval = setInterval(sendPreviewFrame, PREVIEW_INTERVAL_MS);
  state.analysisInterval = setInterval(sendAnalysisFrame, ANALYSIS_INTERVAL_MS);
}

async function connectCamera() {
  clearReconnectTimer();
  detachStream();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 60, min: 30 },
      facingMode: 'user',
    },
    audio: true,
  });

  state.videoStream = stream;
  state.intentionalStop = false;
  state.reconnectAttempt = 0;

  const video = elements.video;
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.loop = false;
  video.onended = null;
  video.srcObject = stream;
  await video.play();

  stream.getVideoTracks().forEach((track) => {
    track.onended = handleTrackEnded;
  });

  setLiveStatus('live');
  startFramePumps();
}

async function startCamera() {
  state.cameraDesired = true;
  elements.cameraLabel.textContent = 'Stop Camera';
  elements.videoShell.classList.add('active');
  setLiveStatus('disconnected', 'Connecting to your live camera...');

  try {
    await connectCamera();
  } catch (_error) {
    scheduleReconnect();
  }
}

function stopCamera() {
  state.cameraDesired = false;
  state.intentionalStop = true;
  clearReconnectTimer();
  detachStream();
  elements.cameraLabel.textContent = 'Share Camera';
  setLiveStatus('disconnected', 'Camera sharing is currently off.');
}

function ensureCallerMap() {
  if (!window.L || state.map || !elements.callerMap) {
    return;
  }

  state.map = L.map(elements.callerMap, { zoomControl: false }).setView([20.5937, 78.9629], 4);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(state.map);
}

function fitCallerMapBounds() {
  if (!state.map || !state.callerLocation) {
    return;
  }

  const points = [[state.callerLocation.lat, state.callerLocation.lng]];
  if (state.responderLocation) {
    points.push([state.responderLocation.lat, state.responderLocation.lng]);
  }

  if (points.length === 1) {
    state.map.setView(points[0], 15);
    return;
  }

  state.map.fitBounds(L.latLngBounds(points), { padding: [36, 36] });
}

function updateCallerMapMarkers() {
  ensureCallerMap();
  if (!state.map || !state.callerLocation) {
    return;
  }

  const callerLatLng = [state.callerLocation.lat, state.callerLocation.lng];
  if (!state.callerMarker) {
    state.callerMarker = L.marker(callerLatLng, {
      icon: L.divIcon({ className: 'caller-pin', html: '🚨', iconSize: [30, 30] }),
    }).bindPopup('🚨 You are here').addTo(state.map);
  } else {
    state.callerMarker.setLatLng(callerLatLng);
  }

  if (!state.accuracyCircle) {
    state.accuracyCircle = L.circle(callerLatLng, {
      radius: state.callerLocation.accuracy || 0,
      color: '#ef4444',
      fillOpacity: 0.12,
    }).addTo(state.map);
  } else {
    state.accuracyCircle.setLatLng(callerLatLng);
    state.accuracyCircle.setRadius(state.callerLocation.accuracy || 0);
  }

  if (state.responderLocation) {
    const responderLatLng = [state.responderLocation.lat, state.responderLocation.lng];
    if (!state.responderMarker) {
      state.responderMarker = L.marker(responderLatLng, {
        icon: L.divIcon({ className: 'responder-pin', html: '🚑', iconSize: [30, 30] }),
      }).bindPopup('🚑 Responder').addTo(state.map);
    } else {
      state.responderMarker.setLatLng(responderLatLng);
    }
  }

  fitCallerMapBounds();
}

async function updateCallerRoute() {
  if (!state.map || !state.callerLocation || !state.responderLocation) {
    elements.mapDistance.textContent = 'ETA --';
    elements.mapStatus.textContent = state.callerLocation ? 'Location shared. Waiting for responder route.' : 'Location pending...';
    return;
  }

  if (Date.now() - state.lastRouteFetchAt < ROUTE_REFRESH_MS - 1000) {
    return;
  }

  state.lastRouteFetchAt = Date.now();
  const { lat: callerLat, lng: callerLng } = state.callerLocation;
  const { lat: responderLat, lng: responderLng } = state.responderLocation;

  try {
    const url = `https://router.project-osrm.org/route/v1/driving/${responderLng},${responderLat};${callerLng},${callerLat}?overview=full&geometries=geojson`;
    const response = await fetch(url);
    const data = await response.json();
    const route = data.routes?.[0];
    if (!route) {
      throw new Error('No route available');
    }

    const routeCoords = route.geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    if (!state.routeLine) {
      state.routeLine = L.polyline(routeCoords, {
        color: '#ef4444',
        weight: 4,
        opacity: 0.8,
        dashArray: '10, 5',
      }).addTo(state.map);
    } else {
      state.routeLine.setLatLngs(routeCoords);
    }

    const distanceKm = (route.distance / 1000).toFixed(1);
    const etaMinutes = Math.ceil(route.duration / 60);
    elements.mapStatus.textContent = `Help is ${distanceKm} km away`;
    elements.mapDistance.textContent = `ETA ${etaMinutes} min`;
    fitCallerMapBounds();
  } catch (_error) {
    elements.mapStatus.textContent = 'Responder route unavailable right now.';
    elements.mapDistance.textContent = 'ETA --';
  }
}

async function updateCallerLocation(forcePrompt = false) {
  if (!navigator.geolocation) {
    elements.locationRequest.classList.add('visible');
    return;
  }

  try {
    const position = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: true,
        maximumAge: 5000,
        timeout: 8000,
      });
    });

    state.callerLocation = {
      lat: position.coords.latitude,
      lng: position.coords.longitude,
      accuracy: position.coords.accuracy,
    };

    elements.locationRequest.classList.remove('visible');
    updateCallerMapMarkers();
    await updateCallerRoute();

    if (state.currentCallId) {
      const eventName = forcePrompt ? 'caller-location' : 'update-caller-location';
      socket.emit(eventName, {
        callId: state.currentCallId,
        lat: state.callerLocation.lat,
        lng: state.callerLocation.lng,
        callerLat: state.callerLocation.lat,
        callerLng: state.callerLocation.lng,
        accuracy: state.callerLocation.accuracy,
      });
    }
  } catch (_error) {
    elements.locationRequest.classList.add('visible');
    elements.mapStatus.textContent = 'Please enable location to help responders find you.';
    if (forcePrompt) {
      elements.mapDistance.textContent = 'Location needed';
    }
  }
}

function startLocationUpdates() {
  if (state.locationInterval) {
    clearInterval(state.locationInterval);
  }

  state.locationInterval = setInterval(() => {
    updateCallerLocation(false).catch(() => {
      // Keep the last known location if the refresh fails.
    });
  }, LOCATION_INTERVAL_MS);
}

async function requestLocationAccess() {
  await updateCallerLocation(true);
}

async function initCall() {
  const response = await fetch('/api/calls', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ callerLanguage: 'en-IN' }),
  });
  const call = await response.json();
  state.currentCallId = call.callId;
  setSeverity(call.analysisSnapshot);
  socket.emit('request_call_analysis', { callId: state.currentCallId });
  await updateCallerLocation(true);
  startLocationUpdates();
}

window.requestLocationAccess = function requestLocationAccess() {
  updateCallerLocation(true).catch(() => {
    elements.locationRequest.classList.add('visible');
  });
};

window.toggleCamera = async function toggleCamera() {
  if (state.cameraDesired) {
    stopCamera();
    return;
  }

  await startCamera();
};

window.callEmergency = function callEmergency() {
  const anchor = document.createElement('a');
  anchor.href = 'tel:112';
  anchor.click();
  setTimeout(() => {
    alert('Emergency number: 112');
  }, 400);
};

window.toggleMic = function toggleMic() {
  if (!state.recognition) {
    alert('Speech recognition is not available in this browser.');
    return;
  }

  if (state.micState === 'idle') {
    try {
      state.recognition.start();
    } catch (_error) {
      setMicState('idle', 'Tap to speak');
    }
    return;
  }

  if (state.micState === 'listening') {
    state.recognition.stop();
    setMicState('idle', 'Tap to speak');
  }
};

function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    setMicState('idle', 'Speech recognition unavailable');
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = 'en-IN';

  state.recognition.onstart = () => {
    setMicState('listening', 'Listening...');
  };

  state.recognition.onresult = (event) => {
    let interimTranscript = '';
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      if (result.isFinal) {
        const finalTranscript = result[0].transcript.trim();
        if (!finalTranscript) {
          continue;
        }

        const timestamp = new Date().toISOString();
        const languageCode = detectLikelyLanguage(finalTranscript);
        appendTranscriptMessage('caller', finalTranscript, timestamp);
        state.recognition.stop();
        setMicState('speaking', 'AI responding...');
        socket.emit('voice_text', {
          text: finalTranscript,
          callId: state.currentCallId,
          sessionId: state.sessionId,
          languageCode,
          timestamp,
        });
      } else {
        interimTranscript += result[0].transcript;
      }
    }

    if (state.micState === 'listening') {
      elements.micLabel.textContent = interimTranscript || 'Listening...';
    }
  };

  state.recognition.onerror = () => {
    if (state.micState !== 'speaking') {
      setMicState('idle', 'Tap to speak');
    }
  };

  state.recognition.onend = () => {
    if (state.micState === 'listening') {
      setMicState('idle', 'Tap to speak');
    }
  };
}

async function pollCallState() {
  if (!state.currentCallId) {
    return;
  }

  const response = await fetch(`/api/calls/${state.currentCallId}`);
  const call = await response.json();
  setSeverity(call.analysisSnapshot);

  if (call.dispatchConfirmed) {
    elements.dispatchBox.classList.add('visible');
    elements.dispatchInfo.textContent = `${call.ambulanceUnit || 'Unit assigned'} · ETA ${call.eta || '--'} min`;
  }

  if (call.callerLanguage) {
    state.currentLanguage = String(call.callerLanguage).slice(0, 5);
    elements.langDisplay.textContent = state.currentLanguage.toUpperCase();
  }

  const responderLocation = call.responderLocation || call.ambulanceLocation;
  if (responderLocation) {
    state.responderLocation = {
      lat: responderLocation.lat,
      lng: responderLocation.lng,
      accuracy: responderLocation.accuracy,
    };
    updateCallerMapMarkers();
    await updateCallerRoute();
  }
}

socket.on('transcript update', ({ callId, entry, transcript, role, timestamp }) => {
  if (callId !== state.currentCallId) {
    return;
  }

  const nextEntry = entry || {
    role: role || 'caller',
    content: transcript || '',
    timestamp: timestamp || new Date().toISOString(),
  };
  appendTranscriptMessage(nextEntry.role, nextEntry.content, nextEntry.timestamp);
});

socket.on('audio_response', async (payload) => {
  if (payload.language) {
    state.currentLanguage = payload.language;
    elements.langDisplay.textContent = payload.language.toUpperCase();
  }

  try {
    const audioBytes = normalizeAudioBytes(payload.audioBytes);
    if (!audioBytes || !audioBytes.byteLength) {
      speakWithBrowserVoice(payload.transcript, payload.language);
      return;
    }

    const blob = new Blob([audioBytes], { type: 'audio/mpeg' });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => {
      URL.revokeObjectURL(url);
      setMicState('idle', 'Tap to reply');
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      speakWithBrowserVoice(payload.transcript, payload.language);
    };
    await audio.play();
  } catch (_error) {
    speakWithBrowserVoice(payload.transcript, payload.language);
  }
});

socket.on('pipeline_error', () => {
  setMicState('idle', 'Stay on the line');
  appendTranscriptMessage('assistant', "I'm here with you. Emergency services have been notified. Stay calm and stay on the line.");
});

socket.on('video_frame_update', ({ callId, latencyMs }) => {
  if (callId !== state.currentCallId) {
    return;
  }
  setVideoLatency(latencyMs);
});

socket.on('analysis_snapshot', ({ callId, snapshot, latencyMs }) => {
  if (callId !== state.currentCallId) {
    return;
  }

  setSeverity(snapshot);
  setVideoLatency(latencyMs ?? snapshot?.latencyMs ?? null);
});

socket.on('dispatch_status', ({ callId, dispatchConfirmed, ambulanceUnit, eta, responderLocation }) => {
  if (callId !== state.currentCallId || !dispatchConfirmed) {
    return;
  }

  elements.dispatchBox.classList.add('visible');
  elements.dispatchInfo.textContent = `${ambulanceUnit || 'Unit assigned'} · ETA ${eta || '--'} min`;
  if (responderLocation?.lat != null && responderLocation?.lng != null) {
    state.responderLocation = responderLocation;
    updateCallerMapMarkers();
    updateCallerRoute().catch(() => {
      // Keep the last map state if the route refresh fails.
    });
  }
});

socket.on('caller-location-update', ({ callId, responderLocation }) => {
  if (callId !== state.currentCallId || !responderLocation) {
    return;
  }

  state.responderLocation = responderLocation;
  updateCallerMapMarkers();
  updateCallerRoute().catch(() => {
    // Keep the last map state if the route refresh fails.
  });
});

socket.on('responder-location-update', ({ callId, lat, lng, accuracy }) => {
  if (callId !== state.currentCallId) {
    return;
  }

  state.responderLocation = { lat, lng, accuracy };
  updateCallerMapMarkers();
  updateCallerRoute().catch(() => {
    // Keep the last map state if the route refresh fails.
  });
});

function bootstrapTranscriptUI() {
  elements.transcriptList.addEventListener('scroll', refreshTranscriptScrollState);
  elements.newMessageBadge.addEventListener('click', () => scrollTranscriptToBottom(true));
}

async function bootstrap() {
  bootstrapTranscriptUI();
  setupSpeechRecognition();
  setMicState('idle', 'Tap to speak');
  setLiveStatus('disconnected', 'Camera sharing is currently off.');
  setVideoLatency(null);
  ensureCallerMap();
  await initCall();
  setInterval(() => {
    pollCallState().catch(() => {
      // Keep the current UI state if polling fails.
    });
  }, 5000);
}

bootstrap().catch(() => {
  setMicState('idle', 'Unable to initialize');
});
