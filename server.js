const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const http = require('http');
const { Server } = require('socket.io');
const { connectDB, connectRedis, redisClient, Call, Protocol, Conversation, AnalysisFeedback } = require('./db');
const { getHumanLikeResponse, murfSTT, murfTTS } = require('./murf_multilingual_service');
const { groundingPass, analysisPass, classificationPass } = require('./ai_pipeline');

const SEVERITY_SERVICE_URL = process.env.SEVERITY_SERVICE_URL || 'http://localhost:5050';
const PORT = process.env.PORT || 3000;
const AI_ANALYSIS_INTERVAL_MS = 3000;
const CLASSIFICATION_VISIBLE_THRESHOLD = 0.25;
const FORCED_CLASSIFICATION_TIMEOUT_MS = 10000;
const GENUINE_ASSESSING_WINDOW_MS = 5000;

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

let firstAidProtocols = {};
const callRuntime = new Map();

app.use(cors());
app.use(express.json({ limit: '12mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.get('/caller', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'caller.html')));
app.get('/dispatcher', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'dispatcher.html')));

function getRuntime(callId) {
  if (!callRuntime.has(callId)) {
    callRuntime.set(callId, {
      recentCallerTranscripts: [],
      groundingInFlight: false,
      latestPreviewFrame: null,
      latestPreviewCapturedAt: null,
      latestAnalysisFrame: null,
      latestAnalysisCapturedAt: null,
      latestFrameMetrics: null,
      firstVideoAt: null,
      lastAnalysisAt: 0,
      lastSnapshot: null,
      lastAnalysis: null,
      successfulClassifications: 0,
      lastVoiceAt: 0,
    });
  }
  return callRuntime.get(callId);
}

function normalizeRole(role) {
  return role === 'assistant' || role === 'narayani' ? 'assistant' : 'caller';
}

function isClassificationVisible(classification) {
  return Number(classification?.confidence || 0) >= CLASSIFICATION_VISIBLE_THRESHOLD;
}

function scoreFromSeverity(severity) {
  return severity === 'CRITICAL' ? 9 : severity === 'HIGH' ? 7 : severity === 'MODERATE' ? 5 : severity === 'LOW' ? 3 : 0;
}

function severityLabelFromClassification(classification) {
  if (!isClassificationVisible(classification)) return 'Assessing';
  return classification.severity === 'CRITICAL' ? 'Critical' : classification.severity === 'HIGH' ? 'High' : classification.severity === 'MODERATE' ? 'Moderate' : 'Low';
}

function createDefaultSnapshot(callId) {
  return {
    callId,
    snapshotRevision: `${callId}-initial`,
    analysisRevision: null,
    generatedAt: new Date().toISOString(),
    status: 'ASSESSING',
    analysisFresh: false,
    confidence: 0.1,
    latencyMs: null,
    grounding: {
      persons_visible: 0,
      subject_positions: ['unknown'],
      visible_injuries: ['none'],
      environment: 'unknown',
      hazards_visible: ['none'],
      caller_speech_transcript: '',
      caller_emotional_state: 'unknown',
      frame_quality: 'obstructed',
      literal_summary: 'Awaiting live video and caller speech.',
    },
    classification: { severity: 'LOW', category: 'UNKNOWN', confidence: 0, triage_color: 'GREEN' },
    analysis: {
      scene_summary: 'Awaiting grounded visual evidence.',
      what_happened: { incident_type: 'Awaiting evidence', confidence: 0, timeline: 'No grounded timeline yet.', mechanism: 'No grounded mechanism yet.' },
      patient_status: { avpu: 'UNKNOWN', breathing: 'UNKNOWN', hemorrhage: 'UNKNOWN', shock_signs: 'UNKNOWN', priority: 'UNKNOWN' },
      caller_instructions: ['Activate the camera if safe and stay connected to the dispatcher.'],
      do_not_do: ['Do not move anyone unless there is an immediate danger.'],
      dispatch_recommendation: { units: ['BLS_AMBULANCE'], lights_and_siren: false, hospital_prealert: 'Awaiting more evidence.', additional_info_to_gather: ['Confirm location and describe what is happening.'] },
      risk_flags: ['Live assessment pending.'],
      overall_confidence: 0,
    },
    contradictions: [],
    debug: { groundedAt: null, classifiedAt: null, analyzedAt: null, lastError: null },
  };
}

function pushTranscriptWindow(callId, entry) {
  const runtime = getRuntime(callId);
  runtime.recentCallerTranscripts.push({ role: normalizeRole(entry.role), content: String(entry.content || ''), timestamp: new Date(entry.timestamp || Date.now()) });
  runtime.recentCallerTranscripts = runtime.recentCallerTranscripts.filter((item) => Date.now() - new Date(item.timestamp).getTime() <= 30000).slice(-20);
}

async function getRecentCallerTranscript(callId) {
  const runtime = getRuntime(callId);
  const recent = runtime.recentCallerTranscripts.filter((entry) => entry.role === 'caller' && Date.now() - new Date(entry.timestamp).getTime() <= 15000).map((entry) => entry.content.trim()).filter(Boolean);
  if (recent.length > 0) return recent.join(' ');
  const call = await Call.findOne({ callId }).select('transcript').lean();
  if (!call || !Array.isArray(call.transcript)) return '';
  return call.transcript.filter((entry) => normalizeRole(entry.role) === 'caller').slice(-3).map((entry) => entry.content).join(' ');
}

function deriveFrameQuality(metrics = {}, grounding = {}) {
  const brightness = Number(metrics.brightness || 0);
  const variance = Number(metrics.variance || 0);
  const faceVisible = grounding.persons_visible > 0 || (grounding.subject_positions || []).some((position) => position.includes('face'));
  if (brightness < 25) return 'dark';
  if (faceVisible) return 'clear';
  if (variance < 18) return 'blurry';
  return grounding.frame_quality || 'clear';
}

function refineGrounding(grounding, metrics = {}) {
  const refined = { ...grounding, frame_quality: deriveFrameQuality(metrics, grounding) };
  if (refined.persons_visible > 0 && refined.frame_quality === 'obstructed') refined.frame_quality = 'clear';
  if (refined.caller_emotional_state === 'unknown' && refined.persons_visible > 0) refined.caller_emotional_state = 'calm';
  if (refined.environment === 'unknown' && /room|wall|indoor|home/i.test(refined.literal_summary)) refined.environment = 'indoor_home';
  if (refined.environment === 'unknown' && /road|street|vehicle|car/i.test(refined.literal_summary)) refined.environment = 'outdoor_road';
  return refined;
}

function forceClassificationIfNeeded(classification, grounding, transcript, runtime) {
  const visibleEvidence = grounding.persons_visible > 0 || grounding.literal_summary || transcript || (grounding.environment && grounding.environment !== 'unknown');
  const videoAge = runtime.firstVideoAt ? Date.now() - runtime.firstVideoAt : 0;
  if (visibleEvidence && videoAge >= FORCED_CLASSIFICATION_TIMEOUT_MS && Number(classification.confidence || 0) < CLASSIFICATION_VISIBLE_THRESHOLD) {
    return { ...classification, confidence: CLASSIFICATION_VISIBLE_THRESHOLD, triage_color: classification.triage_color || 'YELLOW' };
  }
  return classification;
}

function calculatePipelineConfidence(runtime, grounding, classification, transcript) {
  let confidence = 0;
  if (runtime.firstVideoAt) confidence = 0.35;
  if (transcript) confidence = Math.max(confidence, 0.58);
  if (grounding.persons_visible > 0 || grounding.environment !== 'unknown') confidence = Math.max(confidence, 0.72);
  if (isClassificationVisible(classification)) confidence = Math.max(confidence, 0.82);
  if (runtime.successfulClassifications >= 2 && isClassificationVisible(classification)) confidence = Math.max(confidence, 0.93);
  confidence += Math.min(0.08, Number(classification.confidence || 0) * 0.08);
  return Math.max(0, Math.min(1, Number(confidence.toFixed(2))));
}

function buildSnapshot({ callId, grounding, classification, analysis, contradictions, analysisFresh, analysisRevision, latencyMs, runtime }) {
  const transcript = grounding.caller_speech_transcript || '';
  const confidence = calculatePipelineConfidence(runtime, grounding, classification, transcript);
  const withinAssessingWindow = runtime.firstVideoAt && (Date.now() - runtime.firstVideoAt) <= GENUINE_ASSESSING_WINDOW_MS;
  const status = isClassificationVisible(classification) ? 'READY' : withinAssessingWindow ? 'ASSESSING' : 'LOW_CONFIDENCE';
  return {
    callId,
    snapshotRevision: `${callId}-${Date.now()}`,
    analysisRevision,
    generatedAt: new Date().toISOString(),
    status,
    analysisFresh,
    confidence,
    latencyMs,
    grounding,
    classification,
    analysis: { ...analysis, overall_confidence: confidence },
    contradictions,
    debug: { groundedAt: new Date().toISOString(), classifiedAt: new Date().toISOString(), analyzedAt: analysisFresh ? new Date().toISOString() : runtime.lastSnapshot?.debug?.analyzedAt || null, lastError: null },
  };
}

async function logAnalysisSnapshot(callId, snapshot) {
  if (!snapshot.analysisRevision) return;
  await AnalysisFeedback.findOneAndUpdate(
    { callId, analysisRevision: snapshot.analysisRevision },
    { callId, analysisRevision: snapshot.analysisRevision, generatedAt: new Date(snapshot.generatedAt), grounding: snapshot.grounding, classification: snapshot.classification, analysis: snapshot.analysis, confidence: snapshot.confidence, contradictions: snapshot.contradictions },
    { upsert: true, new: true }
  );
}

function deriveCallStatus(call, snapshot) {
  if (call.callStatus === 'ENDED' || call.callStatus === 'RESOLVED' || call.isActive === false) return call.callStatus || 'ENDED';
  return isClassificationVisible(snapshot.classification) ? 'ACTIVE' : 'ASSESSING';
}

async function persistAnalysisSnapshot(callId, snapshot) {
  const call = await Call.findOne({ callId }).lean();
  if (!call) return;
  await Call.findOneAndUpdate(
    { callId },
    {
      severityScore: isClassificationVisible(snapshot.classification) ? scoreFromSeverity(snapshot.classification.severity) : 0,
      severityLabel: severityLabelFromClassification(snapshot.classification),
      emergencyType: isClassificationVisible(snapshot.classification) ? snapshot.classification.category : call.emergencyType || 'UNKNOWN',
      callerEmotion: snapshot.grounding.caller_emotional_state,
      backgroundSound: snapshot.grounding.environment,
      callStatus: deriveCallStatus(call, snapshot),
      visionAnalysis: { grounding: snapshot.grounding, classification: snapshot.classification, analysis: snapshot.analysis, contradictions: snapshot.contradictions },
      analysisSnapshot: snapshot,
    }
  );
}

async function emitCallMetaUpdate(callId, overrides = {}) {
  const call = await Call.findOne({ callId }).lean();
  if (!call) return;
  io.emit('call_meta_update', {
    callId,
    severityScore: call.severityScore,
    severityLabel: call.severityLabel,
    emergencyType: call.emergencyType,
    callStatus: call.callStatus,
    firstCallerMessage: call.firstCallerMessage,
    callerEmotion: call.callerEmotion,
    callerLanguage: call.callerLanguage,
    backgroundSound: call.backgroundSound,
    dispatchConfirmed: call.dispatchConfirmed,
    ambulanceUnit: call.ambulanceUnit,
    eta: call.eta,
    callerLocation: call.callerLocation,
    callerLocationAccuracy: call.callerLocationAccuracy,
    responderLocation: call.responderLocation || call.ambulanceLocation,
    analysisSnapshot: call.analysisSnapshot,
    ...overrides,
  });
}

async function appendTranscriptEntry(callId, role, content, timestamp = new Date()) {
  const entry = { role, content, timestamp: new Date(timestamp) };
  pushTranscriptWindow(callId, entry);
  const call = await Call.findOne({ callId });
  if (!call) return entry;
  call.transcript.push(entry);
  if (!call.firstCallerMessage && role === 'caller') call.firstCallerMessage = content.slice(0, 110);
  await call.save();
  io.emit('transcript update', { callId, entry });
  return entry;
}

async function predictTranscriptSeverity(transcript) {
  try {
    const mlResponse = await axios.post(`${SEVERITY_SERVICE_URL}/predict`, { transcript }, { timeout: 3000 });
    const risk = mlResponse.data?.severity;
    return {
      risk,
      score: risk === 'Critical' ? 9 : risk === 'High' ? 7 : risk === 'Medium' ? 5 : risk === 'Low' ? 3 : 0,
      votes: { rf: mlResponse.data?.rf_label || risk || 'Unknown', xgb: mlResponse.data?.xgb_label || risk || 'Unknown', lgbm: mlResponse.data?.lgbm_label || risk || 'Unknown', allAgreed: Boolean(mlResponse.data?.allAgreed) },
    };
  } catch (_error) {
    return { risk: 'Unknown', score: 0, votes: { rf: 'Unknown', xgb: 'Unknown', lgbm: 'Unknown', allAgreed: false } };
  }
}

function buildCallerContext(callId, languageCode = 'en') {
  const snapshot = getRuntime(callId).lastSnapshot || createDefaultSnapshot(callId);
  return {
    preferredLanguage: languageCode,
    groundedSummary: snapshot.grounding.literal_summary,
    visibleInjuries: (snapshot.grounding.visible_injuries || []).join(', '),
    environment: snapshot.grounding.environment,
    callerEmotion: snapshot.grounding.caller_emotional_state,
  };
}

async function handleCallerText({ text, callId, sessionId, languageCode = 'en', timestamp }, socket) {
  console.log(`[Transcription] ${callId}: "${text}"`);
  if (!text || !callId) return;

  try {
    await appendTranscriptEntry(callId, 'caller', text, timestamp);
    const runtime = getRuntime(callId);
    runtime.lastVoiceAt = Date.now();
    
    if (sessionId) {
      await Conversation.findOneAndUpdate(
        { sessionId },
        { 
          $push: { messages: { role: 'user', content: text, languageCode, timestamp: new Date() } },
          $inc: { 'metadata.totalMessages': 1 }
        },
        { upsert: true, new: true }
      );
    }

    const responseStartedAt = Date.now();
    const conversationHistory = runtime.recentCallerTranscripts.slice(-10);
    
    console.log(`[AI Response] Generating for ${callId}...`);
    const aiResult = await getHumanLikeResponse(text, buildCallerContext(callId, languageCode), conversationHistory);
    const responseTimeSeconds = Number(((Date.now() - responseStartedAt) / 1000).toFixed(2));
    console.log(`[AI Response] SUCCESS in ${responseTimeSeconds}s: "${aiResult.response.slice(0, 50)}..."`);

    // --- ML Severity & Classification logic ---
    const mlSeverity = await predictTranscriptSeverity(text);
    const voiceScore = Math.max(Number(aiResult.severity || 0), mlSeverity.score);

    const grounding = runtime.lastSnapshot?.grounding || createDefaultSnapshot(callId).grounding;
    grounding.caller_speech_transcript = text;
    
    const classification = forceClassificationIfNeeded(
      await classificationPass({ grounding, mlSeverity }),
      grounding,
      text,
      runtime
    );
    if (isClassificationVisible(classification)) runtime.successfulClassifications += 1;

    const analysis = runtime.lastAnalysis || createDefaultSnapshot(callId).analysis;
    const latencyMs = runtime.lastSnapshot?.latencyMs || null;
    const snapshot = buildSnapshot({
      callId, grounding, classification, analysis,
      contradictions: runtime.lastSnapshot?.contradictions || [],
      analysisFresh: false,
      analysisRevision: `${callId}-voice-${Date.now()}`,
      latencyMs,
      runtime,
    });
    runtime.lastSnapshot = snapshot;

    // --- Persist & Notify ---
    await Call.findOneAndUpdate(
      { callId },
      {
        callerLanguage: aiResult['detected language'] || languageCode,
        emergencyType: isClassificationVisible(classification) ? classification.category : (aiResult['emergency type'] || 'UNKNOWN'),
        callerEmotion: aiResult['caller emotion'] || 'unknown',
        backgroundSound: aiResult['background context'] || 'unknown',
        aiBrainUsed: 'NARAYANI CASCADE',
        aiBrainResponseTime: responseTimeSeconds,
        severityScore: Math.max(voiceScore, scoreFromSeverity(classification.severity)),
        severityLabel: severityLabelFromClassification(classification),
        mlVotes: mlSeverity.votes,
        callStatus: isClassificationVisible(classification) ? 'ACTIVE' : (voiceScore > 0 ? 'ACTIVE' : 'ASSESSING'),
        analysisSnapshot: snapshot,
        visionAnalysis: { grounding: snapshot.grounding, classification: snapshot.classification, analysis: snapshot.analysis, contradictions: snapshot.contradictions }
      }
    );

    await appendTranscriptEntry(callId, 'assistant', aiResult.response);
    
    if (sessionId) {
      await Conversation.findOneAndUpdate(
        { sessionId },
        { 
          $push: { messages: { role: 'assistant', content: aiResult.response, languageCode: aiResult['detected language'] || languageCode, llmProvider: 'cascade', responseTimeMs: responseTimeSeconds * 1000, timestamp: new Date() } },
          $inc: { 'metadata.totalMessages': 1 },
          $set: { 'metadata.primaryLanguage': aiResult['detected language'] || languageCode }
        },
        { upsert: true, new: true }
      );
    }

    // --- Send Audio & Updates ---
    console.log(`[TTS] Generating audio for response...`);
    const audioBytes = await murfTTS(aiResult.response, aiResult['detected language'] || languageCode);
    socket.emit('audio_response', { audioBytes, transcript: aiResult.response, language: aiResult['detected language'] || languageCode });
    
    io.emit('analysis_snapshot', { callId, snapshot, thumbnail: runtime.latestPreviewFrame || '', capturedAt: new Date().toISOString(), latencyMs });
    io.emit('combined severity update', { callId, combinedSeverity: Math.max(voiceScore, scoreFromSeverity(classification.severity)), severityLabel: severityLabelFromClassification(classification) });
    await emitCallMetaUpdate(callId, { analysisSnapshot: snapshot });

  } catch (error) {
    console.error(`[handleCallerText] CRITICAL ERROR: ${error.message}`);
    socket.emit('pipeline_error', { error: 'Failed to process voice response' });
  }
}

async function processVisionFrame(callId, frameDataUrl, frameMeta = {}, options = {}) {
  const runtime = getRuntime(callId);
  const now = Date.now();
  if (runtime.groundingInFlight || (!options.force && now - runtime.lastAnalysisAt < AI_ANALYSIS_INTERVAL_MS - 500)) return;
  runtime.groundingInFlight = true;
  runtime.latestAnalysisFrame = frameDataUrl;
  runtime.latestAnalysisCapturedAt = frameMeta.capturedAt || new Date().toISOString();
  runtime.latestFrameMetrics = frameMeta.metrics || null;
  if (!runtime.firstVideoAt) runtime.firstVideoAt = now;

  try {
    console.log(`[AI Pipeline] frame sent for call ${callId}, size=${frameDataUrl.length}`);
    const transcript = await getRecentCallerTranscript(callId);
    const grounding = refineGrounding(await groundingPass({ frameDataUrl, transcript }), frameMeta.metrics || {});
    console.log(`[AI Pipeline] grounding completed for ${callId}: ${JSON.stringify(grounding)}`);
    let classification = forceClassificationIfNeeded(await classificationPass({ grounding }), grounding, transcript, runtime);
    if (isClassificationVisible(classification)) runtime.successfulClassifications += 1;
    console.log(`[AI Pipeline] classification completed for ${callId}: ${JSON.stringify(classification)}`);

    let analysis = runtime.lastAnalysis || createDefaultSnapshot(callId).analysis;
    let contradictions = [];
    let analysisFresh = false;
    if (!runtime.lastAnalysis || now - runtime.lastAnalysisAt >= AI_ANALYSIS_INTERVAL_MS || options.force) {
      const analysisResult = await analysisPass({ grounding, classification });
      analysis = analysisResult.analysis;
      contradictions = analysisResult.contradictions;
      analysisFresh = true;
      runtime.lastAnalysis = analysis;
      console.log(`[AI Pipeline] analysis completed for ${callId}: ${JSON.stringify(analysis)}`);
    }

    const latencyMs = frameMeta.capturedAt ? Math.max(0, Date.now() - new Date(frameMeta.capturedAt).getTime()) : null;
    const snapshot = buildSnapshot({ callId, grounding, classification, analysis, contradictions, analysisFresh, analysisRevision: `${callId}-snapshot-${Date.now()}`, latencyMs, runtime });
    runtime.lastSnapshot = snapshot;
    runtime.lastAnalysisAt = now;
    await persistAnalysisSnapshot(callId, snapshot);
    await logAnalysisSnapshot(callId, snapshot);
    io.emit('analysis_snapshot', { callId, snapshot, thumbnail: frameDataUrl, capturedAt: frameMeta.capturedAt || new Date().toISOString(), latencyMs });
    await emitCallMetaUpdate(callId, { analysisSnapshot: snapshot });
  } catch (error) {
    console.error(`[AI Pipeline] ${callId} failed: ${error.message}`);
    const fallbackSnapshot = runtime.lastSnapshot ? { ...runtime.lastSnapshot, snapshotRevision: `${callId}-fallback-${Date.now()}`, generatedAt: new Date().toISOString(), status: 'ERROR', debug: { ...(runtime.lastSnapshot.debug || {}), lastError: error.message } } : { ...createDefaultSnapshot(callId), status: 'ERROR', debug: { lastError: error.message } };
    io.emit('analysis_snapshot', { callId, snapshot: fallbackSnapshot, thumbnail: frameDataUrl, capturedAt: frameMeta.capturedAt || new Date().toISOString(), latencyMs: frameMeta.capturedAt ? Math.max(0, Date.now() - new Date(frameMeta.capturedAt).getTime()) : null });
  } finally {
    runtime.groundingInFlight = false;
  }
}

async function updateCallerLocation(callId, lat, lng, accuracy) {
  const call = await Call.findOneAndUpdate({ callId }, { callerLocation: { lat, lng }, callerLocationAccuracy: accuracy || null }, { new: true }).lean();
  if (!call) return;
  io.emit('caller-location-update', { callId, lat, lng, accuracy, responderLocation: call.responderLocation || call.ambulanceLocation || null });
  await emitCallMetaUpdate(callId);
}

async function updateResponderLocation(callId, lat, lng, accuracy) {
  const call = await Call.findOneAndUpdate({ callId }, { responderLocation: { lat, lng, accuracy: accuracy || null }, ambulanceLocation: { lat, lng } }, { new: true }).lean();
  if (!call) return;
  io.emit('responder-location-update', { callId, lat, lng, accuracy, callerLocation: call.callerLocation || null });
  await emitCallMetaUpdate(callId);
}

app.get('/api/health', async (_req, res) => {
  let mlStatus = { models_loaded: false, loaded_models: [], message: 'ML service unreachable' };
  try { mlStatus = (await axios.get(`${SEVERITY_SERVICE_URL}/health`, { timeout: 2000 })).data.ml_models; } catch (_error) {}
  res.json({ status: 'ok', db: 'MongoDB Atlas', ml_models: mlStatus, timestamp: new Date().toISOString() });
});

app.get('/api/calls', async (_req, res) => {
  try { res.json(await Call.find().sort({ startTime: -1 }).limit(100).lean()); } catch (_error) { res.status(500).json({ error: 'Failed to list calls' }); }
});

app.post('/api/calls', async (req, res) => {
  try {
    const { callerLanguage = 'English', emergencyType = 'UNKNOWN', callerLocation } = req.body;
    const callId = `call-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const call = await Call.create({ callId, callerLanguage, emergencyType, firstCallerMessage: '', severityScore: 0, severityLabel: 'Assessing', callStatus: 'ASSESSING', callerEmotion: 'unknown', backgroundSound: 'unknown', aiBrainUsed: 'NARAYANI GROUNDING PIPELINE', aiBrainResponseTime: 0, transcript: [], mlVotes: { rf: 'Unknown', xgb: 'Unknown', lgbm: 'Unknown', allAgreed: false }, callerLocation: callerLocation || null, dispatchConfirmed: false, isActive: true, analysisSnapshot: createDefaultSnapshot(callId) });
    getRuntime(callId).lastSnapshot = call.analysisSnapshot;
    io.emit('call_meta_update', {
      callId: call.callId,
      severityScore: call.severityScore,
      severityLabel: call.severityLabel,
      emergencyType: call.emergencyType,
      callStatus: call.callStatus,
      firstCallerMessage: call.firstCallerMessage,
      callerEmotion: call.callerEmotion,
      callerLanguage: call.callerLanguage,
      backgroundSound: call.backgroundSound,
      dispatchConfirmed: call.dispatchConfirmed,
      callerLocation: call.callerLocation,
      analysisSnapshot: call.analysisSnapshot,
      isActive: call.isActive,
      startTime: call.startTime,
      transcript: call.transcript,
    });
    res.status(201).json(call);
  } catch (_error) { res.status(500).json({ error: 'Failed to create call' }); }
});

app.get('/api/calls/:callId', async (req, res) => {
  try {
    const call = await Call.findOne({ callId: req.params.callId }).lean();
    if (!call) return res.status(404).json({ error: 'Call not found' });
    res.json(call);
  } catch (_error) { res.status(500).json({ error: 'Failed to get call' }); }
});

app.post('/api/calls/:callId/dispatch', async (req, res) => {
  try {
    const { ambulanceUnit, paramedicName, eta, ambulanceLocation } = req.body;
    const call = await Call.findOneAndUpdate({ callId: req.params.callId }, { dispatchConfirmed: true, ambulanceUnit, paramedicName, eta, ambulanceLocation: ambulanceLocation || null }, { new: true });
    if (!call) return res.status(404).json({ error: 'Call not found' });
    io.emit('dispatch_status', { callId: call.callId, dispatchConfirmed: true, ambulanceUnit, eta, responderLocation: call.responderLocation || call.ambulanceLocation || null });
    res.json({ callId: call.callId, ambulanceUnit, paramedicName, eta, dispatchedAt: new Date().toISOString() });
  } catch (_error) { res.status(500).json({ error: 'Failed to dispatch' }); }
});

app.post('/api/calls/:callId/end', async (req, res) => {
  try {
    const call = await Call.findOneAndUpdate({ callId: req.params.callId }, { isActive: false, callStatus: 'ENDED', endTime: new Date() }, { new: true });
    if (!call) return res.status(404).json({ error: 'Call not found' });
    io.emit('call_ended', { callId: call.callId });
    await emitCallMetaUpdate(call.callId, { isActive: false, callStatus: 'ENDED' });
    res.json({ ok: true, callId: call.callId });
  } catch (_error) { res.status(500).json({ error: 'Failed to end call' }); }
});

app.delete('/api/calls/ended', async (_req, res) => {
  try { const result = await Call.deleteMany({ callStatus: { $in: ['ENDED', 'RESOLVED'] } }); res.json({ ok: true, deletedCount: result.deletedCount || 0 }); } catch (_error) { res.status(500).json({ error: 'Failed to clear ended calls' }); }
});

app.post('/api/calls/:callId/feedback', async (req, res) => {
  try {
    const { analysisRevision, assessment, notes = '' } = req.body;
    if (!assessment || !['ACCURATE', 'PARTIAL', 'INACCURATE'].includes(assessment)) return res.status(400).json({ error: 'assessment must be ACCURATE, PARTIAL, or INACCURATE' });
    const feedback = await AnalysisFeedback.findOneAndUpdate({ callId: req.params.callId, analysisRevision }, { $set: { dispatcherAssessment: assessment, dispatcherNotes: notes, updatedByDispatcherAt: new Date() } }, { new: true });
    if (!feedback) return res.status(404).json({ error: 'Analysis snapshot not found for feedback' });
    io.emit('feedback_logged', { callId: req.params.callId, analysisRevision, assessment });
    res.json({ ok: true, analysisRevision, assessment });
  } catch (_error) { res.status(500).json({ error: 'Failed to save feedback' }); }
});

app.post('/api/calls/:callId/vision', async (req, res) => {
  try {
    const call = await Call.findOne({ callId: req.params.callId }).select('analysisSnapshot').lean();
    if (!call) return res.status(404).json({ error: 'Call not found' });
    res.json(call.analysisSnapshot || createDefaultSnapshot(req.params.callId));
  } catch (_error) { res.status(500).json({ error: 'Failed to fetch current vision analysis' }); }
});

app.patch('/api/calls/:callId/ambulance', async (req, res) => {
  try {
    const { lat, lng, eta } = req.body;
    const call = await Call.findOneAndUpdate({ callId: req.params.callId }, { ambulanceLocation: { lat, lng }, responderLocation: { lat, lng }, eta }, { new: true });
    if (!call) return res.status(404).json({ error: 'Call not found' });
    io.emit('dispatch_status', { callId: call.callId, dispatchConfirmed: call.dispatchConfirmed, ambulanceUnit: call.ambulanceUnit, eta: call.eta, responderLocation: call.responderLocation || call.ambulanceLocation || null });
    res.json(call);
  } catch (_error) { res.status(500).json({ error: 'Failed to update ambulance location' }); }
});

app.get('/api/first-aid', async (req, res) => {
  const startedAt = Date.now();
  const { type, lang = 'en' } = req.query;
  if (!type) return res.status(400).json({ error: 'type is required' });
  try {
    let protocol = firstAidProtocols[type];
    if (!protocol && redisClient && redisClient.isOpen) {
      const cached = await redisClient.get('first-aid-all');
      if (cached) {
        const parsed = JSON.parse(cached);
        protocol = parsed[type];
        if (protocol) firstAidProtocols[type] = protocol;
      }
    }
    if (!protocol) {
      const dbProtocol = await Protocol.findOne({ emergencyType: type }).lean();
      if (dbProtocol) protocol = { immediate_steps: dbProtocol.immediateSteps, home_remedies: dbProtocol.homeRemedies, what_not_to_do: dbProtocol.whatNotToDo };
    }
    if (!protocol) return res.status(404).json({ error: 'Protocol not found' });
    res.json({
      emergency_type: type,
      steps: protocol.immediate_steps.map((step) => ({ step: step.step, text: step[lang] || step.en, visual: step.visual })),
      home_remedies: (protocol.home_remedies || []).filter((entry) => entry.verified === true).map((entry) => ({ remedy: entry.remedy, text: entry[lang] || entry.en, why: entry.why })),
      what_not_to_do: (protocol.what_not_to_do || []).map((entry) => ({ text: entry[lang] || entry.en, reason: entry.reason })),
      responseTimeMs: Date.now() - startedAt,
    });
  } catch (_error) { res.status(500).json({ error: 'Emergency data temporarily unavailable' }); }
});

app.get('/api/protocols', async (_req, res) => {
  try { res.json(await Protocol.find().lean()); } catch (_error) { res.status(500).json({ error: 'Failed to list protocols' }); }
});

app.post('/api/severity', async (req, res) => {
  try {
    if (!req.body.transcript) return res.status(400).json({ error: 'transcript is required' });
    res.json((await axios.post(`${SEVERITY_SERVICE_URL}/predict`, { transcript: req.body.transcript }, { timeout: 10000 })).data);
  } catch (error) {
    if (error.response?.status === 503) return res.status(503).json({ error: 'ML models still loading, please wait' });
    res.status(500).json({ error: 'Severity prediction failed' });
  }
});

app.get('/api/stats', async (_req, res) => {
  try {
    const now = new Date();
    const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const [todayCalls, activeCalls] = await Promise.all([Call.find({ startTime: { $gte: startOfDay } }).lean(), Call.countDocuments({ isActive: true })]);
    const avgSeverity = todayCalls.length ? todayCalls.reduce((sum, call) => sum + (call.severityScore || 0), 0) / todayCalls.length : 0;
    const avgResponseTime = todayCalls.length ? todayCalls.reduce((sum, call) => sum + (call.aiBrainResponseTime || 0), 0) / todayCalls.length : 0;
    res.json({ callsToday: todayCalls.length, avgSeverity: Number(avgSeverity.toFixed(1)), avgResponseTime: Number(avgResponseTime.toFixed(1)), activeCalls });
  } catch (_error) { res.status(500).json({ error: 'Failed to get stats' }); }
});

app.post('/api/voice/respond', async (req, res) => {
  try {
    if (!req.body.message) return res.status(400).json({ error: 'message is required' });
    const startedAt = Date.now();
    const aiResult = await getHumanLikeResponse(req.body.message, buildCallerContext(req.body.callId || 'voice-preview', req.body.languageCode || 'en'));
    const responseTimeMs = Date.now() - startedAt;
    if (req.body.sessionId) {
      await Conversation.findOneAndUpdate({ sessionId: req.body.sessionId }, { $push: { messages: { $each: [{ role: 'user', content: req.body.message, languageCode: aiResult['detected language'], timestamp: new Date() }, { role: 'assistant', content: aiResult.response, languageCode: aiResult['detected language'], llmProvider: 'cascade', responseTimeMs, timestamp: new Date() }] } }, $inc: { 'metadata.totalMessages': 2 }, $set: { 'metadata.primaryLanguage': aiResult['detected language'] } }, { upsert: true, new: true });
    }
    res.json({ response: aiResult.response, language_code: aiResult['detected language'], responseTimeMs });
  } catch (_error) { res.status(500).json({ error: 'Failed to generate response' }); }
});

app.get('/api/conversations', async (_req, res) => {
  try { res.json(await Conversation.find().sort({ updatedAt: -1 }).limit(50).lean()); } catch (_error) { res.status(500).json({ error: 'Failed to list conversations' }); }
});

app.get('/api/conversations/:sessionId', async (req, res) => {
  try {
    const conversation = await Conversation.findOne({ sessionId: req.params.sessionId }).lean();
    if (!conversation) return res.status(404).json({ error: 'Conversation not found' });
    res.json(conversation);
  } catch (_error) { res.status(500).json({ error: 'Failed to get conversation' }); }
});

io.on('connection', (socket) => {
  socket.on('audio_chunk', async (data) => {
    try {
      if (!data?.audioBytes || !data?.callId) return;
      const { transcript, language } = await murfSTT(data.audioBytes);
      if (!transcript) return;
      await handleCallerText({ text: transcript, callId: data.callId, sessionId: data.sessionId, languageCode: language }, socket);
    } catch (_error) { socket.emit('pipeline_error', { error: 'Failed to process audio chunk' }); }
  });

  socket.on('voice_text', async (data) => {
    try { await handleCallerText(data, socket); } catch (_error) { socket.emit('pipeline_error', { error: 'Failed to process voice text' }); }
  });

  // Video lag fix: fast preview channel plus separate slow AI-analysis channel.
  socket.on('video_preview', async (data) => {
    const { callId, frame, capturedAt, metrics } = data || {};
    if (!callId || !frame) return;
    const runtime = getRuntime(callId);
    runtime.latestPreviewFrame = frame;
    runtime.latestPreviewCapturedAt = capturedAt || new Date().toISOString();
    runtime.latestFrameMetrics = metrics || null;
    if (!runtime.firstVideoAt) runtime.firstVideoAt = Date.now();
    io.emit('video_frame_update', { callId, frame, capturedAt: runtime.latestPreviewCapturedAt, latencyMs: capturedAt ? Math.max(0, Date.now() - new Date(capturedAt).getTime()) : null });
  });

  socket.on('analysis_frame', async (data) => {
    const { callId, frame, capturedAt, metrics } = data || {};
    if (!callId || !frame) return;
    const runtime = getRuntime(callId);
    runtime.latestAnalysisFrame = frame;
    runtime.latestAnalysisCapturedAt = capturedAt || new Date().toISOString();
    runtime.latestFrameMetrics = metrics || null;
    if (!runtime.firstVideoAt) runtime.firstVideoAt = Date.now();
    processVisionFrame(callId, frame, { capturedAt, metrics }, { force: false });
  });

  socket.on('request_call_analysis', async ({ callId }) => {
    const runtime = getRuntime(callId);
    if (runtime.latestAnalysisFrame) processVisionFrame(callId, runtime.latestAnalysisFrame, { capturedAt: runtime.latestAnalysisCapturedAt, metrics: runtime.latestFrameMetrics }, { force: true });
  });

  socket.on('caller-location', async ({ callId, callerLat, callerLng, accuracy }) => {
    if (!callId || callerLat == null || callerLng == null) return;
    await updateCallerLocation(callId, callerLat, callerLng, accuracy);
  });

  socket.on('update-caller-location', async ({ callId, lat, lng, accuracy }) => {
    if (!callId || lat == null || lng == null) return;
    await updateCallerLocation(callId, lat, lng, accuracy);
  });

  socket.on('update-responder-location', async ({ callId, lat, lng, accuracy }) => {
    if (!callId || lat == null || lng == null) return;
    await updateResponderLocation(callId, lat, lng, accuracy);
  });
});

async function startServer() {
  try {
    await connectDB();
    const protocolsPath = path.join(__dirname, 'knowledge', 'first-aid-protocols.json');
    if (fs.existsSync(protocolsPath)) {
      const data = JSON.parse(fs.readFileSync(protocolsPath, 'utf8'));
      firstAidProtocols = data;
      await connectRedis();
      for (const [key, value] of Object.entries(data)) {
        await Protocol.findOneAndUpdate({ emergencyType: key }, { title: key.toUpperCase(), severityRange: value.severity_range, immediateSteps: value.immediate_steps, infantSteps: value.infant_steps || [], homeRemedies: value.home_remedies, whatNotToDo: value.what_not_to_do, paramedicAlert: value.paramedic_alert, estimatedSafeWindowMinutes: value.estimated_safe_window_minutes }, { upsert: true });
      }
      if (redisClient && redisClient.isOpen) await redisClient.set('first-aid-all', JSON.stringify(data));
    }
    // Start the server (Required for Render, Railway, Local)
    if (!process.env.VERCEL) {
      server.listen(PORT, () => console.log(`NARAYANI server listening on port ${PORT}`));
    }
  } catch (error) {
    console.error('Failed to start server:', error.message);
    process.exit(1);
  }
}

startServer();

// Export the Express app for Vercel
module.exports = app;
