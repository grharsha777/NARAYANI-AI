const axios = require('axios');

// Enterprise-grade AI pipeline using local grounding + Gemini text analysis.
// No OpenAI dependency. Vision grounding uses client-side frame metrics + transcript.
// Classification uses ML severity service. Analysis uses Gemini 2.0 Flash.

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || '';
const SEVERITY_SERVICE_URL = process.env.SEVERITY_SERVICE_URL || 'http://localhost:5050';

const ANALYSIS_PROMPT = `
You are a safety-critical emergency reasoning system.

Use ONLY the grounded JSON and transcript provided by the caller.
Do NOT add visual claims that are not supported by the grounding.
If the grounding is uncertain, say "unknown" instead of guessing.
Provide cautious dispatcher guidance.

Return STRICT JSON only with this shape:
{
  "scene_summary": "",
  "what_happened": {
    "incident_type": "",
    "confidence": 0.0,
    "timeline": "",
    "mechanism": ""
  },
  "patient_status": {
    "avpu": "UNKNOWN",
    "breathing": "UNKNOWN",
    "hemorrhage": "UNKNOWN",
    "shock_signs": "UNKNOWN",
    "priority": "UNKNOWN"
  },
  "caller_instructions": ["", ""],
  "do_not_do": ["", ""],
  "dispatch_recommendation": {
    "units": ["ALS_AMBULANCE"],
    "lights_and_siren": false,
    "hospital_prealert": "",
    "additional_info_to_gather": [""]
  },
  "risk_flags": [""],
  "overall_confidence": 0.0
}

Rules:
- scene_summary must be 2 to 3 sentences of plain English.
- caller_instructions must be numbered-ready relay steps, maximum 6 items.
- patient_status.avpu must be one of: "ALERT", "VERBAL", "PAIN", "UNRESPONSIVE", "UNKNOWN"
- patient_status.breathing must be one of: "NORMAL", "LABORED", "ABSENT", "UNKNOWN"
- patient_status.hemorrhage must be one of: "VISIBLE", "NOT_VISIBLE", "UNKNOWN"
- patient_status.shock_signs must be one of: "PRESENT", "NOT_OBSERVED", "UNKNOWN"
- patient_status.priority must be one of: "P1", "P2", "P3", "UNKNOWN"
- dispatch_recommendation.units must use these tokens where relevant:
  "ALS_AMBULANCE", "BLS_AMBULANCE", "FIRE", "POLICE", "AIR_MEDEVAC"
- If uncertain, keep the language conservative and mark unknown rather than inventing.
`;

const CLASSIFICATION_PROMPT = `
You are a strict emergency triage classifier.

Use ONLY the grounded JSON and transcript.
Do NOT invent facts not present in those inputs.
Return STRICT JSON only:
{
  "severity": "CRITICAL",
  "category": "UNKNOWN",
  "confidence": 0.0,
  "triage_color": "RED"
}

Allowed severity values: "CRITICAL", "HIGH", "MODERATE", "LOW"
Allowed category values: "TRAUMA", "CARDIAC", "RESPIRATORY", "NEUROLOGICAL", "FIRE", "ACCIDENT", "MENTAL_HEALTH", "UNKNOWN"
Allowed triage_color values: "RED", "ORANGE", "YELLOW", "GREEN"

If the evidence is weak, use lower confidence instead of forcing a diagnosis.
`;

function parseJsonStrict(content) {
  if (typeof content !== 'string') {
    throw new Error('Model response was not text');
  }
  const cleaned = content
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  return JSON.parse(cleaned);
}

function clampConfidence(value, fallback = 0.35) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(0, Math.min(1, numeric));
}

function normalizeArray(value, fallback = ['unknown']) {
  if (!Array.isArray(value) || value.length === 0) return fallback;
  return value.map((entry) => String(entry || '').trim()).filter(Boolean).slice(0, 8);
}

function normalizeGrounding(raw = {}, transcript = '') {
  const environmentValues = new Set(['indoor_home', 'outdoor_road', 'vehicle', 'public_space', 'workplace', 'unknown']);
  const emotionalValues = new Set(['panicked', 'calm', 'crying', 'unresponsive', 'unknown']);
  const frameQualityValues = new Set(['clear', 'blurry', 'dark', 'obstructed']);
  const personsVisible = Number.isFinite(Number(raw.persons_visible))
    ? Math.max(0, Math.min(10, Number(raw.persons_visible)))
    : 0;

  return {
    persons_visible: personsVisible,
    subject_positions: normalizeArray(raw.subject_positions, ['unknown']),
    visible_injuries: normalizeArray(raw.visible_injuries, ['none']),
    environment: environmentValues.has(raw.environment) ? raw.environment : 'unknown',
    hazards_visible: normalizeArray(raw.hazards_visible, ['none']),
    caller_speech_transcript: String(raw.caller_speech_transcript || transcript || '').trim(),
    caller_emotional_state: emotionalValues.has(raw.caller_emotional_state) ? raw.caller_emotional_state : 'unknown',
    frame_quality: frameQualityValues.has(raw.frame_quality) ? raw.frame_quality : 'obstructed',
    literal_summary: String(raw.literal_summary || 'Visible details are limited.').trim(),
  };
}

function normalizeClassification(raw = {}) {
  const severityValues = new Set(['CRITICAL', 'HIGH', 'MODERATE', 'LOW']);
  const categoryValues = new Set(['TRAUMA', 'CARDIAC', 'RESPIRATORY', 'NEUROLOGICAL', 'FIRE', 'ACCIDENT', 'MENTAL_HEALTH', 'UNKNOWN']);
  const colorValues = new Set(['RED', 'ORANGE', 'YELLOW', 'GREEN']);

  return {
    severity: severityValues.has(raw.severity) ? raw.severity : 'LOW',
    category: categoryValues.has(raw.category) ? raw.category : 'UNKNOWN',
    confidence: clampConfidence(raw.confidence),
    triage_color: colorValues.has(raw.triage_color) ? raw.triage_color : 'GREEN',
  };
}

function normalizeAnalysis(raw = {}) {
  const patientStatus = raw.patient_status || {};
  const dispatchRecommendation = raw.dispatch_recommendation || {};
  const avpuValues = new Set(['ALERT', 'VERBAL', 'PAIN', 'UNRESPONSIVE', 'UNKNOWN']);
  const breathingValues = new Set(['NORMAL', 'LABORED', 'ABSENT', 'UNKNOWN']);
  const hemorrhageValues = new Set(['VISIBLE', 'NOT_VISIBLE', 'UNKNOWN']);
  const shockValues = new Set(['PRESENT', 'NOT_OBSERVED', 'UNKNOWN']);
  const priorityValues = new Set(['P1', 'P2', 'P3', 'UNKNOWN']);

  return {
    scene_summary: String(raw.scene_summary || 'Grounded visual details are limited; continue reassessment.').trim(),
    what_happened: {
      incident_type: String(raw.what_happened?.incident_type || 'Unclear incident').trim(),
      confidence: clampConfidence(raw.what_happened?.confidence),
      timeline: String(raw.what_happened?.timeline || 'Timeline is unclear from the current evidence.').trim(),
      mechanism: String(raw.what_happened?.mechanism || 'Mechanism not confirmed from current evidence.').trim(),
    },
    patient_status: {
      avpu: avpuValues.has(patientStatus.avpu) ? patientStatus.avpu : 'UNKNOWN',
      breathing: breathingValues.has(patientStatus.breathing) ? patientStatus.breathing : 'UNKNOWN',
      hemorrhage: hemorrhageValues.has(patientStatus.hemorrhage) ? patientStatus.hemorrhage : 'UNKNOWN',
      shock_signs: shockValues.has(patientStatus.shock_signs) ? patientStatus.shock_signs : 'UNKNOWN',
      priority: priorityValues.has(patientStatus.priority) ? patientStatus.priority : 'UNKNOWN',
    },
    caller_instructions: normalizeArray(raw.caller_instructions, [
      'Ensure the scene is safe before moving closer.',
      'Stay with the patient and monitor for any change in breathing or responsiveness.',
    ]).slice(0, 6),
    do_not_do: normalizeArray(raw.do_not_do, [
      'Do not move the patient unless there is an immediate danger.',
    ]).slice(0, 4),
    dispatch_recommendation: {
      units: normalizeArray(dispatchRecommendation.units, ['BLS_AMBULANCE']).slice(0, 5),
      lights_and_siren: Boolean(dispatchRecommendation.lights_and_siren),
      hospital_prealert: String(dispatchRecommendation.hospital_prealert || 'Monitor before pre-alert.').trim(),
      additional_info_to_gather: normalizeArray(dispatchRecommendation.additional_info_to_gather, [
        'Confirm exact location and callback number.',
      ]).slice(0, 5),
    },
    risk_flags: normalizeArray(raw.risk_flags, ['Continue monitoring for deterioration.']).slice(0, 6),
    overall_confidence: clampConfidence(raw.overall_confidence),
  };
}

// ========== LOCAL GROUNDING (no external API needed) ==========
// Uses frame metrics from client-side OpenCV + transcript keywords to build grounding.

function detectEnvironmentFromTranscript(transcript) {
  const text = (transcript || '').toLowerCase();
  if (/room|wall|indoor|home|house|kitchen|bathroom|bedroom|living|ceiling/.test(text)) return 'indoor_home';
  if (/road|street|vehicle|car|bike|highway|accident|crash|traffic/.test(text)) return 'outdoor_road';
  if (/office|work|factory|warehouse/.test(text)) return 'workplace';
  if (/mall|park|market|school|station|shop|store|public/.test(text)) return 'public_space';
  return 'unknown';
}

function detectEmotionFromTranscript(transcript) {
  const text = (transcript || '').toLowerCase();
  if (/please|help|hurry|jaldi|bachao|dying|critical|emergency|please help/.test(text)) return 'panicked';
  if (/crying|cry|tears|sobbing|rona/.test(text)) return 'crying';
  if (/unresponsive|unconscious|not responding|fainted/.test(text)) return 'unresponsive';
  return 'calm';
}

function detectInjuriesFromTranscript(transcript) {
  const text = (transcript || '').toLowerCase();
  const injuries = [];
  if (/bleeding|blood|khoon|cut|wound|gash/.test(text)) injuries.push('bleeding');
  if (/burn|jal|fire|acid/.test(text)) injuries.push('burn_marks');
  if (/fracture|broken|haddi|bone/.test(text)) injuries.push('possible_fracture');
  if (/swelling|swell|soojan/.test(text)) injuries.push('swelling');
  if (/bruise|neel/.test(text)) injuries.push('bruising');
  return injuries.length > 0 ? injuries : ['none'];
}

function detectHazardsFromTranscript(transcript) {
  const text = (transcript || '').toLowerCase();
  const hazards = [];
  if (/fire|smoke|aag|flame|burning/.test(text)) hazards.push('fire');
  if (/gas|chemical|poison|zeher/.test(text)) hazards.push('chemical');
  if (/traffic|road|car|vehicle/.test(text)) hazards.push('traffic');
  if (/water|flood|drown|paani/.test(text)) hazards.push('water');
  if (/weapon|gun|knife|attack/.test(text)) hazards.push('weapon');
  return hazards.length > 0 ? hazards : ['none'];
}

function buildLiteralSummary(transcript, metrics = {}) {
  const parts = [];
  if (metrics.brightness < 25) parts.push('The frame is very dark.');
  else if (metrics.facesDetected > 0) parts.push(`${metrics.facesDetected} person(s) visible in the camera feed.`);
  else parts.push('Camera feed is active.');

  if (transcript) {
    const short = transcript.length > 100 ? transcript.slice(0, 100) + '...' : transcript;
    parts.push(`Caller reports: "${short}"`);
  }
  return parts.join(' ');
}

async function groundingPass({ frameDataUrl, transcript = '', frameMeta = {} }) {
  const metrics = frameMeta || {};
  const facesDetected = Number(metrics.facesDetected || 0);
  const brightness = Number(metrics.brightness || 128);

  let frameQuality = 'clear';
  if (brightness < 25) frameQuality = 'dark';
  else if (Number(metrics.variance || 50) < 18) frameQuality = 'blurry';
  else if (!frameDataUrl) frameQuality = 'obstructed';

  const grounding = {
    persons_visible: facesDetected,
    subject_positions: facesDetected > 0 ? ['face_visible'] : ['unknown'],
    visible_injuries: detectInjuriesFromTranscript(transcript),
    environment: detectEnvironmentFromTranscript(transcript),
    hazards_visible: detectHazardsFromTranscript(transcript),
    caller_speech_transcript: transcript,
    caller_emotional_state: detectEmotionFromTranscript(transcript),
    frame_quality: frameQuality,
    literal_summary: buildLiteralSummary(transcript, { ...metrics, facesDetected }),
  };

  return normalizeGrounding(grounding, transcript);
}

// ========== CLASSIFICATION (ML service + keyword heuristics) ==========

function keywordClassification(transcript) {
  const text = (transcript || '').toLowerCase();

  if (/not breathing|no pulse|cardiac arrest|cpr|heart stopped|died|dying|unconscious.*not breathing/.test(text)) {
    return { severity: 'CRITICAL', category: 'CARDIAC', confidence: 0.75, triage_color: 'RED' };
  }
  if (/chest pain|heart attack|angina|heart|dil ka dard/.test(text)) {
    return { severity: 'CRITICAL', category: 'CARDIAC', confidence: 0.7, triage_color: 'RED' };
  }
  if (/fire|smoke|burn|aag|jal raha/.test(text)) {
    return { severity: 'HIGH', category: 'FIRE', confidence: 0.65, triage_color: 'ORANGE' };
  }
  if (/accident|crash|collision|hit by|takkar/.test(text)) {
    return { severity: 'HIGH', category: 'ACCIDENT', confidence: 0.6, triage_color: 'ORANGE' };
  }
  if (/heavy bleeding|blood.*everywhere|khoon bahar|not stop.*bleed/.test(text)) {
    return { severity: 'HIGH', category: 'TRAUMA', confidence: 0.65, triage_color: 'ORANGE' };
  }
  if (/seizure|convulsion|fitting|shaking|daure/.test(text)) {
    return { severity: 'HIGH', category: 'NEUROLOGICAL', confidence: 0.6, triage_color: 'ORANGE' };
  }
  if (/stroke|face drooping|arm weak|speech slurred/.test(text)) {
    return { severity: 'CRITICAL', category: 'NEUROLOGICAL', confidence: 0.7, triage_color: 'RED' };
  }
  if (/cant breathe|breathing difficulty|saans|asthma|choking/.test(text)) {
    return { severity: 'HIGH', category: 'RESPIRATORY', confidence: 0.6, triage_color: 'ORANGE' };
  }
  // Minor cuts, scratches — these should be LOW, not moderate
  if (/scratch|small cut|minor cut|chhota cut|light cut|paper cut|nick|graze|chil gaya|halka sa/.test(text)) {
    return { severity: 'LOW', category: 'TRAUMA', confidence: 0.35, triage_color: 'GREEN' };
  }
  if (/bleeding|cut|wound|fracture|fell|broken|injury|chot/.test(text)) {
    // Check if context suggests it's minor
    if (/finger|ungli|small|little|thoda|chhota|halka|slight/.test(text)) {
      return { severity: 'LOW', category: 'TRAUMA', confidence: 0.4, triage_color: 'GREEN' };
    }
    return { severity: 'MODERATE', category: 'TRAUMA', confidence: 0.55, triage_color: 'YELLOW' };
  }
  if (/head.*ach|headache|sar.*dard|migraine|sir.*dard|dizzy|chakkar/.test(text)) {
    return { severity: 'MODERATE', category: 'NEUROLOGICAL', confidence: 0.5, triage_color: 'YELLOW' };
  }
  if (/pain|dard|hurt|ache|fever|bukhar|vomit|ulti/.test(text)) {
    return { severity: 'MODERATE', category: 'UNKNOWN', confidence: 0.45, triage_color: 'YELLOW' };
  }
  if (/suicide|kill myself|want to die|desperate|hopeless|depression/.test(text)) {
    return { severity: 'HIGH', category: 'MENTAL_HEALTH', confidence: 0.65, triage_color: 'ORANGE' };
  }

  return { severity: 'LOW', category: 'UNKNOWN', confidence: 0.3, triage_color: 'GREEN' };
}

function mlSeverityToClassification(mlResult) {
  if (!mlResult || mlResult.risk === 'Unknown') return null;
  const score = Number(mlResult.score || 0);
  if (score >= 8) return { severity: 'CRITICAL', confidence: 0.8, triage_color: 'RED' };
  if (score >= 6) return { severity: 'HIGH', confidence: 0.7, triage_color: 'ORANGE' };
  if (score >= 4) return { severity: 'MODERATE', confidence: 0.55, triage_color: 'YELLOW' };
  if (score > 0) return { severity: 'LOW', confidence: 0.4, triage_color: 'GREEN' };
  return null;
}

async function classificationPass({ grounding, mlSeverity = null }) {
  const transcript = grounding?.caller_speech_transcript || '';

  // Combine keyword-based and ML-based classification
  const keywordResult = keywordClassification(transcript);
  const mlResult = mlSeverityToClassification(mlSeverity);

  if (mlResult) {
    // Take the higher severity between keyword and ML
    const severityRank = { CRITICAL: 4, HIGH: 3, MODERATE: 2, LOW: 1 };
    const keywordRank = severityRank[keywordResult.severity] || 0;
    const mlRank = severityRank[mlResult.severity] || 0;

    if (mlRank >= keywordRank) {
      return normalizeClassification({
        ...keywordResult,
        severity: mlResult.severity,
        confidence: Math.max(keywordResult.confidence, mlResult.confidence),
        triage_color: mlResult.triage_color,
      });
    }
  }

  return normalizeClassification(keywordResult);
}

// ========== ANALYSIS (Groq primary, Mistral fallback for text reasoning) ==========

async function callGroqJson({ prompt, input, debugLabel = '' }) {
  if (!GROQ_API_KEY) throw new Error('GROQ_API_KEY not configured');

  const response = await axios.post(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      model: 'llama-3.1-8b-instant',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content: JSON.stringify(input) },
      ],
      temperature: 0.3,
    },
    {
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    }
  );

  const content = response.data?.choices?.[0]?.message?.content;
  if (debugLabel) console.log(`[AI Pipeline] ${debugLabel} raw: ${content}`);
  const parsed = parseJsonStrict(content);
  if (debugLabel) console.log(`[AI Pipeline] ${debugLabel} parsed: ${JSON.stringify(parsed)}`);
  return parsed;
}

async function callMistralJson({ prompt, input, debugLabel = '' }) {
  if (!MISTRAL_API_KEY) throw new Error('MISTRAL_API_KEY not configured');

  const response = await axios.post(
    'https://api.mistral.ai/v1/chat/completions',
    {
      model: 'mistral-small-latest',
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: prompt },
        { role: 'user', content: JSON.stringify(input) },
      ],
      temperature: 0.3,
    },
    {
      headers: {
        Authorization: `Bearer ${MISTRAL_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    }
  );

  const content = response.data?.choices?.[0]?.message?.content;
  if (debugLabel) console.log(`[AI Pipeline] ${debugLabel} raw: ${content}`);
  const parsed = parseJsonStrict(content);
  if (debugLabel) console.log(`[AI Pipeline] ${debugLabel} parsed: ${JSON.stringify(parsed)}`);
  return parsed;
}

async function callLLMJsonCascade({ prompt, input, debugLabel = '' }) {
  // Try Groq first, then Mistral
  try {
    return await callGroqJson({ prompt, input, debugLabel });
  } catch (groqError) {
    console.warn(`[AI Pipeline] Groq failed: ${groqError.message}`);
  }
  try {
    return await callMistralJson({ prompt, input, debugLabel: `${debugLabel}-mistral` });
  } catch (mistralError) {
    console.warn(`[AI Pipeline] Mistral failed: ${mistralError.message}`);
  }
  throw new Error('All LLM providers failed');
}

function fallbackAnalysis(grounding, classification) {
  const transcript = grounding?.caller_speech_transcript || '';

  return normalizeAnalysis({
    scene_summary: grounding?.literal_summary || 'Visible details are limited; continue reassessment.',
    what_happened: {
      incident_type: classification.category === 'UNKNOWN' ? 'Unclear incident' : classification.category,
      confidence: classification.confidence,
      timeline: transcript ? 'Caller transcript suggests an active emergency, but the sequence is not fully confirmed.' : 'Timeline is unclear from current evidence.',
      mechanism: grounding?.hazards_visible?.includes('fire') ? 'Possible fire-related mechanism.' : 'Mechanism not confirmed from current evidence.',
    },
    patient_status: {
      avpu: 'UNKNOWN',
      breathing: /not breathing|cannot breathe|blue/.test(transcript.toLowerCase()) ? 'ABSENT' : 'UNKNOWN',
      hemorrhage: grounding?.visible_injuries?.some((entry) => /bleed/i.test(entry)) ? 'VISIBLE' : 'UNKNOWN',
      shock_signs: 'UNKNOWN',
      priority: classification.severity === 'CRITICAL' ? 'P1' : classification.severity === 'HIGH' ? 'P2' : 'P3',
    },
    caller_instructions: [
      'Keep the scene safe and stay with the patient.',
      'Follow dispatcher instructions and watch for any change in breathing or responsiveness.',
    ],
    do_not_do: ['Do not move the patient unless there is an immediate danger.'],
    dispatch_recommendation: {
      units: classification.severity === 'CRITICAL' ? ['ALS_AMBULANCE'] : ['BLS_AMBULANCE'],
      lights_and_siren: classification.severity === 'CRITICAL',
      hospital_prealert: classification.severity === 'CRITICAL' ? 'Consider pre-alert based on dispatcher confirmation.' : 'No pre-alert until more evidence is available.',
      additional_info_to_gather: ['Confirm exact location, patient age, and current breathing status.'],
    },
    risk_flags: ['Continue manual reassessment.'],
    overall_confidence: Math.min(classification.confidence, 0.5),
  });
}

function findContradictions(grounding, analysis) {
  const supportedTranscript = (grounding?.caller_speech_transcript || '').toLowerCase();
  const combinedText = [
    analysis?.scene_summary,
    analysis?.what_happened?.incident_type,
    analysis?.what_happened?.timeline,
    analysis?.what_happened?.mechanism,
  ].filter(Boolean).join(' ').toLowerCase();

  const contradictions = [];
  const hazards = grounding?.hazards_visible || [];

  if (grounding?.persons_visible <= 1 && /\b(multiple people|several people|crowd|two people|three people)\b/.test(combinedText)) {
    contradictions.push('Analysis mentions multiple people but grounding does not support that.');
  }

  if (!hazards.includes('fire') && !/fire|smoke|burn/.test(supportedTranscript)) {
    if (/\bfire\b|\bsmoke\b|\bburning\b/.test(combinedText)) {
      contradictions.push('Analysis mentions fire or smoke without grounded support.');
    }
  }

  return contradictions;
}

async function analysisPass({ grounding, classification }) {
  // Try Groq/Mistral cascade, fall back to keyword-based analysis
  if (!GROQ_API_KEY && !MISTRAL_API_KEY) {
    return {
      analysis: fallbackAnalysis(grounding, classification || normalizeClassification()),
      contradictions: ['No LLM API keys configured; using safe fallback analysis.'],
      rerunCount: 0,
    };
  }

  try {
    const rawAnalysis = await callLLMJsonCascade({
      prompt: ANALYSIS_PROMPT,
      input: { grounding, classification },
      debugLabel: 'analysis',
    });

    const normalized = normalizeAnalysis(rawAnalysis);
    const contradictions = findContradictions(grounding, normalized);

    if (contradictions.length > 0) {
      return {
        analysis: fallbackAnalysis(grounding, classification),
        contradictions,
        rerunCount: 1,
      };
    }

    return { analysis: normalized, contradictions: [], rerunCount: 0 };
  } catch (error) {
    console.warn(`[AI Pipeline] LLM analysis failed: ${error.message}`);
    return {
      analysis: fallbackAnalysis(grounding, classification || normalizeClassification()),
      contradictions: [`Analysis API failed: ${error.message}`],
      rerunCount: 0,
    };
  }
}

module.exports = {
  ANALYSIS_PROMPT,
  CLASSIFICATION_PROMPT,
  groundingPass,
  analysisPass,
  classificationPass,
  normalizeGrounding,
  normalizeClassification,
  normalizeAnalysis,
  findContradictions,
};
