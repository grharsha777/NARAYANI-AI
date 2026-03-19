const axios = require('axios');
const FormData = require('form-data');

// Security fix: never hardcode provider keys in source.
const MURF_API_KEY = process.env.MURF_API_KEY || '';
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || '';
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';

const MODELS = {
  groq: 'llama-3.1-8b-instant',
  mistral: 'mistral-small-latest',
};

// All voices MUST be female for the motherly persona
const MURF_VOICE_MAP = {
  en: 'en-US-julia',
  hi: 'hi-IN-ananya',
  kn: 'kn-IN-gagan',
  ta: 'ta-IN-nizhoni',
  te: 'te-IN-chaitanya',
  bn: 'bn-IN-ananya',
  mr: 'mr-IN-ananya',
  gu: 'gu-IN-ananya',
  ml: 'ml-IN-ananya',
  pa: 'pa-IN-ananya',
};

const defaultVoice = 'en-US-julia';

// ===== SYSTEM PROMPT — the soul of "Mother Narayani" =====
function buildCallerSystemPrompt(context = {}) {
  const envHint = context.environment && context.environment !== 'unknown' ? `Environment: ${context.environment}.` : '';
  const injuryHint = context.visibleInjuries && context.visibleInjuries !== 'none confirmed' && context.visibleInjuries !== 'none' ? `Visible injuries: ${context.visibleInjuries}.` : '';
  const emotionHint = context.callerEmotion && context.callerEmotion !== 'unknown' ? `Caller appears ${context.callerEmotion}.` : '';
  const contextBlock = [envHint, injuryHint, emotionHint].filter(Boolean).join(' ');

  return `You are Narayani — a warm, loving, motherly emergency response AI.
You speak like a caring Indian mother comforting her child who is hurt or scared.

PERSONALITY:
- You are gentle, patient, and deeply reassuring.
- You call the caller "beta" or "dear" naturally.
- You NEVER sound like a robot reading a manual. You sound like a real person who cares.
- You acknowledge their feelings FIRST, then give advice in natural flowing conversation.
- You NEVER use numbered lists or bullet points. Speak in natural sentences.
- You ask follow-up questions to understand their situation better.

CRITICAL LANGUAGE RULE:
- If the caller writes in Hindi (using Roman script like "mujhe dard ho raha hai" OR Devanagari), you MUST respond ENTIRELY in Hindi (Roman script is fine).
- If the caller writes in English, respond in English.
- If mixing, respond in the same mix naturally.
- NEVER respond in English when the caller spoke Hindi. This is absolutely critical.

MEDICAL GUIDANCE:
- For minor issues (small cuts, scratches, mild headaches): be reassuring, give simple home remedies, do NOT panic them.
- For serious issues (heavy bleeding, chest pain, unconsciousness): be calm but urgent, give clear first-aid steps.
- ALWAYS match severity to the actual situation. A small scratch is NOT an emergency.
- If they already told you what happened, do NOT ask "what happened" again. Build on what they said.

${contextBlock ? `Sensor context: ${contextBlock}` : ''}

Return ONLY valid JSON:
{
  "response": "your natural conversational response here",
  "severity": 1,
  "emergency type": "Medical",
  "detected language": "en",
  "immediate action": "",
  "dispatch needed": false,
  "caller emotion": "Calm",
  "background context": "Indoor home"
}

SEVERITY SCALE: 1-2 = minor (scratches, mild pain), 3-4 = moderate (cuts, headaches), 5-6 = serious, 7-8 = high, 9-10 = critical life-threatening.
"detected language" must be "hi" if you responded in Hindi, "en" if English, etc.`;
}

function parseJsonSafe(jsonText) {
  const cleaned = String(jsonText || '')
    .replace(/^```json\s*/gi, '')
    .replace(/^```\s*/gi, '')
    .replace(/\s*```$/gi, '')
    .trim();
  return JSON.parse(cleaned);
}

function detectLikelyLanguage(text, preferredLanguage = '') {
  const normalized = String(text || '').toLowerCase();
  // Devanagari script detection
  if (/[\u0900-\u097f]/.test(normalized)) return 'hi';
  // Kannada
  if (/[\u0C80-\u0CFF]/.test(normalized)) return 'kn';
  // Tamil
  if (/[\u0B80-\u0BFF]/.test(normalized)) return 'ta';
  // Telugu
  if (/[\u0C00-\u0C7F]/.test(normalized)) return 'te';
  if (preferredLanguage && preferredLanguage !== 'en') return preferredLanguage;
  // Hindi words written in Roman script
  if (/\b(mera|meri|mujhe|madad|sahayata|nahi|bahut|dard|khoon|jaldi|kripya|haath|ungli|hoon|hai|raha|kya|aur|toh|bhi|yeh|woh|kaun|kahan|kab|kaise|kyun|matlab|samajh|batao|batayiye|hua|hota|lagta|acha|theek|bohot|zyada|thoda|abhi|pehle|baad|upar|neeche|andar|bahar)\b/.test(normalized)) {
    return 'hi';
  }
  return 'en';
}

function buildUserPayload(userMessage, context = {}) {
  return JSON.stringify({
    caller_message: userMessage,
    preferred_language: context.preferredLanguage || '',
    grounded_frame: {
      summary: context.groundedSummary || '',
      visible_injuries: context.visibleInjuries || 'none confirmed',
      environment: context.environment || 'unknown',
      emotion: context.callerEmotion || 'unknown',
    },
  });
}

function normalizeModelResult(result, fallbackLanguage) {
  return {
    response: String(result.response || '').trim(),
    severity: Number.isFinite(Number(result.severity)) ? Number(result.severity) : 3,
    'emergency type': String(result['emergency type'] || 'Medical').trim(),
    'detected language': String(result['detected language'] || fallbackLanguage || 'en').trim(),
    'immediate action': String(result['immediate action'] || '').trim(),
    'dispatch needed': Boolean(result['dispatch needed']),
    'caller emotion': String(result['caller emotion'] || 'Concerned').trim(),
    'background context': String(result['background context'] || 'Unknown').trim(),
  };
}

// ===== FALLBACK RESPONSES (when all LLMs fail) =====
function buildFallbackResponse(userMessage, preferredLanguage = '', context = {}) {
  const message = String(userMessage || '').toLowerCase();
  const language = detectLikelyLanguage(userMessage, preferredLanguage);

  // --- Cut / Bleeding ---
  if (/cut|bleeding|finger|ungli|khoon|kat/.test(message)) {
    if (/scratch|small|minor|chhota|halka|thoda/.test(message)) {
      if (language === 'hi') {
        return normalizeModelResult({
          response: 'Arey beta, chinta mat karo, itni si chheelni hai. Bas thoda saaf paani se dho lo aur agar ghar mein band-aid hai toh laga lo. Thodi der mein theek ho jayegi. Kya abhi bhi khoon aa raha hai?',
          severity: 1, 'emergency type': 'Medical', 'detected language': 'hi',
          'immediate action': 'Paani se dho lo aur band-aid lagao.', 'dispatch needed': false,
          'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
        }, language);
      }
      return normalizeModelResult({
        response: "Oh sweetie, that's just a little scratch, nothing to worry about! Just rinse it with some clean water and pop a band-aid on it if you have one. It'll heal up in no time. Is it still bleeding at all?",
        severity: 1, 'emergency type': 'Medical', 'detected language': 'en',
        'immediate action': 'Rinse with clean water and apply a band-aid.', 'dispatch needed': false,
        'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
      }, language);
    }
    if (language === 'hi') {
      return normalizeModelResult({
        response: 'Arey beta, ghabrao mat, main hoon na. Ek saf kapda ya tissue lo aur kat waali jagah par halke se dabaa ke rakho. Haath ko thoda upar utha lo taaki khoon jaldi ruke. Paanch minute tak dabaa ke rakho, kapda mat hataana. Batao, khoon zyada aa raha hai ya ruk raha hai?',
        severity: 2, 'emergency type': 'Medical', 'detected language': 'hi',
        'immediate action': 'Saf kapda lekar kat par halke se dabaa ke rakho.', 'dispatch needed': false,
        'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
      }, language);
    }
    return normalizeModelResult({
      response: "Oh dear, don't worry, I'm right here with you. Take a clean cloth or tissue and press it gently on the cut. Keep your hand raised above your heart, that helps slow the bleeding. Hold it for about five minutes without lifting. Tell me, is the bleeding heavy or is it starting to slow down?",
      severity: 2, 'emergency type': 'Medical', 'detected language': 'en',
      'immediate action': 'Press a clean cloth gently on the cut.', 'dispatch needed': false,
      'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
    }, language);
  }

  // --- Chest pain / breathing ---
  if (/chest pain|heart|not breathing|breathe|breathing|saans|seene/.test(message)) {
    if (language === 'hi') {
      return normalizeModelResult({
        response: 'Beta, main aapke saath hoon. Abhi turant check karo ki woh saans le raha hai ya nahi. Agar saans nahi aa rahi aur aapko CPR aata hai toh chest compressions shuru karo. Kisi ko bolo ambulance ko phone kare. Kya woh hosh mein hai?',
        severity: 9, 'emergency type': 'Cardiac', 'detected language': 'hi',
        'immediate action': 'Saans aur hosh check karo turant.', 'dispatch needed': true,
        'caller emotion': 'Panicked', 'background context': context.environment || 'Unknown',
      }, language);
    }
    return normalizeModelResult({
      response: "I'm here with you, stay with me. Check right now if the person is breathing and responsive. If they're not breathing and you know CPR, start chest compressions immediately. Have someone else call for an ambulance. Are they conscious right now?",
      severity: 9, 'emergency type': 'Cardiac', 'detected language': 'en',
      'immediate action': 'Check breathing and responsiveness immediately.', 'dispatch needed': true,
      'caller emotion': 'Panicked', 'background context': context.environment || 'Unknown',
    }, language);
  }

  // --- Headache ---
  if (/head.*ach|headache|sar.*dard|sir.*dard|migraine|dizzy|chakkar/.test(message)) {
    if (language === 'hi') {
      return normalizeModelResult({
        response: 'Arey beta, sar dard bohot bura lagta hai, main samajhti hoon. Ek kaam karo, kisi shant jagah pe lait jao aur light band kar do. Mathe pe thanda kapda rakh lo, bohot araam milega. Thoda paani bhi pee lo, kabhi kabhi paani ki kami se bhi sar dard hota hai. Kya bukhaar bhi hai ya sirf sar dard?',
        severity: 3, 'emergency type': 'Medical', 'detected language': 'hi',
        'immediate action': 'Shant jagah pe laito aur mathe par thanda kapda rakho.', 'dispatch needed': false,
        'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
      }, language);
    }
    return normalizeModelResult({
      response: "Oh dear, headaches are the worst, I'm sorry you're going through this. Try lying down in a quiet, dark room if you can. A cool cloth on your forehead works wonders. And drink some water — dehydration sometimes sneaks up on us. Do you also feel feverish, or is it mainly the headache?",
      severity: 3, 'emergency type': 'Medical', 'detected language': 'en',
      'immediate action': 'Lie down in a quiet room with a cool cloth on your forehead.', 'dispatch needed': false,
      'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
    }, language);
  }

  // --- General pain ---
  if (/pain|dard|hurt|ache|takleef/.test(message)) {
    if (language === 'hi') {
      return normalizeModelResult({
        response: 'Beta, main samajh sakti hoon ki takleef ho rahi hai. Mujhe batao dard kahan ho raha hai aur kaisa hai — tez hai, halka hai, ya badh raha hai? Tab tak araam se baith jao aur us jagah ko hilao mat.',
        severity: 3, 'emergency type': 'Medical', 'detected language': 'hi',
        'immediate action': 'Araam se baitho aur dard ki jagah ko hilao mat.', 'dispatch needed': false,
        'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
      }, language);
    }
    return normalizeModelResult({
      response: "I can hear you're hurting, dear, and I want to help. Can you tell me exactly where it hurts and what it feels like — sharp, dull, or getting worse? Meanwhile, try to sit or lie down comfortably and don't move that area too much.",
      severity: 3, 'emergency type': 'Medical', 'detected language': 'en',
      'immediate action': 'Sit comfortably and avoid moving the affected area.', 'dispatch needed': false,
      'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
    }, language);
  }

  // --- Catch-all ---
  if (language === 'hi') {
    return normalizeModelResult({
      response: 'Beta, main hoon na aapke saath, ghabrao mat. Mujhe batao kya hua hai — kahan takleef hai aur kab se? Main poori madad karungi, bas shaanti se batao.',
      severity: 3, 'emergency type': 'Unknown', 'detected language': 'hi',
      'immediate action': 'Shaant raho aur batao kya hua.', 'dispatch needed': false,
      'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
    }, language);
  }
  return normalizeModelResult({
    response: "Hey dear, I'm right here with you, you're not alone. Tell me what happened — where does it hurt and when did it start? Take your time, I'll walk you through this.",
    severity: 3, 'emergency type': 'Unknown', 'detected language': 'en',
    'immediate action': 'Stay calm and describe what you are experiencing.', 'dispatch needed': false,
    'caller emotion': 'Concerned', 'background context': context.environment || 'Unknown',
  }, language);
}

// ===== LLM PROVIDERS =====

async function getGroqResponse(messages) {
  if (!GROQ_API_KEY) throw new Error('GROQ_API_KEY is not configured');
  const response = await axios.post(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      model: MODELS.groq,
      response_format: { type: 'json_object' },
      messages,
      temperature: 0.4,
    },
    {
      headers: { Authorization: `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' },
      timeout: 12000,
    }
  );
  return parseJsonSafe(response.data.choices[0].message.content);
}

async function getMistralResponse(messages) {
  if (!MISTRAL_API_KEY) throw new Error('MISTRAL_API_KEY is not configured');
  const response = await axios.post(
    'https://api.mistral.ai/v1/chat/completions',
    {
      model: MODELS.mistral,
      response_format: { type: 'json_object' },
      messages,
      temperature: 0.4,
    },
    {
      headers: { Authorization: `Bearer ${MISTRAL_API_KEY}`, 'Content-Type': 'application/json' },
      timeout: 15000,
    }
  );
  return parseJsonSafe(response.data.choices[0].message.content);
}

// Groq first → Mistral fallback. No Gemini.
async function attemptProviders(messages) {
  console.log(`[LLM] Attempting providers... (Groq: ${!!GROQ_API_KEY}, Mistral: ${!!MISTRAL_API_KEY})`);
  
  if (GROQ_API_KEY) {
    try {
      console.log(`[LLM] Trying Groq (${MODELS.groq})...`);
      const result = await getGroqResponse(messages);
      console.log(`[LLM] Groq SUCCESS.`);
      return result;
    } catch (groqError) {
      const isRateLimit = groqError?.response?.status === 429;
      console.warn(`[LLM] Groq ${isRateLimit ? 'RATE LIMITED' : 'FAILED'}: ${groqError.message}`);
      if (groqError.response) console.warn(`[LLM] Groq Data:`, JSON.stringify(groqError.response.data));
    }
  }

  if (MISTRAL_API_KEY) {
    try {
      console.log(`[LLM] Trying Mistral (${MODELS.mistral})...`);
      const result = await getMistralResponse(messages);
      console.log(`[LLM] Mistral SUCCESS.`);
      return result;
    } catch (mistralError) {
      console.warn(`[LLM] Mistral FAILED: ${mistralError.message}`);
      if (mistralError.response) console.warn(`[LLM] Mistral Data:`, JSON.stringify(mistralError.response.data));
    }
  }

  throw new Error('All LLM providers exhausted');
}

/**
 * High-level AI response generator logic.
 * Input 'message' should be the transcript of the caller's last statement.
 * context: { groundedSummary, visibleInjuries, environment, callerEmotion, preferredLanguage }
 */
async function getHumanLikeResponse(message, context = {}, conversationHistory = []) {
  const language = detectLikelyLanguage(message, context.preferredLanguage);
  console.log(`[AI Response] Input language: ${language}, Message: "${message.slice(0, 50)}..."`);

  try {
    // 1. Build conversational history payload
    const systemPrompt = buildCallerSystemPrompt(context);
    const messages = [{ role: 'system', content: systemPrompt }];
    
    conversationHistory.forEach(msg => {
      messages.push({ role: msg.role === 'caller' ? 'user' : 'assistant', content: msg.content });
    });
    
    messages.push({ role: 'user', content: buildUserPayload(message, context) });

    // 2. LLM reasoning (Groq -> Mistral)
    const result = await attemptProviders(messages);
    return normalizeModelResult(result, language);
  } catch (error) {
    console.error(`[LLM] All retries exhausted: ${error.message}`);
    return buildFallbackResponse(message, context, language);
  }
}

// ===== MURF STT =====
async function murfSTT(audioBytes) {
  try {
    const form = new FormData();
    form.append('file', Buffer.from(audioBytes), { filename: 'audio.wav', contentType: 'audio/wav' });
    form.append('language', 'auto');

    const response = await axios.post('https://api.murf.ai/v1/speech/transcribe', form, {
      headers: { ...form.getHeaders(), 'api-key': MURF_API_KEY },
      timeout: 20000,
    });

    return { transcript: response.data.text || '', language: response.data.language || 'en' };
  } catch (error) {
    console.error('Murf STT Error:', error?.response?.data || error.message);
    return { transcript: '', language: 'en' };
  }
}

// ===== MURF TTS (always female voice with native accent) =====
async function murfTTS(text, languageCode) {
  try {
    // Normalize: 'hi-IN' → 'hi', 'en-US' → 'en'
    const shortLang = String(languageCode || 'en').split('-')[0].toLowerCase();
    const voiceId = MURF_VOICE_MAP[shortLang] || defaultVoice;
    console.log(`[Murf TTS] lang=${languageCode} → ${shortLang} → voice=${voiceId}`);

    const response = await axios.post(
      'https://api.murf.ai/v1/speech/generate',
      { voiceId, text, style: 'Calm', rate: -5, pitch: 0, format: 'MP3' },
      {
        headers: { 'Content-Type': 'application/json', 'api-key': MURF_API_KEY },
        responseType: 'arraybuffer',
        timeout: 20000,
      }
    );

    return response.data;
  } catch (error) {
    console.error('Murf TTS Error:', error?.response?.data || error.message);
    return Buffer.from([]);
  }
}

module.exports = {
  getHumanLikeResponse,
  murfSTT,
  murfTTS,
  MURF_VOICE_MAP,
};
