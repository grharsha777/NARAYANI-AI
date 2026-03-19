const groqKey = process.env.GROQ_API_KEY || '';
const geminiKey = process.env.GEMINI_API_KEY || '';

async function fetchGroqModels() {
  console.log('--- Fetching Groq Models ---');
  try {
    const res = await fetch('https://api.groq.com/openai/v1/models', {
      headers: { 'Authorization': `Bearer ${groqKey}` }
    });
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    const modelIds = data.data.map(m => m.id);
    console.log('Groq Models:', modelIds.slice(0, 5).join(', '), '...');
    return modelIds[0]; // return first model to test
  } catch (e) {
    console.error('Groq Error:', e.message);
  }
}

async function fetchGeminiModels() {
  console.log('\n--- Fetching Gemini Models ---');
  try {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${geminiKey}`);
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    const modelIds = data.models.map(m => m.name);
    console.log('Gemini Models:', modelIds.filter(m => m.includes('flash') || m.includes('pro')).slice(0, 5).join(', '));
  } catch (e) {
    console.error('Gemini Error:', e.message);
  }
}

async function run() {
  await fetchGroqModels();
  await fetchGeminiModels();
}

run();
