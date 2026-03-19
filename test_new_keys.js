const geminiKey1 = process.env.GEMINI_API_KEY_PRIMARY || '';
const geminiKey2 = process.env.GEMINI_API_KEY_SECONDARY || '';

async function testGemini(key, index) {
  console.log(`\n--- Testing Gemini Key ${index} ---`);
  try {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${key}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: 'Say hello in 1 word.' }] }]
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    console.log(`✅ Gemini ${index} OK:`, data.candidates[0].content.parts[0].text.trim());
    return true;
  } catch (e) {
    console.error(`❌ Gemini ${index} Error:`, e.message);
    return false;
  }
}

async function runTests() {
  const gem1 = await testGemini(geminiKey1, 1);
  const gem2 = await testGemini(geminiKey2, 2);
  
  console.log('\n--- Summary ---');
  console.log(`Gemini1: ${gem1 ? 'OK' : 'FAILED'}`);
  console.log(`Gemini2: ${gem2 ? 'OK' : 'FAILED'}`);
}

runTests();
