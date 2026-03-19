const axios = require('axios');

const testCases = [
  {
    name: "Anaphylaxis (Critical)",
    transcript: "HELP! My son stopped breathing after eating a peanut. He is turning blue!"
  },
  {
    name: "Major Trauma/Fire (Critical)",
    transcript: "There is a massive crash on the highway. A car is on fire and someone is trapped inside bleeding heavily."
  },
  {
    name: "Cardiac/Stroke (Critical)",
    transcript: "My wife is having chest pain and her left arm is numb. She can't speak properly."
  },
  {
    name: "Compound Fracture (High)",
    transcript: "I fell off a ladder and my leg is definitely broken. I can see the bone. Please hurry."
  },
  {
    name: "Severe Illness (Medium)",
    transcript: "I have a high fever for three days and a very bad cough. I am alone and feeling weak."
  },
  {
    name: "Minor Cut (Low)",
    transcript: "I cut my finger while chopping vegetables. It's bleeding a little but I've washed it. Just wondering what to do next."
  },
  {
    name: "Non-Medical Stress (Low/Edge)",
    transcript: "Hey, I lost my keys and I'm really stressed out. Can you tell me what to do?"
  },
  {
    name: "Quiet/Silent Emergency (Critical)",
    transcript: "I... can't... breathe... please... help..."
  },
  {
    name: "Child/Infant Emergency (Critical)",
    transcript: "My baby is not moving and his skin is cold. He was sleeping but won't wake up."
  },
  {
    name: "Multiple Injuries/Serious (Critical)",
    transcript: "A gas cylinder exploded. Multiple people are burned and one person is not moving."
  }
];

async function runTests() {
  console.log("🚀 Starting Advanced ML Model Evaluation (v3 Engine)\n");
  console.log("=".repeat(80));
  
  const results = [];
  
  for (const test of testCases) {
    try {
      const start = Date.now();
      const response = await axios.post('http://localhost:3000/api/severity', {
        transcript: test.transcript
      });
      const end = Date.now();
      
      results.push({
        name: test.name,
        transcript: test.transcript,
        prediction: response.data,
        latency: end - start
      });
      
      console.log(`PASS: ${test.name}`);
    } catch (err) {
      console.error(`FAIL: ${test.name} - ${err.message}`);
    }
  }

  console.log("\n" + "=".repeat(80));
  console.log("\n📊 FINAL EVALUATION REPORT (v3)\n");
  
  results.forEach(res => {
    const p = res.prediction;
    console.log(`[${res.name.toUpperCase()}]`);
    console.log(`Transcript: "${res.transcript}"`);
    console.log(`Final Decision: ${p.ensemble_decision.toUpperCase()} (Score: ${p.severity})`);
    console.log(`Confidence:     ${p.confidence}`);
    console.log(`Agreement:      ${p.all_agreed ? "✅ ALL MODELS AGREED" : "⚠️ DISAGREEMENT"}`);
    console.log(`Votes:          RF: ${p.random_forest}, XGB: ${p.xgboost}, LGBM: ${p.lightgbm}`);
    console.log(`Latency:        ${res.latency}ms`);
    console.log("-".repeat(40));
  });
}

runTests();
