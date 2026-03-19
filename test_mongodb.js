/**
 * Test script to verify MongoDB Atlas connection and basic CRUD operations.
 */
const { connectDB, disconnectDB, Call, Protocol, Conversation } = require('./db');

async function testConnection() {
  console.log('--- Testing MongoDB Atlas Connection ---\n');

  try {
    await connectDB();

    // 1. Test creating a Call document
    console.log('1. Creating a test Call...');
    const testCall = await Call.create({
      callId: `test-${Date.now()}`,
      callerLanguage: 'Hindi',
      emergencyType: 'Medical',
      severityScore: 7,
      severityLabel: 'High',
      callerEmotion: 'Anxious',
      aiBrainUsed: 'NARAYANI v2',
      aiBrainResponseTime: 0.5,
      transcript: [
        { role: 'caller', content: 'मुझे सांस लेने में तकलीफ हो रही है', timestamp: new Date().toISOString() },
        { role: 'narayani', content: 'मैं आपके साथ हूँ, बताइए क्या हुआ', timestamp: new Date().toISOString() },
      ],
      isActive: true,
    });
    console.log(`   ✅ Call created: ${testCall.callId}`);

    // 2. Test reading back
    console.log('2. Reading the call back...');
    const found = await Call.findOne({ callId: testCall.callId });
    console.log(`   ✅ Found call: ${found.callId}, Language: ${found.callerLanguage}, Severity: ${found.severityLabel}`);

    // 3. Test creating a Conversation
    console.log('3. Creating a test Conversation...');
    const testConvo = await Conversation.create({
      sessionId: `session-${Date.now()}`,
      userId: 'harsha',
      language: 'hi-IN',
      messages: [
        { role: 'user', content: 'नमस्ते, मुझे मदद चाहिए', languageCode: 'hi-IN', llmProvider: 'groq', responseTimeMs: 320 },
        { role: 'assistant', content: 'मैं यहाँ हूँ, बताओ क्या हुआ', languageCode: 'hi-IN', llmProvider: 'groq', responseTimeMs: 280 },
      ],
      metadata: { totalMessages: 2, avgResponseTimeMs: 300, primaryLanguage: 'hi-IN' },
    });
    console.log(`   ✅ Conversation created: ${testConvo.sessionId}`);

    // 4. Test creating a Protocol
    console.log('4. Creating a test Protocol...');
    const testProtocol = await Protocol.create({
      emergencyType: `Test-${Date.now()}`,
      title: 'Test Emergency Protocol',
      steps: ['Step 1: Assess the situation', 'Step 2: Call for help'],
      warnings: ['Do NOT move the patient'],
      severity: 'High',
    });
    console.log(`   ✅ Protocol created: ${testProtocol.title}`);

    // 5. Cleanup test data
    console.log('5. Cleaning up test data...');
    await Call.deleteOne({ callId: testCall.callId });
    await Conversation.deleteOne({ sessionId: testConvo.sessionId });
    await Protocol.deleteOne({ _id: testProtocol._id });
    console.log('   ✅ Test data cleaned up');

    console.log('\n--- ALL TESTS PASSED ✅ ---');
    console.log('MongoDB Atlas is fully connected and operational!');

  } catch (error) {
    console.error('\n❌ TEST FAILED:', error.message);
  } finally {
    await disconnectDB();
  }
}

testConnection();
