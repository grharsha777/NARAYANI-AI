const { mongoose } = require('../connection');

/**
 * Conversation model stores full voice conversation sessions.
 * This pairs with IndexedDB on the client for offline-first caching.
 */
const conversationSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true, index: true },
  userId: { type: String, default: 'anonymous', index: true },
  language: { type: String, default: 'en-US' },
  messages: [{
    role: { type: String, enum: ['user', 'assistant'], required: true },
    content: { type: String, required: true },
    languageCode: { type: String, default: 'en-US' },
    llmProvider: { type: String, default: null },   // Which LLM answered (groq/mistral/gemini)
    responseTimeMs: { type: Number, default: null }, // Track latency per message
    timestamp: { type: Date, default: Date.now },
  }],
  metadata: {
    totalMessages: { type: Number, default: 0 },
    avgResponseTimeMs: { type: Number, default: 0 },
    primaryLanguage: { type: String, default: 'en-US' },
  },
  isActive: { type: Boolean, default: true },
}, {
  timestamps: true
});

// Index for fast "recent conversations" queries
conversationSchema.index({ isActive: 1, updatedAt: -1 });

const Conversation = mongoose.model('Conversation', conversationSchema);

module.exports = Conversation;
