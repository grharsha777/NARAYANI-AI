const { mongoose } = require('../connection');

const transcriptEntrySchema = new mongoose.Schema({
  role: { type: String, enum: ['caller', 'assistant', 'dispatcher', 'user', 'narayani'], required: true },
  content: { type: String, required: true },
  timestamp: { type: Date, default: Date.now, required: true },
}, { _id: false });

const callSchema = new mongoose.Schema({
  callId: { type: String, required: true, unique: true, index: true },  // Indexed for fast lookups
  startTime: { type: Date, default: Date.now, index: true },           // Indexed for sorting/filtering
  endTime: { type: Date, default: null },
  callerLanguage: { type: String, default: 'English' },
  firstCallerMessage: { type: String, default: '' },
  severityScore: { type: Number, default: 1 },
  severityLabel: { type: String, default: 'Low', index: true },        // Indexed for severity filtering
  callStatus: { type: String, enum: ['ACTIVE', 'ASSESSING', 'ENDED', 'RESOLVED'], default: 'ACTIVE', index: true },
  emergencyType: { type: String, default: 'Unknown', index: true },    // Indexed for type filtering
  callerEmotion: { type: String, default: 'Calm' },
  backgroundSound: { type: String, default: 'Quiet' },
  aiBrainUsed: { type: String, default: 'NARAYANI v2' },
  aiBrainResponseTime: { type: Number, default: 0.8 },
  transcript: { type: [transcriptEntrySchema], default: [] },
  visionAnalysis: { type: mongoose.Schema.Types.Mixed, default: null },
  analysisSnapshot: { type: mongoose.Schema.Types.Mixed, default: null },
  mlVotes: { type: mongoose.Schema.Types.Mixed, default: null },
  dispatchConfirmed: { type: Boolean, default: false },
  ambulanceUnit: { type: String, default: null },
  paramedicName: { type: String, default: null },
  eta: { type: Number, default: null },
  ambulanceLocation: { type: mongoose.Schema.Types.Mixed, default: null },
  responderLocation: { type: mongoose.Schema.Types.Mixed, default: null },
  callerLocation: { type: mongoose.Schema.Types.Mixed, default: null },
  callerLocationAccuracy: { type: Number, default: null },
  isActive: { type: Boolean, default: true, index: true },             // Indexed for active call filtering
}, {
  timestamps: true  // Auto-creates createdAt and updatedAt
});

// Compound index for fast "recent active calls" queries
callSchema.index({ isActive: 1, startTime: -1 });

const Call = mongoose.model('Call', callSchema);

module.exports = Call;
