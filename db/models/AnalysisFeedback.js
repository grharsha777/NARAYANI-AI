const { mongoose } = require('../connection');

// New feedback log: persist grounded JSON, analysis output, and dispatcher corrections without storing raw frames.
const analysisFeedbackSchema = new mongoose.Schema({
  callId: { type: String, required: true, index: true },
  analysisRevision: { type: String, required: true, index: true },
  generatedAt: { type: Date, default: Date.now, index: true },
  grounding: { type: mongoose.Schema.Types.Mixed, required: true },
  classification: { type: mongoose.Schema.Types.Mixed, required: true },
  analysis: { type: mongoose.Schema.Types.Mixed, required: true },
  confidence: { type: Number, default: 0 },
  contradictions: { type: [String], default: [] },
  dispatcherAssessment: { type: String, enum: ['ACCURATE', 'PARTIAL', 'INACCURATE', null], default: null },
  dispatcherNotes: { type: String, default: '' },
  updatedByDispatcherAt: { type: Date, default: null },
}, {
  timestamps: true,
});

analysisFeedbackSchema.index({ callId: 1, generatedAt: -1 });
analysisFeedbackSchema.index({ callId: 1, analysisRevision: 1 }, { unique: true });

const AnalysisFeedback = mongoose.model('AnalysisFeedback', analysisFeedbackSchema);

module.exports = AnalysisFeedback;
