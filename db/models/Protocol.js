const { mongoose } = require('../connection');

const protocolSchema = new mongoose.Schema({
  emergencyType: { type: String, required: true, unique: true, index: true },
  title: { type: String, required: true },
  severityRange: { type: [Number], default: [] },
  immediateSteps: { type: [mongoose.Schema.Types.Mixed], default: [] },
  infantSteps: { type: [mongoose.Schema.Types.Mixed], default: [] },
  homeRemedies: { type: [mongoose.Schema.Types.Mixed], default: [] },
  whatNotToDo: { type: [mongoose.Schema.Types.Mixed], default: [] },
  paramedicAlert: { type: String, default: '' },
  estimatedSafeWindowMinutes: { type: Number, default: 0 },
}, {
  timestamps: true
});

const Protocol = mongoose.model('Protocol', protocolSchema);

module.exports = Protocol;
