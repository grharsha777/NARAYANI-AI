const mongoose = require('mongoose');

const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://harshagr702_db_user:A0Zd7F9Gf0VAftGc@ganapathi.6qx90ga.mongodb.net/NarayaniVoiceAI?retryWrites=true&w=majority&appName=GANAPATHI';

let isConnected = false;

/**
 * Connect to MongoDB Atlas. Uses connection pooling and caching 
 * to avoid redundant connections (critical for serverless/edge environments).
 */
async function connectDB() {
  if (isConnected) {
    return mongoose.connection;
  }

  try {
    await mongoose.connect(MONGO_URI, {
      maxPoolSize: 10,          // Fast connection reuse
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    });
    isConnected = true;
    console.log('✅ MongoDB Atlas connected successfully');
    return mongoose.connection;
  } catch (error) {
    console.error('❌ MongoDB connection failed:', error.message);
    throw error;
  }
}

/**
 * Disconnect from MongoDB (used during graceful shutdown).
 */
async function disconnectDB() {
  if (!isConnected) return;
  await mongoose.disconnect();
  isConnected = false;
  console.log('MongoDB disconnected');
}

module.exports = { connectDB, disconnectDB, mongoose };
