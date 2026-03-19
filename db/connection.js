const mongoose = require('mongoose');

const MONGO_URI = process.env.MONGODB_URI;

if (!MONGO_URI) {
  console.error('❌ MONGODB_URI is not defined in environment variables!');
}


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
