const redis = require('redis');

/**
 * Redis client for high-performance protocol caching.
 * Fully optional — if Redis is not running, everything falls back
 * to In-Memory / MongoDB with zero error spam.
 */
let client = null;
let isConnected = false;

function createClient() {
  try {
    const c = redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      socket: {
        connectTimeoutMs: 3000,
        reconnectStrategy: (retries) => {
          // Stop retrying after 1 attempt — no Redis means optional
          if (retries >= 1) {
            return false; // Stop reconnecting
          }
          return 1000;
        }
      }
    });

    // Suppress ALL connection errors silently
    c.on('error', () => {});

    return c;
  } catch (_err) {
    return null;
  }
}

async function connectRedis() {
  if (isConnected && client) return client;

  try {
    client = createClient();
    if (!client) {
      console.log('ℹ️  Redis skipped (client creation failed). Using In-Memory/MongoDB fallback.');
      return null;
    }

    await client.connect();
    isConnected = true;
    console.log('✅ Redis connected successfully');
    return client;
  } catch (_err) {
    console.log('ℹ️  Redis not available. Using In-Memory/MongoDB fallback (no errors).');
    // Destroy the failed client to prevent error event spam
    if (client) {
      try { client.disconnect(); } catch (_) {}
    }
    client = null;
    isConnected = false;
    return null;
  }
}

module.exports = { connectRedis, get redisClient() { return client; } };
