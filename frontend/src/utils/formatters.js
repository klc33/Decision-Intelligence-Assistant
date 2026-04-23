/**
 * Format latency from milliseconds to a readable string.
 */
export function formatLatency(ms) {
  if (ms < 1) return '<1ms';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Format cost in USD. Display up to 6 decimals, strip unnecessary zeros.
 */
export function formatCost(cost) {
  if (cost === 0 || cost == null) return '$0.00';
  return `$${cost.toFixed(6)}`;
}

/**
 * Format confidence (0-1) as percentage with 1 decimal.
 */
export function formatConfidence(confidence) {
  return `${(confidence * 100).toFixed(1)}%`;
}