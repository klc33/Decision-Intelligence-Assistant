import React from 'react';
import { formatLatency, formatCost } from '../utils/formatters';

export default function AnswerDisplay({ ragAnswer, nonRagAnswer }) {
  return (
    <div className="answer-display">
      <div className="answer-card">
        <h3>🔍 RAG Answer (with context)</h3>
        <p>{ragAnswer?.answer || 'No answer yet'}</p>
        <div className="meta">
          <span>Latency: {formatLatency(ragAnswer?.latency_ms)}</span>
          <span>Cost: {formatCost(ragAnswer?.cost_usd)}</span>
        </div>
      </div>
      <div className="answer-card">
        <h3>🧠 Non‑RAG Answer (LLM only)</h3>
        <p>{nonRagAnswer?.answer || 'No answer yet'}</p>
        <div className="meta">
          <span>Latency: {formatLatency(nonRagAnswer?.latency_ms)}</span>
          <span>Cost: {formatCost(nonRagAnswer?.cost_usd)}</span>
        </div>
      </div>
    </div>
  );
}