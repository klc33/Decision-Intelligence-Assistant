import React from 'react';
import MetricsCard from './MetricsCard';
import { formatLatency, formatCost, formatConfidence } from '../utils/formatters';

export default function ComparisonPanel({ mlPriority, llmPriority }) {
  return (
    <div className="comparison-panel">
      <h3>⚖️ Priority Prediction Comparison</h3>
      <div className="comparison-grid">
        <div className="method-column">
          <h4>ML Classifier (baseline)</h4>
          <p className="prediction">
            Prediction: <strong>{mlPriority?.prediction || '-'}</strong>
          </p>
          <div className="metrics-row">
            <MetricsCard
              label="Confidence"
              value={mlPriority?.confidence ? formatConfidence(mlPriority.confidence) : '-'}
            />
            <MetricsCard
              label="Latency"
              value={mlPriority?.latency_ms != null ? formatLatency(mlPriority.latency_ms) : '-'}
            />
            <MetricsCard
              label="Cost"
              value={mlPriority?.cost_usd != null ? formatCost(mlPriority.cost_usd) : '-'}
              highlight
            />
          </div>
        </div>
        <div className="method-column">
          <h4>LLM Zero‑Shot</h4>
          <p className="prediction">
            Prediction: <strong>{llmPriority?.prediction || '-'}</strong>
          </p>
          {llmPriority?.reasoning && (
            <p className="reasoning">Reason: {llmPriority.reasoning}</p>
          )}
          <div className="metrics-row">

            <MetricsCard
              label="Latency"
              value={llmPriority?.latency_ms != null ? formatLatency(llmPriority.latency_ms) : '-'}
            />
            <MetricsCard
              label="Cost"
              value={llmPriority?.cost_usd != null ? formatCost(llmPriority.cost_usd) : '-'}
            />
          </div>
        </div>
      </div>
    </div>
  );
}