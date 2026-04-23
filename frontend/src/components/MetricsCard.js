import React from 'react';

export default function MetricsCard({ label, value, highlight }) {
  return (
    <div className={`metrics-card ${highlight ? 'highlight' : ''}`}>
      <span className="metrics-label">{label}</span>
      <span className="metrics-value">{value}</span>
    </div>
  );
}