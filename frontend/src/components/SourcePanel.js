import React from 'react';

export default function SourcePanel({ sources }) {
  if (!sources || sources.length === 0) {
    return <div className="source-panel">No similar tickets found.</div>;
  }

  return (
    <div className="source-panel">
      <h3>📋 Retrieved Past Tickets</h3>
      {sources.map((ticket, idx) => (
        <div key={idx} className="source-item">
          <p className="source-text">{ticket.text}</p>
          <div className="source-meta">
            <span>Priority: {ticket.priority}</span>
            <span>Similarity: {(ticket.similarity * 100).toFixed(1)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}