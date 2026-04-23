import React, { useState } from 'react';

export default function QueryInput({ onAsk, isLoading }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onAsk(text.trim());
    }
  };

  return (
    <form className="query-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a customer support question..."
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading || !text.trim()}>
        {isLoading ? 'Asking...' : 'Ask'}
      </button>
    </form>
  );
}