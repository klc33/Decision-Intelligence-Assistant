import React, { useState } from 'react';
import QueryInput from './components/QueryInput';
import AnswerDisplay from './components/AnswerDisplay';
import SourcePanel from './components/SourcePanel';
import ComparisonPanel from './components/ComparisonPanel';
import { postQuery } from './services/api';
import './App.css';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAsk = async (query) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const { success, data, error: err } = await postQuery(query);
    if (success) {
      setResult(data);
    } else {
      setError(err);
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header>
        <h1>Decision Intelligence Assistant</h1>
        <p>Compare RAG answers, zero‑shot LLM, and ML baseline side‑by‑side.</p>
      </header>

      <QueryInput onAsk={handleAsk} isLoading={loading} />

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <AnswerDisplay
            ragAnswer={result.rag_answer}
            nonRagAnswer={result.non_rag_answer}
          />
          <SourcePanel sources={result.rag_answer?.sources} />
          <ComparisonPanel
            mlPriority={result.ml_priority}
            llmPriority={result.llm_priority}
          />
        </div>
      )}
    </div>
  );
}