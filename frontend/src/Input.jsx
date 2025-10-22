import React, { useState } from 'react';

export default function Input({ onSend, isLoading }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div
      style={{
        borderTop: '1px solid rgba(255,255,255,0.1)',
        padding: '16px 20px',
        background: '#242938',
        boxShadow: '0 -2px 10px rgba(0,0,0,0.1)',
      }}
    >
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type error code (e.g., BMCR01) or ask a question..."
          disabled={isLoading}
          style={{
            flex: 1,
            background: '#1a1d29',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '10px',
            padding: '12px 16px',
            color: 'white',
            fontSize: '14px',
            outline: 'none',
            transition: 'border 0.2s',
          }}
          onFocus={(e) => (e.target.style.border = '1px solid #667eea')}
          onBlur={(e) =>
            (e.target.style.border = '1px solid rgba(255,255,255,0.1)')
          }
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{
            background:
              isLoading || !input.trim()
                ? 'rgba(102, 126, 234, 0.5)'
                : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            padding: '12px 24px',
            borderRadius: '10px',
            border: 'none',
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
            fontWeight: '600',
            fontSize: '14px',
            transition: 'all 0.2s',
            boxShadow:
              isLoading || !input.trim()
                ? 'none'
                : '0 2px 8px rgba(102, 126, 234, 0.3)',
            minWidth: '80px',
          }}
          onMouseEnter={(e) => {
            if (!isLoading && input.trim()) {
              e.target.style.transform = 'translateY(-1px)';
              e.target.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
            }
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow =
              isLoading || !input.trim()
                ? 'none'
                : '0 2px 8px rgba(102, 126, 234, 0.3)';
          }}
        >
          {isLoading ? <span>Thinking...</span> : <span>Send 📤</span>}
        </button>
      </form>
    </div>
  );
}
