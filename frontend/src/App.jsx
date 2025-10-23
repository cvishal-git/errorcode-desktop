import React, { useState, useEffect } from 'react';
import Chat from './Chat';
import Input from './Input';
import { queryErrorCode, healthCheck } from './api';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    let retries = 0;
    const maxRetries = 20;

    const checkBackend = async () => {
      try {
        const health = await healthCheck();
        if (health.status === 'ready' || health.status === 'not_ready') {
          setStatus('connected');
        } else {
          throw new Error('Backend not ready');
        }
      } catch (error) {
        retries++;
        if (retries < maxRetries) {
          setTimeout(checkBackend, 1000);
        } else {
          console.error('Backend error:', error);
          setStatus('error');
        }
      }
    };

    checkBackend();
  }, []);

  const handleSend = async (query) => {
    setMessages((prev) => [...prev, { text: query, isUser: true }]);
    setIsLoading(true);

    try {
      const response = await queryErrorCode(query);
      setMessages((prev) => [
        ...prev,
        {
          text: response.answer,
          isUser: false,
          media: response.media || [],
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          text: `Error: ${error.message}`,
          isUser: false,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  if (status === 'connecting') {
    return (
      <div
        style={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#1a1d29',
        }}
      >
        <div
          style={{
            width: '80px',
            height: '80px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '20px',
            fontSize: '36px',
            animation: 'pulse 2s ease-in-out infinite',
          }}
        >
          💬
        </div>
        <h2
          style={{
            fontSize: '24px',
            fontWeight: '600',
            marginBottom: '8px',
            color: '#e5e7eb',
          }}
        >
          Starting ErrorCode Assistant
        </h2>
        <p style={{ fontSize: '14px', color: '#9ca3af' }}>
          Initializing AI models...
        </p>
        <div style={{ display: 'flex', gap: '8px', marginTop: '20px' }}>
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: '#667eea',
              animation: 'bounce 1.4s infinite ease-in-out both',
              animationDelay: '-0.32s',
            }}
          />
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: '#667eea',
              animation: 'bounce 1.4s infinite ease-in-out both',
              animationDelay: '-0.16s',
            }}
          />
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: '#667eea',
              animation: 'bounce 1.4s infinite ease-in-out both',
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: '#1a1d29',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          padding: '16px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              background: 'white',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '18px',
              fontWeight: 'bold',
              color: '#667eea',
            }}
          >
            Q
          </div>
          <h1
            style={{
              color: 'white',
              fontSize: '18px',
              fontWeight: '600',
              margin: 0,
            }}
          >
            ErrorCode Assistant
          </h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background:
                status === 'connected'
                  ? '#10b981'
                  : status === 'error'
                  ? '#ef4444'
                  : '#f59e0b',
            }}
          ></span>
          <span style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px' }}>
            {status === 'connected'
              ? 'Online'
              : status === 'error'
              ? 'Offline'
              : 'Connecting...'}
          </span>
        </div>
      </div>

      <Chat messages={messages} isLoading={isLoading} />
      <Input onSend={handleSend} isLoading={isLoading} />
    </div>
  );
}

export default App;
