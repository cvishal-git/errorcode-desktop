import React, { useState, useEffect } from 'react';
import Chat from './Chat';
import Input from './Input';
import { queryErrorCode, healthCheck } from './api';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await healthCheck();
        setStatus('connected');
      } catch (error) {
        console.error('Backend error:', error);
        setStatus('error');
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
                  : status === 'connecting'
                  ? '#f59e0b'
                  : '#ef4444',
            }}
          ></span>
          <span style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px' }}>
            {status === 'connected'
              ? 'Online'
              : status === 'connecting'
              ? 'Connecting...'
              : 'Offline'}
          </span>
        </div>
      </div>

      <Chat messages={messages} isLoading={isLoading} />
      <Input onSend={handleSend} isLoading={isLoading} />
    </div>
  );
}

export default App;
