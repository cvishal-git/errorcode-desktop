import React from 'react';
import ReactMarkdown from 'react-markdown';

export default function Chat({ messages, isLoading }) {
  const endRef = React.useRef(null);

  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        background: '#1a1d29',
      }}
    >
      {messages.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            marginTop: '80px',
            color: '#9ca3af',
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
              margin: '0 auto 20px',
              fontSize: '36px',
              color: 'white',
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
            ErrorCode Assistant
          </h2>
          <p style={{ fontSize: '14px' }}>
            Type an error code or describe your issue
          </p>
        </div>
      ) : (
        messages.map((msg, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              marginBottom: '16px',
              justifyContent: msg.isUser ? 'flex-end' : 'flex-start',
            }}
          >
            <div
              style={{
                borderRadius: '12px',
                padding: '12px 16px',
                maxWidth: '75%',
                background: msg.isUser
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : '#2d3748',
                color: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                fontSize: '14px',
                lineHeight: '1.5',
              }}
            >
              {msg.isUser ? (
                <p style={{ margin: 0 }}>{msg.text}</p>
              ) : (
                <div
                  style={{
                    color: '#e5e7eb',
                    wordBreak: 'break-word',
                  }}
                >
                  <ReactMarkdown
                    components={{
                      p: ({ node, ...props }) => (
                        <p style={{ margin: '8px 0' }} {...props} />
                      ),
                      h2: ({ node, ...props }) => (
                        <h2
                          style={{
                            fontSize: '16px',
                            fontWeight: '600',
                            margin: '12px 0 8px',
                          }}
                          {...props}
                        />
                      ),
                      ul: ({ node, ...props }) => (
                        <ul
                          style={{ paddingLeft: '20px', margin: '8px 0' }}
                          {...props}
                        />
                      ),
                      ol: ({ node, ...props }) => (
                        <ol
                          style={{ paddingLeft: '20px', margin: '8px 0' }}
                          {...props}
                        />
                      ),
                      li: ({ node, ...props }) => (
                        <li style={{ margin: '4px 0' }} {...props} />
                      ),
                      strong: ({ node, ...props }) => (
                        <strong style={{ color: '#60a5fa' }} {...props} />
                      ),
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
              )}
              {msg.media?.length > 0 && (
                <div
                  style={{
                    marginTop: '12px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                  }}
                >
                  {msg.media.map((img, idx) => (
                    <img
                      key={idx}
                      src={img.path}
                      alt={img.caption || ''}
                      style={{
                        borderRadius: '8px',
                        maxWidth: '300px',
                        cursor: 'pointer',
                        border: '2px solid rgba(255,255,255,0.1)',
                      }}
                      onClick={() => window.open(img.path, '_blank')}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        ))
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div
          style={{
            display: 'flex',
            marginBottom: '16px',
            justifyContent: 'flex-start',
          }}
        >
          <div
            style={{
              borderRadius: '12px',
              padding: '12px 16px',
              background: '#2d3748',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            }}
          >
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: '#667eea',
                  animation: 'bounce 1.4s infinite ease-in-out both',
                  animationDelay: '-0.32s',
                }}
              />
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: '#667eea',
                  animation: 'bounce 1.4s infinite ease-in-out both',
                  animationDelay: '-0.16s',
                }}
              />
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: '#667eea',
                  animation: 'bounce 1.4s infinite ease-in-out both',
                }}
              />
            </div>
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  );
}
