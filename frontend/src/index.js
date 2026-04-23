import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log error to monitoring service (e.g., Sentry, LogRocket)
    console.error('Application Error:', error, errorInfo);
    
    // You can send to your backend analytics
    fetch('/api/log-error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        error: error.toString(),
        stack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent
      })
    }).catch(console.error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          textAlign: 'center',
          padding: '20px'
        }}>
          <div>
            <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
              🚨 Something went wrong
            </h1>
            <p style={{ fontSize: '1.1rem', marginBottom: '2rem', opacity: 0.9 }}>
              Our team has been notified. Please try refreshing the page.
            </p>
            <button
              onClick={() => window.location.reload()}
              style={{
                padding: '12px 32px',
                fontSize: '1rem',
                background: 'white',
                color: '#667eea',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 600,
                transition: 'transform 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
              onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
            >
              Refresh Page
            </button>
            {process.env.NODE_ENV === 'development' && (
              <pre style={{
                marginTop: '2rem',
                padding: '1rem',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '8px',
                textAlign: 'left',
                fontSize: '0.9rem',
                maxWidth: '600px',
                overflow: 'auto'
              }}>
                {this.state.error?.toString()}
              </pre>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Performance monitoring
if ('performance' in window && 'getEntriesByType' in performance) {
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = performance.getEntriesByType('navigation')[0];
      if (perfData) {
        const metrics = {
          dns: perfData.domainLookupEnd - perfData.domainLookupStart,
          connect: perfData.connectEnd - perfData.connectStart,
          ttfb: perfData.responseStart - perfData.requestStart,
          domLoad: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
          windowLoad: perfData.loadEventEnd - perfData.loadEventStart,
          timestamp: new Date().toISOString()
        };
        
        console.log('Performance Metrics:', metrics);
        
        // Send to analytics (optional)
        if (process.env.NODE_ENV === 'production') {
          fetch('/api/performance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(metrics)
          }).catch(console.error);
        }
      }
    }, 0);
  });
}

// Add service worker for PWA support (optional)
if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js').catch(error => {
      console.error('Service worker registration failed:', error);
    });
  });
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
