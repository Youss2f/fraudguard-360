import React from 'react';

const SimpleTest = () => {
  return (
    <div style={{ 
      padding: '50px', 
      fontSize: '24px', 
      textAlign: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      minHeight: '100vh'
    }}>
      <h1>🚀 FraudGuard-360 is Working!</h1>
      <p>React is successfully rendering!</p>
      <div style={{ marginTop: '30px', padding: '20px', background: 'rgba(255,255,255,0.1)', borderRadius: '10px' }}>
        <p>✅ Frontend Container: Running</p>
        <p>✅ React App: Loaded</p>
        <p>✅ TypeScript: Compiled</p>
      </div>
    </div>
  );
};

export default SimpleTest;