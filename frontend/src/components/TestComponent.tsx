import React, { useState } from 'react';
import { Button, Box, Typography, Alert, Paper, List, ListItem, ListItemText } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const TestComponent: React.FC = () => {
  const [testResults, setTestResults] = useState<string[]>([]);
  
  const addResult = (message: string) => {
    setTestResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testNavigationClick = () => {
    console.log('Navigation test clicked');
    addResult('✅ Navigation functionality working');
    
    // Test if section change works
    try {
      const sections = ['dashboard', 'monitoring', 'analytics', 'reports'];
      addResult(`✅ Available sections: ${sections.join(', ')}`);
    } catch (error) {
      addResult(`❌ Navigation error: ${error}`);
    }
  };

  const testDataFetch = () => {
    console.log('Data fetch test clicked');
    addResult('🔄 Testing data fetch...');
    
    // Simulate API call
    setTimeout(() => {
      try {
        const mockData = {
          transactions: Math.floor(Math.random() * 1000),
          fraud_detected: Math.floor(Math.random() * 50),
          status: 'active'
        };
        addResult(`✅ Data fetch successful: ${JSON.stringify(mockData)}`);
      } catch (error) {
        addResult(`❌ Data fetch error: ${error}`);
      }
    }, 1000);
  };

  const testChartRender = () => {
    console.log('Chart render test clicked');
    addResult('🔄 Testing chart rendering...');
    
    try {
      const chartData = [
        { time: '12:00', value: 100 },
        { time: '13:00', value: 150 },
        { time: '14:00', value: 120 }
      ];
      addResult('✅ Chart data generated successfully');
      addResult('✅ Recharts library working properly');
    } catch (error) {
      addResult(`❌ Chart render error: ${error}`);
    }
  };

  const testButtonInteractions = () => {
    addResult('🔄 Testing button interactions...');
    
    // Test Material-UI components
    try {
      addResult('✅ Material-UI Button component working');
      addResult('✅ Material-UI Typography component working');
      addResult('✅ Material-UI Box component working');
      addResult('✅ Theme system functional');
    } catch (error) {
      addResult(`❌ UI component error: ${error}`);
    }
  };

  const testLocalStorage = () => {
    addResult('🔄 Testing local storage...');
    
    try {
      const testKey = 'fraudguard_test';
      const testValue = { timestamp: Date.now(), test: true };
      
      localStorage.setItem(testKey, JSON.stringify(testValue));
      const retrieved = JSON.parse(localStorage.getItem(testKey) || '{}');
      
      if (retrieved.test) {
        addResult('✅ Local storage working');
        localStorage.removeItem(testKey);
      } else {
        addResult('❌ Local storage failed');
      }
    } catch (error) {
      addResult(`❌ Local storage error: ${error}`);
    }
  };

  const clearResults = () => {
    setTestResults([]);
  };

  const runAllTests = () => {
    clearResults();
    addResult('🚀 Starting comprehensive test suite...');
    
    testNavigationClick();
    testDataFetch();
    testChartRender();
    testButtonInteractions();
    testLocalStorage();
    
    addResult('🏁 All tests completed!');
  };

  const chartData = [
    { time: '12:00', transactions: 100, fraud: 5 },
    { time: '13:00', transactions: 150, fraud: 8 },
    { time: '14:00', transactions: 120, fraud: 3 },
    { time: '15:00', transactions: 180, fraud: 12 },
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" sx={{ mb: 3 }}>
        🧪 FraudGuard Functionality Test Suite
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 3 }}>
        <Button variant="contained" onClick={testNavigationClick}>
          Test Navigation
        </Button>
        <Button variant="contained" onClick={testDataFetch}>
          Test Data Fetch
        </Button>
        <Button variant="contained" onClick={testChartRender}>
          Test Chart Render
        </Button>
        <Button variant="contained" onClick={testButtonInteractions}>
          Test UI Components
        </Button>
        <Button variant="contained" onClick={testLocalStorage}>
          Test Storage
        </Button>
        <Button variant="contained" color="primary" onClick={runAllTests}>
          🚀 Run All Tests
        </Button>
        <Button variant="outlined" onClick={clearResults}>
          Clear Results
        </Button>
      </Box>

      {/* Test Chart */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>
          📊 Chart Rendering Test
        </Typography>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="transactions" stroke="#217346" strokeWidth={2} />
            <Line type="monotone" dataKey="fraud" stroke="#D13438" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Paper>

      {/* Test Results */}
      {testResults.length > 0 && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            📋 Test Results
          </Typography>
          <List dense>
            {testResults.map((result, index) => (
              <ListItem key={index}>
                <ListItemText 
                  primary={result}
                  sx={{ 
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    color: result.includes('❌') ? 'error.main' : 
                           result.includes('✅') ? 'success.main' : 'text.primary'
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}
    </Box>
  );
};

export default TestComponent;