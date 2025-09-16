/**
 * Professional Fraud Detection Platform
 * Main application with dashboard and network visualization
 */

import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  Fade
} from '@mui/material';
import { Timeline, Dashboard, Security } from '@mui/icons-material';
import FraudDetectionDashboard from './components/FraudDetectionDashboard';
import NetworkGraphVisualization from './components/NetworkGraphVisualization';

// Professional fraud detection theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      dark: '#115293',
      light: '#42a5f5',
    },
    secondary: {
      main: '#dc004e',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    success: {
      main: '#4caf50',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 8,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 8,
        },
      },
    },
  },
});

type ViewMode = 'dashboard' | 'graph';

function App() {
  const [currentView, setCurrentView] = useState<ViewMode>('dashboard');

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* Navigation Header */}
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <Security sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            FraudGuard 360 - Professional Fraud Detection Platform
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              color="inherit"
              startIcon={<Dashboard />}
              onClick={() => setCurrentView('dashboard')}
              variant={currentView === 'dashboard' ? 'outlined' : 'text'}
              sx={{ 
                borderColor: currentView === 'dashboard' ? 'rgba(255,255,255,0.5)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                }
              }}
            >
              Dashboard
            </Button>
            <Button
              color="inherit"
              startIcon={<Timeline />}
              onClick={() => setCurrentView('graph')}
              variant={currentView === 'graph' ? 'outlined' : 'text'}
              sx={{ 
                borderColor: currentView === 'graph' ? 'rgba(255,255,255,0.5)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                }
              }}
            >
              Network Graph
            </Button>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Box sx={{ minHeight: 'calc(100vh - 64px)' }}>
        {currentView === 'dashboard' && (
          <Fade in={currentView === 'dashboard'} timeout={300}>
            <div>
              <FraudDetectionDashboard 
                onGraphViewClick={() => setCurrentView('graph')}
              />
            </div>
          </Fade>
        )}
        
        {currentView === 'graph' && (
          <Fade in={currentView === 'graph'} timeout={300}>
            <div>
              <Container maxWidth={false} sx={{ py: 3 }}>
                <NetworkGraphVisualization />
              </Container>
            </div>
          </Fade>
        )}
      </Box>

      {/* Status Footer */}
      <Box 
        component="footer" 
        sx={{ 
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: 'rgba(255,255,255,0.9)',
          backdropFilter: 'blur(10px)',
          borderTop: '1px solid rgba(0,0,0,0.1)',
          py: 1,
          px: 2,
          zIndex: 1000
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            🛡️ Real-time fraud detection system active
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Professional demonstration platform • Interactive fraud analysis
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
