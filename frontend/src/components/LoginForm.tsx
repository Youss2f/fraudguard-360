import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  Divider,
  Avatar,
  CircularProgress
} from '@mui/material';
import { LockOutlined, BusinessOutlined } from '@mui/icons-material';
import { styled } from '@mui/material/styles';

const LoginContainer = styled(Box)(({ theme }) => ({
  minHeight: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
  padding: theme.spacing(2)
}));

const LoginCard = styled(Card)(({ theme }) => ({
  minWidth: 400,
  maxWidth: 450,
  padding: theme.spacing(3),
  borderRadius: 12,
  boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
  border: '1px solid #e0e0e0'
}));

const LogoContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginBottom: theme.spacing(3),
  gap: theme.spacing(1)
}));

const DemoCredentials = styled(Box)(({ theme }) => ({
  backgroundColor: '#f8f9fa',
  border: '1px solid #e9ecef',
  borderRadius: 8,
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  fontSize: '0.875rem'
}));

interface LoginFormProps {
  onLogin: (user: any) => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (data.success) {
        // Store token
        localStorage.setItem('fraudguard_token', data.token);
        localStorage.setItem('fraudguard_user', JSON.stringify(data.user));
        onLogin(data.user);
      } else {
        setError('Invalid credentials');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = (demoUser: string) => {
    const credentials = {
      admin: { username: 'admin', password: 'admin123' },
      analyst: { username: 'analyst', password: 'analyst123' },
      viewer: { username: 'viewer', password: 'viewer123' }
    };

    const creds = credentials[demoUser as keyof typeof credentials];
    setUsername(creds.username);
    setPassword(creds.password);
  };

  return (
    <LoginContainer>
      <LoginCard>
        <CardContent>
          <LogoContainer>
            <Avatar sx={{ bgcolor: '#1976d2', width: 48, height: 48 }}>
              <BusinessOutlined />
            </Avatar>
            <Box>
              <Typography variant="h4" fontWeight="bold" color="primary">
                FraudGuard
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Enterprise Fraud Detection
              </Typography>
            </Box>
          </LogoContainer>

          <Divider sx={{ mb: 3 }} />

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Username"
              variant="outlined"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              margin="normal"
              required
              autoFocus
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="Password"
              type="password"
              variant="outlined"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              margin="normal"
              required
              sx={{ mb: 3 }}
            />

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <LockOutlined />}
              sx={{
                py: 1.5,
                fontSize: '1rem',
                fontWeight: 600,
                textTransform: 'none'
              }}
            >
              {loading ? 'Signing In...' : 'Sign In'}
            </Button>
          </form>

          <DemoCredentials>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              Demo Credentials:
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button
                size="small"
                variant="text"
                onClick={() => handleDemoLogin('admin')}
                sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
              >
                <strong>Administrator:</strong>&nbsp;admin / admin123
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => handleDemoLogin('analyst')}
                sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
              >
                <strong>Fraud Analyst:</strong>&nbsp;analyst / analyst123
              </Button>
              <Button
                size="small"
                variant="text"
                onClick={() => handleDemoLogin('viewer')}
                sx={{ justifyContent: 'flex-start', textTransform: 'none' }}
              >
                <strong>Viewer:</strong>&nbsp;viewer / viewer123
              </Button>
            </Box>
          </DemoCredentials>
        </CardContent>
      </LoginCard>
    </LoginContainer>
  );
};

export default LoginForm;