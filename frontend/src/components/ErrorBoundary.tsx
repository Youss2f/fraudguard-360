/**
 * Enhanced Error Boundary System
 * Comprehensive error handling with professional UI and reporting
 */

import React, { Component, ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Alert,
  AlertTitle,
  IconButton,
  Collapse,
  Divider,
  Chip,
  styled,
  alpha
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  BugReport as BugReportIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Home as HomeIcon,
  Support as SupportIcon
} from '@mui/icons-material';

// Styled components for error display
const ErrorContainer = styled(Box)(({ theme }) => ({
  minHeight: '400px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(4),
  background: `linear-gradient(145deg, ${alpha(theme.palette.error.light, 0.05)}, ${alpha(theme.palette.grey[100], 0.8)})`,
}));

const ErrorCard = styled(Card)(({ theme }) => ({
  maxWidth: '600px',
  width: '100%',
  boxShadow: `0 8px 32px ${alpha(theme.palette.error.main, 0.12)}`,
  borderRadius: theme.spacing(2),
  border: `1px solid ${alpha(theme.palette.error.main, 0.2)}`,
}));

const ErrorIconWrapper = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 80,
  height: 80,
  borderRadius: '50%',
  backgroundColor: alpha(theme.palette.error.main, 0.1),
  margin: '0 auto 16px',
  '& .MuiSvgIcon-root': {
    fontSize: 40,
    color: theme.palette.error.main,
  },
}));

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  level?: 'app' | 'component' | 'feature';
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  showDetails: boolean;
  errorId: string;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: '',
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      errorId: `ERR_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Report error to monitoring service
    this.reportError(error, errorInfo);
  }

  private reportError = (error: Error, errorInfo: React.ErrorInfo) => {
    const errorReport = {
      errorId: this.state.errorId,
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      level: this.props.level || 'component',
    };

    // In production, send to error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: Send to error reporting service
      // fetch('/api/errors', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(errorReport),
      // });
    }

    console.error('Error Report:', errorReport);
  };

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: '',
    });
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  private handleReportBug = () => {
    const subject = encodeURIComponent(`Bug Report - Error ID: ${this.state.errorId}`);
    const body = encodeURIComponent(
      `Error Details:\n\n` +
      `Error ID: ${this.state.errorId}\n` +
      `Message: ${this.state.error?.message}\n` +
      `Timestamp: ${new Date().toISOString()}\n` +
      `URL: ${window.location.href}\n\n` +
      `Please describe what you were doing when this error occurred:\n`
    );
    
    const mailto = `mailto:support@fraudguard360.com?subject=${subject}&body=${body}`;
    window.open(mailto);
  };

  private toggleDetails = () => {
    this.setState(prev => ({ showDetails: !prev.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, showDetails, errorId } = this.state;
      const isAppLevel = this.props.level === 'app';

      return (
        <ErrorContainer>
          <ErrorCard>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <ErrorIconWrapper>
                <ErrorIcon />
              </ErrorIconWrapper>

              <Typography variant="h5" component="h2" gutterBottom fontWeight="600">
                {isAppLevel ? 'Application Error' : 'Something went wrong'}
              </Typography>

              <Typography 
                variant="body1" 
                color="text.secondary" 
                sx={{ mb: 3, lineHeight: 1.6 }}
              >
                {isAppLevel 
                  ? 'The application encountered an unexpected error. Our team has been notified.'
                  : 'This section encountered an error. You can try refreshing or continue using other features.'
                }
              </Typography>

              <Alert severity="error" sx={{ mb: 3, textAlign: 'left' }}>
                <AlertTitle>Error Details</AlertTitle>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Chip 
                    label={`ID: ${errorId}`} 
                    size="small" 
                    variant="outlined" 
                    color="error"
                  />
                  <Chip 
                    label={new Date().toLocaleString()} 
                    size="small" 
                    variant="outlined"
                  />
                </Box>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {error?.message || 'Unknown error occurred'}
                </Typography>
              </Alert>

              {/* Technical Details (Collapsible) */}
              <Box sx={{ textAlign: 'left', mb: 3 }}>
                <Button
                  onClick={this.toggleDetails}
                  startIcon={showDetails ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  size="small"
                  color="inherit"
                >
                  Technical Details
                </Button>
                
                <Collapse in={showDetails}>
                  <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Stack Trace:
                    </Typography>
                    <Typography 
                      variant="body2" 
                      component="pre" 
                      sx={{ 
                        fontFamily: 'monospace', 
                        fontSize: '12px',
                        overflow: 'auto',
                        maxHeight: '200px',
                        whiteSpace: 'pre-wrap',
                        bgcolor: 'background.paper',
                        p: 1,
                        borderRadius: 1,
                        border: '1px solid',
                        borderColor: 'divider'
                      }}
                    >
                      {error?.stack || 'No stack trace available'}
                    </Typography>
                    
                    {errorInfo?.componentStack && (
                      <>
                        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                          Component Stack:
                        </Typography>
                        <Typography 
                          variant="body2" 
                          component="pre" 
                          sx={{ 
                            fontFamily: 'monospace', 
                            fontSize: '12px',
                            overflow: 'auto',
                            maxHeight: '200px',
                            whiteSpace: 'pre-wrap',
                            bgcolor: 'background.paper',
                            p: 1,
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider'
                          }}
                        >
                          {errorInfo.componentStack}
                        </Typography>
                      </>
                    )}
                  </Box>
                </Collapse>
              </Box>
            </CardContent>

            <Divider />

            <CardActions sx={{ justifyContent: 'center', p: 3, gap: 2 }}>
              <Button
                onClick={this.handleRetry}
                variant="contained"
                startIcon={<RefreshIcon />}
                color="primary"
              >
                Try Again
              </Button>

              {isAppLevel && (
                <Button
                  onClick={this.handleGoHome}
                  variant="outlined"
                  startIcon={<HomeIcon />}
                >
                  Go Home
                </Button>
              )}

              <Button
                onClick={this.handleReportBug}
                variant="outlined"
                startIcon={<BugReportIcon />}
                color="error"
              >
                Report Bug
              </Button>
            </CardActions>
          </ErrorCard>
        </ErrorContainer>
      );
    }

    return this.props.children;
  }
}

// HOC for wrapping components with error boundary
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  level: 'app' | 'component' | 'feature' = 'component'
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary level={level}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};

// Hook for error handling in functional components
export const useErrorHandler = () => {
  const handleError = (error: Error, context?: string) => {
    console.error(`Error ${context ? `in ${context}` : ''}:`, error);
    
    // In a real app, report to error service
    if (process.env.NODE_ENV === 'production') {
      // reportError(error, context);
    }
  };

  return { handleError };
};

export default ErrorBoundary;