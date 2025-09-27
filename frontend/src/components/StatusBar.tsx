/**
 * Professional Status Bar Component
 * Bottom status information bar with system status, user context, and quick actions
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  LinearProgress,
  styled,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error,
  CloudDone,
  CloudOff,
  Security,
  Speed,
  Person,
  Schedule,
  Notifications,
  ZoomIn,
  ZoomOut,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const StatusBarContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '2px 12px',
  backgroundColor: customColors.background.ribbon,
  borderTop: `1px solid ${customColors.neutral[200]}`,
  height: '24px',
  fontSize: '11px',
  color: customColors.neutral[700],
}));

const StatusSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
}));

const StatusText = styled(Typography)(({ theme }) => ({
  fontSize: '11px',
  fontWeight: 400,
  color: customColors.neutral[700],
}));

const StatusIndicator = styled(Box)<{ status: 'success' | 'warning' | 'error' }>(({ status }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
  color: status === 'success' 
    ? customColors.success[600] 
    : status === 'warning' 
    ? customColors.warning[600] 
    : customColors.error[600],
}));

interface StatusBarProps {
  currentUser?: string;
  systemStatus?: 'online' | 'offline' | 'maintenance';
  lastUpdate?: Date;
  processingTasks?: number;
  zoomLevel?: number;
  onZoomChange?: (level: number) => void;
}

const StatusBar: React.FC<StatusBarProps> = ({
  currentUser = 'User',
  systemStatus = 'online',
  lastUpdate = new Date(),
  processingTasks = 0,
  zoomLevel = 100,
  onZoomChange,
}) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected'>('connected');
  const [alertCount, setAlertCount] = useState(3);

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Simulate connection monitoring
  useEffect(() => {
    const checkConnection = () => {
      setConnectionStatus(navigator.onLine ? 'connected' : 'disconnected');
    };

    window.addEventListener('online', checkConnection);
    window.addEventListener('offline', checkConnection);
    
    return () => {
      window.removeEventListener('online', checkConnection);
      window.removeEventListener('offline', checkConnection);
    };
  }, []);

  const getSystemStatusIcon = () => {
    switch (systemStatus) {
      case 'online':
        return <CheckCircle sx={{ fontSize: 12 }} />;
      case 'maintenance':
        return <Warning sx={{ fontSize: 12 }} />;
      case 'offline':
        return <Error sx={{ fontSize: 12 }} />;
      default:
        return <CheckCircle sx={{ fontSize: 12 }} />;
    }
  };

  const getSystemStatusColor = (): 'success' | 'warning' | 'error' => {
    switch (systemStatus) {
      case 'online':
        return 'success';
      case 'maintenance':
        return 'warning';
      case 'offline':
        return 'error';
      default:
        return 'success';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  const formatLastUpdate = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins === 1) return '1 min ago';
    if (diffMins < 60) return `${diffMins} mins ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours === 1) return '1 hour ago';
    if (diffHours < 24) return `${diffHours} hours ago`;
    
    return date.toLocaleDateString();
  };

  const handleZoomIn = () => {
    if (onZoomChange && zoomLevel < 200) {
      onZoomChange(zoomLevel + 10);
    }
  };

  const handleZoomOut = () => {
    if (onZoomChange && zoomLevel > 50) {
      onZoomChange(zoomLevel - 10);
    }
  };

  return (
    <StatusBarContainer>
      {/* Left Section - System Status */}
      <StatusSection>
        <StatusIndicator status={getSystemStatusColor()}>
          {getSystemStatusIcon()}
          <StatusText>
            System {systemStatus === 'online' ? 'Online' : systemStatus === 'maintenance' ? 'Maintenance' : 'Offline'}
          </StatusText>
        </StatusIndicator>
        
        <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
        
        <StatusIndicator status={connectionStatus === 'connected' ? 'success' : 'error'}>
          {connectionStatus === 'connected' ? 
            <CloudDone sx={{ fontSize: 12 }} /> : 
            <CloudOff sx={{ fontSize: 12 }} />
          }
          <StatusText>
            {connectionStatus === 'connected' ? 'Connected' : 'Offline'}
          </StatusText>
        </StatusIndicator>
        
        {processingTasks > 0 && (
          <>
            <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
            <StatusIndicator status="warning">
              <Speed sx={{ fontSize: 12 }} />
              <StatusText>
                Processing {processingTasks} task{processingTasks !== 1 ? 's' : ''}
              </StatusText>
            </StatusIndicator>
          </>
        )}
      </StatusSection>

      {/* Center Section - Context Information */}
      <StatusSection>
        <StatusIndicator status="success">
          <Person sx={{ fontSize: 12 }} />
          <StatusText>{currentUser}</StatusText>
        </StatusIndicator>
        
        <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
        
        <StatusIndicator status="success">
          <Security sx={{ fontSize: 12 }} />
          <StatusText>Secure Session</StatusText>
        </StatusIndicator>
        
        {alertCount > 0 && (
          <>
            <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
            <StatusIndicator status="warning">
              <Notifications sx={{ fontSize: 12 }} />
              <StatusText>{alertCount} Alert{alertCount !== 1 ? 's' : ''}</StatusText>
            </StatusIndicator>
          </>
        )}
      </StatusSection>

      {/* Right Section - Time and Controls */}
      <StatusSection>
        <StatusText>
          Last Update: {formatLastUpdate(lastUpdate)}
        </StatusText>
        
        <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
        
        {/* Zoom Controls */}
        <Box display="flex" alignItems="center" gap="4px">
          <Tooltip title="Zoom Out">
            <IconButton 
              size="small" 
              onClick={handleZoomOut}
              disabled={zoomLevel <= 50}
              sx={{ width: 16, height: 16, p: 0 }}
            >
              <ZoomOut sx={{ fontSize: 10 }} />
            </IconButton>
          </Tooltip>
          
          <StatusText sx={{ minWidth: '35px', textAlign: 'center' }}>
            {zoomLevel}%
          </StatusText>
          
          <Tooltip title="Zoom In">
            <IconButton 
              size="small" 
              onClick={handleZoomIn}
              disabled={zoomLevel >= 200}
              sx={{ width: 16, height: 16, p: 0 }}
            >
              <ZoomIn sx={{ fontSize: 10 }} />
            </IconButton>
          </Tooltip>
        </Box>
        
        <Divider orientation="vertical" flexItem sx={{ height: '16px' }} />
        
        <StatusIndicator status="success">
          <Schedule sx={{ fontSize: 12 }} />
          <StatusText>{formatTime(currentTime)}</StatusText>
        </StatusIndicator>
      </StatusSection>
    </StatusBarContainer>
  );
};

export default StatusBar;