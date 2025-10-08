/**
 * Enhanced Notification System Component
 * Professional toast notifications to replace alert() calls
 */

import React, { memo } from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  Slide,
  Stack,
  Box,
  styled,
  alpha,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Info,
  Warning,
} from '@mui/icons-material';

const NotificationContainer = styled(Stack)(({ theme }) => ({
  position: 'fixed',
  top: 80, // Below the ribbon navigation
  right: 20,
  zIndex: 9999,
  maxWidth: '400px',
  width: '100%',
  pointerEvents: 'none',
}));

const NotificationItem = styled(Alert)(({ theme }) => ({
  pointerEvents: 'auto',
  marginBottom: theme.spacing(1),
  borderRadius: theme.spacing(1),
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  backdropFilter: 'blur(10px)',
  backgroundColor: alpha(theme.palette.background.paper, 0.95),
  border: '1px solid',
  borderColor: alpha(theme.palette.divider, 0.2),
  
  '& .MuiAlert-icon': {
    fontSize: '1.2rem',
  },
  
  '& .MuiAlert-message': {
    fontSize: '0.9rem',
    fontWeight: 500,
  },
  
  '&.MuiAlert-standardSuccess': {
    borderColor: alpha(theme.palette.success.main, 0.3),
    backgroundColor: alpha(theme.palette.success.light, 0.1),
  },
  
  '&.MuiAlert-standardError': {
    borderColor: alpha(theme.palette.error.main, 0.3),
    backgroundColor: alpha(theme.palette.error.light, 0.1),
  },
  
  '&.MuiAlert-standardWarning': {
    borderColor: alpha(theme.palette.warning.main, 0.3),
    backgroundColor: alpha(theme.palette.warning.light, 0.1),
  },
  
  '&.MuiAlert-standardInfo': {
    borderColor: alpha(theme.palette.info.main, 0.3),
    backgroundColor: alpha(theme.palette.info.light, 0.1),
  },
}));

interface Notification {
  id: number;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  timestamp: string;
  title?: string;
}

interface NotificationSystemProps {
  notifications: Notification[];
  onClose: (id: number) => void;
}

const getNotificationIcon = (type: string) => {
  switch (type) {
    case 'success':
      return <CheckCircle />;
    case 'error':
      return <Error />;
    case 'warning':
      return <Warning />;
    case 'info':
      return <Info />;
    default:
      return <Info />;
  }
};

const getNotificationTitle = (type: string) => {
  switch (type) {
    case 'success':
      return 'Success';
    case 'error':
      return 'Error';
    case 'warning':
      return 'Warning';
    case 'info':
      return 'Information';
    default:
      return 'Notification';
  }
};

export const NotificationSystem: React.FC<NotificationSystemProps> = ({
  notifications,
  onClose
}) => {
  return (
    <NotificationContainer spacing={1}>
      {notifications.map((notification) => (
        <Slide
          key={notification.id}
          direction="left"
          in={true}
          timeout={300}
        >
          <NotificationItem
            severity={notification.type}
            onClose={() => onClose(notification.id)}
            icon={getNotificationIcon(notification.type)}
            variant="standard"
          >
            <AlertTitle sx={{ fontSize: '0.85rem', fontWeight: 600 }}>
              {notification.title || getNotificationTitle(notification.type)}
            </AlertTitle>
            {notification.message}
          </NotificationItem>
        </Slide>
      ))}
    </NotificationContainer>
  );
};

export default memo(NotificationSystem);