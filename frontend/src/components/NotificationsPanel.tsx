/**
 * Professional Notifications Panel Component
 * Real-time alerts and notification management
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Avatar,
  IconButton,
  Chip,
  Divider,
  Tabs,
  Tab,
  Badge,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  styled,
} from '@mui/material';
import {
  Notifications,
  Close,
  Warning,
  Error,
  Info,
  CheckCircle,
  Security,
  Person,
  TrendingUp,
  Delete,
  MarkEmailRead,
  Settings,
  VolumeOff,
  Schedule,
} from '@mui/icons-material';
import { format, formatDistanceToNow } from 'date-fns';
import { customColors } from '../theme/enterpriseTheme';

const NotificationCard = styled(Card)(({ theme, priority }: { theme?: any; priority: string }) => {
  const getBorderColor = () => {
    switch (priority) {
      case 'critical': return customColors.error[500];
      case 'high': return customColors.warning[500];
      case 'medium': return customColors.primary[500];
      case 'low': return customColors.success[500];
      default: return customColors.neutral[200];
    }
  };

  return {
    backgroundColor: customColors.background.paper,
    border: `1px solid ${customColors.neutral[200]}`,
    borderLeft: `4px solid ${getBorderColor()}`,
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
    marginBottom: '8px',
    transition: 'all 0.2s ease',
    '&:hover': {
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
      transform: 'translateY(-1px)',
    },
    '&.unread': {
      backgroundColor: customColors.primary[50],
    },
  };
});

interface Notification {
  id: string;
  type: 'fraud_alert' | 'system' | 'user_activity' | 'investigation' | 'report';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  actionRequired: boolean;
  relatedEntity?: {
    type: 'alert' | 'user' | 'transaction' | 'case';
    id: string;
    name: string;
  };
}

interface NotificationsPanelProps {
  open: boolean;
  onClose: () => void;
  notificationCount?: number;
}

const NotificationsPanel: React.FC<NotificationsPanelProps> = ({ 
  open, 
  onClose,
  notificationCount = 0
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [settings, setSettings] = useState({
    emailNotifications: true,
    pushNotifications: true,
    criticalOnly: false,
    quietHours: false,
    quietStart: '22:00',
    quietEnd: '08:00',
  });

  useEffect(() => {
    // Generate mock notifications
    const mockNotifications: Notification[] = [
      {
        id: 'not-001',
        type: 'fraud_alert',
        priority: 'critical',
        title: 'Critical Fraud Alert Detected',
        message: 'High-risk transaction pattern detected for user USR-7529. Immediate investigation required.',
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
        read: false,
        actionRequired: true,
        relatedEntity: { type: 'alert', id: 'ALT-69757', name: 'Card Testing Pattern' }
      },
      {
        id: 'not-002',
        type: 'system',
        priority: 'high',
        title: 'System Performance Alert',
        message: 'ML model processing latency increased by 15%. Check system resources.',
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
        read: false,
        actionRequired: false,
      },
      {
        id: 'not-003', 
        type: 'investigation',
        priority: 'medium',
        title: 'Case Assignment',
        message: 'You have been assigned to investigate case CASE-2024-0891.',
        timestamp: new Date(Date.now() - 45 * 60 * 1000),
        read: true,
        actionRequired: true,
        relatedEntity: { type: 'case', id: 'CASE-2024-0891', name: 'Account Takeover Investigation' }
      },
      {
        id: 'not-004',
        type: 'user_activity',
        priority: 'low',
        title: 'User Profile Updated',
        message: 'Risk profile updated for customer CUST-5252 based on recent activity.',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
        read: true,
        actionRequired: false,
        relatedEntity: { type: 'user', id: 'CUST-5252', name: 'John Smith' }
      },
      {
        id: 'not-005',
        type: 'report',
        priority: 'low',
        title: 'Weekly Report Ready',
        message: 'Your weekly fraud analysis report is ready for review.',
        timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000),
        read: true,
        actionRequired: false,
      }
    ];
    setNotifications(mockNotifications);
  }, []);

  const getNotificationIcon = (type: string, priority: string) => {
    const iconColor = priority === 'critical' ? 'error' : 
                     priority === 'high' ? 'warning' :
                     priority === 'medium' ? 'primary' : 'success';

    switch (type) {
      case 'fraud_alert':
        return <Security color={iconColor as any} />;
      case 'system':
        return <Warning color={iconColor as any} />;
      case 'user_activity':
        return <Person color={iconColor as any} />;
      case 'investigation':
        return <Info color={iconColor as any} />;
      case 'report':
        return <TrendingUp color={iconColor as any} />;
      default:
        return <Notifications color={iconColor as any} />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return customColors.error[500];
      case 'high': return customColors.warning[500];
      case 'medium': return customColors.primary[500];
      case 'low': return customColors.success[500];
      default: return customColors.neutral[500];
    }
  };

  const handleMarkAsRead = (notificationId: string) => {
    setNotifications(prev => 
      prev.map(notif => 
        notif.id === notificationId ? { ...notif, read: true } : notif
      )
    );
  };

  const handleDeleteNotification = (notificationId: string) => {
    setNotifications(prev => prev.filter(notif => notif.id !== notificationId));
  };

  const handleMarkAllAsRead = () => {
    setNotifications(prev => prev.map(notif => ({ ...notif, read: true })));
  };

  const handleClearAll = () => {
    if (window.confirm('Are you sure you want to clear all notifications?')) {
      setNotifications([]);
    }
  };

  const getFilteredNotifications = () => {
    switch (currentTab) {
      case 0: // All
        return notifications;
      case 1: // Unread
        return notifications.filter(n => !n.read);
      case 2: // Critical
        return notifications.filter(n => n.priority === 'critical' || n.priority === 'high');
      case 3: // Action Required
        return notifications.filter(n => n.actionRequired);
      default:
        return notifications;
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;
  const criticalCount = notifications.filter(n => n.priority === 'critical' || n.priority === 'high').length;
  const actionRequiredCount = notifications.filter(n => n.actionRequired).length;

  const tabLabels = [
    { label: 'All', count: notifications.length },
    { label: 'Unread', count: unreadCount },
    { label: 'Critical', count: criticalCount },
    { label: 'Action Required', count: actionRequiredCount },
  ];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '70vh',
          backgroundColor: customColors.background.default,
        }
      }}
    >
      <DialogTitle sx={{ 
        backgroundColor: customColors.background.ribbon,
        borderBottom: `2px solid ${customColors.primary[500]}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Box display="flex" alignItems="center" gap={2}>
          <Badge badgeContent={unreadCount} color="error">
            <Notifications color="primary" />
          </Badge>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Notifications
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <Box sx={{ backgroundColor: customColors.background.ribbon }}>
        <Tabs
          value={currentTab}
          onChange={(_, newValue) => setCurrentTab(newValue)}
          variant="fullWidth"
        >
          {tabLabels.map((tab, index) => (
            <Tab
              key={tab.label}
              label={
                <Box display="flex" alignItems="center" gap={1}>
                  {tab.label}
                  {tab.count > 0 && (
                    <Chip 
                      size="small" 
                      label={tab.count} 
                      color={index === 2 ? 'error' : 'primary'}
                    />
                  )}
                </Box>
              }
            />
          ))}
        </Tabs>
      </Box>

      <DialogContent sx={{ p: 0, height: '50vh', overflow: 'auto' }}>
        {currentTab < 4 ? (
          <Box>
            {/* Action Bar */}
            <Box sx={{ 
              p: 2, 
              borderBottom: `1px solid ${customColors.neutral[200]}`,
              backgroundColor: customColors.background.paper 
            }}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="subtitle2" color="text.secondary">
                  {getFilteredNotifications().length} notifications
                </Typography>
                <Box display="flex" gap={1}>
                  <Button
                    size="small"
                    startIcon={<MarkEmailRead />}
                    onClick={handleMarkAllAsRead}
                    disabled={unreadCount === 0}
                  >
                    Mark All Read
                  </Button>
                  <Button
                    size="small"
                    startIcon={<Delete />}
                    onClick={handleClearAll}
                    color="error"
                    disabled={notifications.length === 0}
                  >
                    Clear All
                  </Button>
                </Box>
              </Box>
            </Box>

            {/* Notifications List */}
            <List sx={{ p: 2 }}>
              {getFilteredNotifications().length === 0 ? (
                <Box textAlign="center" py={4}>
                  <Notifications sx={{ fontSize: 48, color: customColors.neutral[400], mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    No notifications
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    You're all caught up!
                  </Typography>
                </Box>
              ) : (
                getFilteredNotifications().map((notification) => (
                  <Box key={notification.id}>
                    <NotificationCard 
                      priority={notification.priority}
                      className={!notification.read ? 'unread' : ''}
                    >
                      <CardContent sx={{ p: 2 }}>
                        <Box display="flex" alignItems="flex-start" gap={2}>
                          <Avatar 
                            sx={{ 
                              bgcolor: 'transparent',
                              border: `2px solid ${getPriorityColor(notification.priority)}`,
                              width: 40,
                              height: 40
                            }}
                          >
                            {getNotificationIcon(notification.type, notification.priority)}
                          </Avatar>
                          
                          <Box flex={1}>
                            <Box display="flex" alignItems="center" gap={1} mb={1}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {notification.title}
                              </Typography>
                              <Chip 
                                size="small" 
                                label={notification.priority}
                                sx={{ 
                                  bgcolor: getPriorityColor(notification.priority),
                                  color: 'white',
                                  fontSize: '0.7rem'
                                }}
                              />
                              {notification.actionRequired && (
                                <Chip 
                                  size="small" 
                                  label="Action Required"
                                  color="warning"
                                  variant="outlined"
                                />
                              )}
                            </Box>
                            
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {notification.message}
                            </Typography>
                            
                            {notification.relatedEntity && (
                              <Typography variant="caption" color="primary">
                                Related: {notification.relatedEntity.name} ({notification.relatedEntity.id})
                              </Typography>
                            )}
                            
                            <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
                              {formatDistanceToNow(notification.timestamp, { addSuffix: true })}
                            </Typography>
                          </Box>
                          
                          <Box display="flex" flexDirection="column" gap={1}>
                            {!notification.read && (
                              <IconButton 
                                size="small"
                                onClick={() => handleMarkAsRead(notification.id)}
                                title="Mark as read"
                              >
                                <MarkEmailRead fontSize="small" />
                              </IconButton>
                            )}
                            <IconButton 
                              size="small"
                              onClick={() => handleDeleteNotification(notification.id)}
                              title="Delete notification"
                            >
                              <Delete fontSize="small" />
                            </IconButton>
                          </Box>
                        </Box>
                      </CardContent>
                    </NotificationCard>
                  </Box>
                ))
              )}
            </List>
          </Box>
        ) : (
          // Settings Tab
          <Box p={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
              Notification Settings
            </Typography>
            
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Delivery Preferences
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.emailNotifications}
                      onChange={(e) => setSettings(prev => ({ ...prev, emailNotifications: e.target.checked }))}
                    />
                  }
                  label="Email Notifications"
                  sx={{ mb: 1 }}
                />
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.pushNotifications}
                      onChange={(e) => setSettings(prev => ({ ...prev, pushNotifications: e.target.checked }))}
                    />
                  }
                  label="Browser Push Notifications"
                  sx={{ mb: 1 }}
                />
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.criticalOnly}
                      onChange={(e) => setSettings(prev => ({ ...prev, criticalOnly: e.target.checked }))}
                    />
                  }
                  label="Critical Alerts Only"
                />
              </CardContent>
            </Card>
            
            <Card>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Quiet Hours
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.quietHours}
                      onChange={(e) => setSettings(prev => ({ ...prev, quietHours: e.target.checked }))}
                    />
                  }
                  label="Enable Quiet Hours"
                  sx={{ mb: 2 }}
                />
                
                {settings.quietHours && (
                  <Box display="flex" gap={2}>
                    <TextField
                      label="Start Time"
                      type="time"
                      size="small"
                      value={settings.quietStart}
                      onChange={(e) => setSettings(prev => ({ ...prev, quietStart: e.target.value }))}
                      InputLabelProps={{ shrink: true }}
                    />
                    <TextField
                      label="End Time"
                      type="time"
                      size="small"
                      value={settings.quietEnd}
                      onChange={(e) => setSettings(prev => ({ ...prev, quietEnd: e.target.value }))}
                      InputLabelProps={{ shrink: true }}
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ 
        p: 2, 
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default NotificationsPanel;