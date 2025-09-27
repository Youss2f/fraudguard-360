/**
 * Professional Share Dashboard Panel Component
 * Dashboard sharing with permissions and link generation
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
  Card,
  CardContent,
  CardActions,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Avatar,
  Switch,
  FormControlLabel,
  RadioGroup,
  Radio,
  FormGroup,
  Checkbox,
  Divider,
  Paper,
  Tooltip,
  Alert,
  styled,
} from '@mui/material';
import {
  Share,
  Close,
  Link,
  Email,
  ContentCopy,
  QrCode,
  Security,
  Visibility,
  VisibilityOff,
  Schedule,
  Person,
  Group,
  Public,
  Lock,
  CheckCircle,
  Warning,
  Delete,
  Edit,
  Send,
  Download,
} from '@mui/icons-material';
import { format, addDays, addWeeks, addMonths } from 'date-fns';
import { customColors } from '../theme/enterpriseTheme';

const ShareCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '8px',
  transition: 'all 0.2s ease',
  '&:hover': {
    borderColor: customColors.primary[500],
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
  },
}));

interface ShareDashboardPanelProps {
  open: boolean;
  onClose: () => void;
}

interface ShareLink {
  id: string;
  name: string;
  url: string;
  permissions: string[];
  expiration: Date | null;
  password: boolean;
  createdBy: string;
  createdDate: Date;
  accessCount: number;
  lastAccessed: Date | null;
  recipients: string[];
  status: 'active' | 'expired' | 'disabled';
}

interface SharePermission {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

const ShareDashboardPanel: React.FC<ShareDashboardPanelProps> = ({ 
  open, 
  onClose 
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [shareLinks, setShareLinks] = useState<ShareLink[]>([]);
  const [newShare, setNewShare] = useState({
    name: '',
    description: '',
    permissions: [] as string[],
    expiration: 'never',
    customExpiration: new Date(),
    password: '',
    requirePassword: false,
    recipients: [] as string[],
    allowPublicAccess: false,
  });
  const [generatedLink, setGeneratedLink] = useState<string>('');
  const [showLinkGenerated, setShowLinkGenerated] = useState(false);

  const availablePermissions: SharePermission[] = [
    {
      id: 'view_dashboard',
      name: 'View Dashboard',
      description: 'Can view the main dashboard and charts',
      icon: <Visibility />
    },
    {
      id: 'view_alerts',
      name: 'View Alerts',
      description: 'Can view fraud alerts and notifications',
      icon: <Warning />
    },
    {
      id: 'export_data',
      name: 'Export Data',
      description: 'Can export dashboard data and reports',
      icon: <Download />
    },
    {
      id: 'view_investigations',
      name: 'View Investigations',
      description: 'Can view ongoing investigations',
      icon: <Security />
    },
    {
      id: 'view_reports',
      name: 'View Reports',
      description: 'Can access and view generated reports',
      icon: <CheckCircle />
    }
  ];

  useEffect(() => {
    // Generate mock share links
    const now = new Date();
    const mockLinks: ShareLink[] = [
      {
        id: 'link-001',
        name: 'Monthly Executive Report',
        url: 'https://fraudguard.company.com/share/abc123def456',
        permissions: ['view_dashboard', 'view_reports'],
        expiration: addMonths(now, 1),
        password: true,
        createdBy: 'Sarah Johnson',
        createdDate: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000),
        accessCount: 15,
        lastAccessed: new Date(now.getTime() - 2 * 60 * 60 * 1000),
        recipients: ['executive@company.com', 'ceo@company.com'],
        status: 'active'
      },
      {
        id: 'link-002',
        name: 'Team Dashboard Access',
        url: 'https://fraudguard.company.com/share/xyz789abc123',
        permissions: ['view_dashboard', 'view_alerts', 'export_data'],
        expiration: addWeeks(now, 2),
        password: false,
        createdBy: 'Mike Chen',
        createdDate: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
        accessCount: 42,
        lastAccessed: new Date(now.getTime() - 30 * 60 * 1000),
        recipients: ['team@company.com'],
        status: 'active'
      },
      {
        id: 'link-003',
        name: 'Audit Review Access',
        url: 'https://fraudguard.company.com/share/def456ghi789',
        permissions: ['view_dashboard', 'view_investigations', 'view_reports'],
        expiration: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000),
        password: true,
        createdBy: 'Emma Davis',
        createdDate: new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000),
        accessCount: 8,
        lastAccessed: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000),
        recipients: ['audit@company.com'],
        status: 'expired'
      }
    ];
    setShareLinks(mockLinks);
  }, []);

  const handlePermissionToggle = (permissionId: string) => {
    setNewShare(prev => ({
      ...prev,
      permissions: prev.permissions.includes(permissionId)
        ? prev.permissions.filter(p => p !== permissionId)
        : [...prev.permissions, permissionId]
    }));
  };

  const handleGenerateLink = () => {
    // Generate a mock shareable link
    const linkId = Math.random().toString(36).substring(2, 15);
    const baseUrl = 'https://fraudguard.company.com/share/';
    const newLink = baseUrl + linkId;
    
    setGeneratedLink(newLink);
    setShowLinkGenerated(true);

    // Add to share links list
    const newShareLink: ShareLink = {
      id: `link-${Date.now()}`,
      name: newShare.name || 'Untitled Share',
      url: newLink,
      permissions: newShare.permissions,
      expiration: newShare.expiration === 'never' ? null : 
                  newShare.expiration === '1day' ? addDays(new Date(), 1) :
                  newShare.expiration === '1week' ? addWeeks(new Date(), 1) :
                  newShare.expiration === '1month' ? addMonths(new Date(), 1) :
                  newShare.customExpiration,
      password: newShare.requirePassword,
      createdBy: 'Current User',
      createdDate: new Date(),
      accessCount: 0,
      lastAccessed: null,
      recipients: newShare.recipients,
      status: 'active'
    };

    setShareLinks(prev => [newShareLink, ...prev]);
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(generatedLink);
    // You could show a toast notification here
  };

  const handleDeleteLink = (linkId: string) => {
    setShareLinks(prev => prev.filter(link => link.id !== linkId));
  };

  const handleToggleLinkStatus = (linkId: string) => {
    setShareLinks(prev => prev.map(link => 
      link.id === linkId 
        ? { ...link, status: link.status === 'active' ? 'disabled' : 'active' }
        : link
    ));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return customColors.success[500];
      case 'expired': return customColors.error[500];
      case 'disabled': return customColors.neutral[500];
      default: return customColors.neutral[500];
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle color="success" />;
      case 'expired': return <Warning color="error" />;
      case 'disabled': return <Lock color="disabled" />;
      default: return <Warning />;
    }
  };

  const renderCreateShare = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Create New Share Link
      </Typography>
      
      {/* Basic Information */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Basic Information
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Share Name"
                value={newShare.name}
                onChange={(e) => setNewShare(prev => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Executive Dashboard Access"
                sx={{ mb: 2 }}
              />
              
              <TextField
                fullWidth
                label="Description (Optional)"
                multiline
                rows={3}
                value={newShare.description}
                onChange={(e) => setNewShare(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe what this share link provides access to..."
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Expiration</InputLabel>
                <Select
                  value={newShare.expiration}
                  onChange={(e) => setNewShare(prev => ({ ...prev, expiration: e.target.value }))}
                >
                  <MenuItem value="never">Never Expires</MenuItem>
                  <MenuItem value="1day">1 Day</MenuItem>
                  <MenuItem value="1week">1 Week</MenuItem>
                  <MenuItem value="1month">1 Month</MenuItem>
                  <MenuItem value="custom">Custom Date</MenuItem>
                </Select>
              </FormControl>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={newShare.requirePassword}
                    onChange={(e) => setNewShare(prev => ({ ...prev, requirePassword: e.target.checked }))}
                  />
                }
                label="Require Password"
              />
              
              {newShare.requirePassword && (
                <TextField
                  fullWidth
                  label="Password"
                  type="password"
                  value={newShare.password}
                  onChange={(e) => setNewShare(prev => ({ ...prev, password: e.target.value }))}
                  sx={{ mt: 1 }}
                />
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Permissions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Access Permissions
          </Typography>
          
          <Grid container spacing={2}>
            {availablePermissions.map((permission) => (
              <Grid item xs={12} sm={6} md={4} key={permission.id}>
                <Card 
                  variant="outlined"
                  sx={{ 
                    cursor: 'pointer',
                    border: newShare.permissions.includes(permission.id) 
                      ? `2px solid ${customColors.primary[500]}` 
                      : `1px solid ${customColors.neutral[200]}`,
                    backgroundColor: newShare.permissions.includes(permission.id) 
                      ? customColors.primary[50] 
                      : 'transparent'
                  }}
                  onClick={() => handlePermissionToggle(permission.id)}
                >
                  <CardContent sx={{ textAlign: 'center', py: 2 }}>
                    <Box sx={{ color: customColors.primary[500], mb: 1 }}>
                      {permission.icon}
                    </Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                      {permission.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {permission.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Recipients */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            Recipients (Optional)
          </Typography>
          
          <TextField
            fullWidth
            label="Email Addresses"
            placeholder="Enter email addresses separated by commas"
            helperText="Leave empty for public access link"
            multiline
            rows={2}
          />
        </CardContent>
      </Card>

      {/* Generate Button */}
      <Box display="flex" gap={2}>
        <Button
          variant="contained"
          startIcon={<Link />}
          onClick={handleGenerateLink}
          disabled={newShare.permissions.length === 0}
        >
          Generate Share Link
        </Button>
        <Button
          variant="outlined"
          startIcon={<QrCode />}
          disabled={!generatedLink}
        >
          Generate QR Code
        </Button>
      </Box>

      {/* Generated Link Display */}
      {showLinkGenerated && (
        <Alert severity="success" sx={{ mt: 3 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Share Link Generated Successfully!
          </Typography>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <TextField
              fullWidth
              value={generatedLink}
              variant="outlined"
              size="small"
              InputProps={{
                readOnly: true,
              }}
            />
            <Tooltip title="Copy Link">
              <IconButton onClick={handleCopyLink}>
                <ContentCopy />
              </IconButton>
            </Tooltip>
          </Box>
          <Typography variant="body2" color="text.secondary">
            This link has been added to your active shares and can be managed below.
          </Typography>
        </Alert>
      )}
    </Box>
  );

  const renderActiveShares = () => (
    <Box p={3}>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
        Active Share Links
      </Typography>
      
      <Grid container spacing={2}>
        {shareLinks.map((link) => (
          <Grid item xs={12} md={6} key={link.id}>
            <ShareCard>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {link.name}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getStatusIcon(link.status)}
                    <Chip 
                      size="small" 
                      label={link.status}
                      sx={{ 
                        bgcolor: getStatusColor(link.status),
                        color: 'white'
                      }}
                    />
                  </Box>
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Created by {link.createdBy} on {format(link.createdDate, 'MMM dd, yyyy')}
                </Typography>
                
                <Box mb={2}>
                  <Typography variant="caption" color="text.secondary">
                    Share URL:
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                    <TextField
                      fullWidth
                      value={link.url}
                      variant="outlined"
                      size="small"
                      InputProps={{
                        readOnly: true,
                        style: { fontSize: '0.8rem' }
                      }}
                    />
                    <Tooltip title="Copy Link">
                      <IconButton size="small" onClick={() => navigator.clipboard.writeText(link.url)}>
                        <ContentCopy fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                
                <Box mb={2}>
                  <Typography variant="caption" color="text.secondary">
                    Permissions:
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={0.5} mt={0.5}>
                    {link.permissions.map((permissionId) => {
                      const permission = availablePermissions.find(p => p.id === permissionId);
                      return (
                        <Chip 
                          key={permissionId}
                          size="small" 
                          label={permission?.name || permissionId} 
                          variant="outlined"
                        />
                      );
                    })}
                  </Box>
                </Box>
                
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Access Count: {link.accessCount}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Expires: {link.expiration ? format(link.expiration, 'MMM dd, yyyy') : 'Never'}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="caption" color="text.secondary">
                      Last Accessed: {link.lastAccessed ? format(link.lastAccessed, 'MMM dd, yyyy HH:mm') : 'Never'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
              
              <CardActions>
                <Button size="small" startIcon={<Edit />}>
                  Edit
                </Button>
                <Button 
                  size="small" 
                  startIcon={link.status === 'active' ? <Lock /> : <CheckCircle />}
                  onClick={() => handleToggleLinkStatus(link.id)}
                >
                  {link.status === 'active' ? 'Disable' : 'Enable'}
                </Button>
                <Button 
                  size="small" 
                  startIcon={<Delete />}
                  color="error"
                  onClick={() => handleDeleteLink(link.id)}
                >
                  Delete
                </Button>
              </CardActions>
            </ShareCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  const tabLabels = [
    'Create Share Link',
    'Active Shares'
  ];

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
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
          <Share color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Share Dashboard
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
          {tabLabels.map((label, index) => (
            <Tab key={label} label={label} />
          ))}
        </Tabs>
      </Box>

      <DialogContent sx={{ p: 0, height: '60vh', overflow: 'auto' }}>
        {currentTab === 0 && renderCreateShare()}
        {currentTab === 1 && renderActiveShares()}
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

export default ShareDashboardPanel;