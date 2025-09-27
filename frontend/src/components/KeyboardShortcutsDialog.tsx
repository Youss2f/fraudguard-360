/**
 * Keyboard Shortcuts Help Dialog
 * Professional help system for displaying all available keyboard shortcuts
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  styled,
} from '@mui/material';
import {
  Close,
  Keyboard,
  Dashboard,
  Search,
  FilterList,
  Assessment,
  Settings,
  Timeline
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

const ShortcutCard = styled(Card)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  border: `1px solid ${customColors.neutral[200]}`,
  borderRadius: '6px',
  '&:hover': {
    borderColor: customColors.primary[300],
    backgroundColor: customColors.primary[50],
  },
}));

const KeyboardKey = styled(Chip)(({ theme }) => ({
  backgroundColor: customColors.neutral[100],
  color: customColors.neutral[800],
  fontSize: '11px',
  height: '20px',
  fontFamily: 'Consolas, "Courier New", monospace',
  fontWeight: 600,
  minWidth: '24px',
  '& .MuiChip-label': {
    padding: '0 6px',
  },
}));

interface KeyboardShortcut {
  action: string;
  shortcut: string;
  description: string;
  category: 'navigation' | 'data' | 'views' | 'tools' | 'system';
}

interface KeyboardShortcutsDialogProps {
  open: boolean;
  onClose: () => void;
}

const KeyboardShortcutsDialog: React.FC<KeyboardShortcutsDialogProps> = ({
  open,
  onClose,
}) => {
  const [currentTab, setCurrentTab] = useState(0);

  const shortcuts: KeyboardShortcut[] = [
    // Navigation shortcuts
    { action: 'Dashboard', shortcut: 'Ctrl+1', description: 'Switch to Dashboard tab', category: 'navigation' },
    { action: 'Investigation', shortcut: 'Ctrl+2', description: 'Switch to Investigation tab', category: 'navigation' },
    { action: 'Reports', shortcut: 'Ctrl+3', description: 'Switch to Reports tab', category: 'navigation' },
    { action: 'Admin', shortcut: 'Ctrl+4', description: 'Switch to Admin tab', category: 'navigation' },
    { action: 'Global Search', shortcut: 'Ctrl+K', description: 'Open global search', category: 'navigation' },
    { action: 'Notifications', shortcut: 'Ctrl+N', description: 'View notifications panel', category: 'navigation' },
    { action: 'Profile Settings', shortcut: 'Ctrl+,', description: 'Open profile management', category: 'navigation' },
    
    // Data shortcuts
    { action: 'Refresh Data', shortcut: 'F5', description: 'Refresh current data view', category: 'data' },
    { action: 'Save Dashboard', shortcut: 'Ctrl+S', description: 'Save current dashboard state', category: 'data' },
    { action: 'Export Data', shortcut: 'Ctrl+E', description: 'Open export dialog', category: 'data' },
    { action: 'Filter Data', shortcut: 'Ctrl+F', description: 'Open filter panel', category: 'data' },
    { action: 'Print Dashboard', shortcut: 'Ctrl+P', description: 'Print current view', category: 'data' },
    
    // Views shortcuts
    { action: 'Timeline View', shortcut: 'Ctrl+T', description: 'Open timeline analysis', category: 'views' },
    { action: 'View Options', shortcut: 'Ctrl+Shift+V', description: 'Customize view settings', category: 'views' },
    { action: 'Zoom In', shortcut: 'Ctrl++', description: 'Increase zoom level', category: 'views' },
    { action: 'Zoom Out', shortcut: 'Ctrl+-', description: 'Decrease zoom level', category: 'views' },
    { action: 'Reset Zoom', shortcut: 'Ctrl+0', description: 'Reset zoom to 100%', category: 'views' },
    
    // Tools shortcuts
    { action: 'Investigation Tools', shortcut: 'Ctrl+I', description: 'Open investigation panel', category: 'tools' },
    { action: 'Generate Report', shortcut: 'Ctrl+R', description: 'Open reports panel', category: 'tools' },
    { action: 'Share Dashboard', shortcut: 'Ctrl+Shift+S', description: 'Share current dashboard', category: 'tools' },
    { action: 'Admin Panel', shortcut: 'Ctrl+Shift+A', description: 'Open system administration', category: 'tools' },
    
    // System shortcuts
    { action: 'Help', shortcut: 'F1', description: 'Show this help dialog', category: 'system' },
    { action: 'Logout', shortcut: 'Ctrl+Shift+Q', description: 'Sign out of the system', category: 'system' },
    { action: 'Full Screen', shortcut: 'F11', description: 'Toggle full screen mode', category: 'system' },
    { action: 'Quick Access', shortcut: 'Alt+Q', description: 'Focus quick access toolbar', category: 'system' },
  ];

  const categories = [
    { id: 'navigation', label: 'Navigation', icon: <Dashboard fontSize="small" /> },
    { id: 'data', label: 'Data Operations', icon: <Search fontSize="small" /> },
    { id: 'views', label: 'Views & Display', icon: <FilterList fontSize="small" /> },
    { id: 'tools', label: 'Tools & Analysis', icon: <Assessment fontSize="small" /> },
    { id: 'system', label: 'System Controls', icon: <Settings fontSize="small" /> },
  ];

  const getCurrentCategoryShortcuts = () => {
    const categoryId = categories[currentTab]?.id;
    return shortcuts.filter(shortcut => shortcut.category === categoryId);
  };

  const formatShortcut = (shortcut: string) => {
    return shortcut.split('+').map((key, index, array) => (
      <React.Fragment key={key}>
        <KeyboardKey label={key} size="small" />
        {index < array.length - 1 && (
          <Typography variant="body2" component="span" sx={{ mx: 0.5 }}>
            +
          </Typography>
        )}
      </React.Fragment>
    ));
  };

  const getCategoryIcon = (category: string) => {
    const categoryData = categories.find(cat => cat.id === category);
    return categoryData?.icon || <Keyboard fontSize="small" />;
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
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
          <Keyboard color="primary" />
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Keyboard Shortcuts
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
          {categories.map((category, index) => (
            <Tab
              key={category.id}
              label={
                <Box display="flex" alignItems="center" gap={1}>
                  {category.icon}
                  <Typography variant="body2">{category.label}</Typography>
                </Box>
              }
            />
          ))}
        </Tabs>
      </Box>

      <DialogContent sx={{ p: 3, height: '50vh', overflow: 'auto' }}>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          {categories[currentTab]?.label} Shortcuts
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Use these keyboard shortcuts to quickly access {categories[currentTab]?.label.toLowerCase()} features and improve your productivity.
        </Typography>

        <Grid container spacing={2}>
          {getCurrentCategoryShortcuts().map((shortcut, index) => (
            <Grid item xs={12} md={6} key={index}>
              <ShortcutCard>
                <CardContent sx={{ py: 2 }}>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {shortcut.action}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      {formatShortcut(shortcut.shortcut)}
                    </Box>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {shortcut.description}
                  </Typography>
                </CardContent>
              </ShortcutCard>
            </Grid>
          ))}
        </Grid>

        {getCurrentCategoryShortcuts().length === 0 && (
          <Box textAlign="center" py={4}>
            <Typography variant="body1" color="text.secondary">
              No shortcuts available for this category.
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{
        p: 2,
        backgroundColor: customColors.background.ribbon,
        borderTop: `1px solid ${customColors.neutral[200]}`
      }}>
        <Typography variant="body2" color="text.secondary" sx={{ mr: 'auto' }}>
          Press <KeyboardKey label="F1" size="small" /> anytime to open this help
        </Typography>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default KeyboardShortcutsDialog;