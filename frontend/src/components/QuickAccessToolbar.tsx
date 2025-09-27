/**
 * Top Quick-Access Toolbar
 * Persistent horizontal bar for global, high-frequency actions
 */

import React, { useState } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Tooltip,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Typography,
  styled,
} from '@mui/material';
import {
  Save,
  Undo,
  Redo,
  Search,
  Person,
  Help,
  Settings,
  Logout,
  Notifications,
  MoreHoriz,
  Refresh,
  FilterList,
  Timeline,
  Assessment,
  GetApp,
  Share,
  Print,
  FindInPage,
} from '@mui/icons-material';
import { technicalColorSystem } from '../theme/technicalTheme';

const ToolbarContainer = styled(Box)({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  height: '40px',
  backgroundColor: technicalColorSystem.background.secondary,
  borderBottom: `1px solid ${technicalColorSystem.background.border}`,
  display: 'flex',
  alignItems: 'center',
  padding: '0 16px',
  zIndex: 1200,
  gap: '8px',
});

const SearchField = styled(TextField)({
  '& .MuiOutlinedInput-root': {
    height: '28px',
    fontSize: '13px',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: '3px',
    '& fieldset': {
      borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    '&:hover fieldset': {
      borderColor: technicalColorSystem.accent.primary,
    },
    '&.Mui-focused fieldset': {
      borderColor: technicalColorSystem.accent.primary,
      borderWidth: '1px',
    },
  },
  '& .MuiOutlinedInput-input': {
    padding: '4px 8px',
    color: technicalColorSystem.text.primary,
    '&::placeholder': {
      color: technicalColorSystem.text.muted,
      opacity: 0.7,
    },
  },
});

const ToolbarButton = styled(IconButton)({
  width: '28px',
  height: '28px',
  color: '#FFFFFF', // Explicit white for better contrast
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    color: '#FFFFFF',
  },
  '&:disabled': {
    color: 'rgba(255, 255, 255, 0.4)',
  },
  '& .MuiSvgIcon-root': {
    fontSize: '16px',
  },
});

const AppActionButton = styled(IconButton)({
  width: '28px',
  height: '28px',
  color: '#FFFFFF', // Explicit white for better contrast
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    color: '#FFFFFF',
  },
  '&:disabled': {
    color: 'rgba(255, 255, 255, 0.4)',
  },
  '& .MuiSvgIcon-root': {
    fontSize: '16px',
  },
});

const UserSection = styled(Box)({
  marginLeft: 'auto',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
});

interface QuickAccessToolbarProps {
  // Standard actions
  onSave?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onSearch?: (query: string) => void;
  onHelp?: () => void;
  onSettings?: () => void;
  onLogout?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  userName?: string;
  notificationCount?: number;
  // Application-specific actions (preserved from original)
  onActionClick?: (action: string) => void;
  customActions?: Array<{
    id: string;
    icon: React.ReactNode;
    tooltip: string;
    action: () => void;
  }>;
}

const QuickAccessToolbar: React.FC<QuickAccessToolbarProps> = ({
  onSave,
  onUndo,
  onRedo,
  onSearch,
  onHelp,
  onSettings,
  onLogout,
  canUndo = false,
  canRedo = false,
  userName = 'User',
  notificationCount = 0,
  onActionClick,
  customActions = [],
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [customMenuAnchor, setCustomMenuAnchor] = useState<null | HTMLElement>(null);

  // Keyboard shortcuts for application actions
  React.useEffect(() => {
    if (!onActionClick) return;
    
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey) {
        switch (event.key.toLowerCase()) {
          case 'f':
            event.preventDefault();
            onActionClick('filter');
            break;
          case 't':
            event.preventDefault();
            onActionClick('timeline');
            break;
          case 'r':
            event.preventDefault();
            onActionClick('reports');
            break;
          case 'e':
            event.preventDefault();
            onActionClick('export');
            break;
        }
      } else if (event.key === 'F5') {
        event.preventDefault();
        onActionClick('refresh');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onActionClick]);

  const handleSearch = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && onSearch) {
      onSearch(searchQuery);
    }
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleCustomMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setCustomMenuAnchor(event.currentTarget);
  };

  const handleCustomMenuClose = () => {
    setCustomMenuAnchor(null);
  };

  return (
    <ToolbarContainer title="FraudGuard 360 - Quick Access Toolbar">
      {/* Core Actions */}
      <Tooltip title="Save (Ctrl+S)">
        <ToolbarButton onClick={onSave} size="small">
          <Save fontSize="small" />
        </ToolbarButton>
      </Tooltip>

      <Tooltip title="Undo (Ctrl+Z)">
        <ToolbarButton onClick={onUndo} disabled={!canUndo} size="small">
          <Undo fontSize="small" />
        </ToolbarButton>
      </Tooltip>

      <Tooltip title="Redo (Ctrl+Y)">
        <ToolbarButton onClick={onRedo} disabled={!canRedo} size="small">
          <Redo fontSize="small" />
        </ToolbarButton>
      </Tooltip>

      <Divider 
        orientation="vertical" 
        flexItem 
        sx={{ 
          backgroundColor: 'rgba(255, 255, 255, 0.2)',
          margin: '0 8px',
          height: '20px',
          alignSelf: 'center'
        }} 
      />

      {/* Application-Specific Actions */}
      {onActionClick && (
        <>
          <Tooltip title="Refresh Data (F5)">
            <AppActionButton 
              onClick={() => {
                console.log('Refresh button clicked');
                onActionClick('refresh');
              }} 
              size="small"
            >
              <Refresh fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Tooltip title="Filter Data (Ctrl+F)">
            <AppActionButton onClick={() => onActionClick('filter')} size="small">
              <FilterList fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Tooltip title="Timeline View (Ctrl+T)">
            <AppActionButton onClick={() => onActionClick('timeline')} size="small">
              <Timeline fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Tooltip title="Generate Report (Ctrl+R)">
            <AppActionButton onClick={() => onActionClick('reports')} size="small">
              <Assessment fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Tooltip title="Export Data (Ctrl+E)">
            <AppActionButton onClick={() => onActionClick('export')} size="small">
              <GetApp fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Tooltip title="Share Dashboard">
            <AppActionButton onClick={() => onActionClick('share')} size="small">
              <Share fontSize="small" />
            </AppActionButton>
          </Tooltip>

          <Divider 
            orientation="vertical" 
            flexItem 
            sx={{ 
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              margin: '0 8px',
              height: '20px',
              alignSelf: 'center'
            }} 
          />
        </>
      )}

      {/* Global Search */}
      <SearchField
        size="small"
        placeholder="Global search... (Ctrl+K)"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        onKeyDown={handleSearch}
        InputProps={{
          startAdornment: (
            <Search 
              fontSize="small" 
              sx={{ 
                color: technicalColorSystem.text.muted, 
                marginRight: '4px' 
              }} 
            />
          ),
        }}
        sx={{ width: '300px' }}
      />

      {/* Custom Actions */}
      {customActions.length > 0 && (
        <>
          <Divider 
            orientation="vertical" 
            flexItem 
            sx={{ 
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              margin: '0 8px',
              height: '20px',
              alignSelf: 'center'
            }} 
          />
          
          {customActions.slice(0, 3).map((action) => (
            <Tooltip key={action.id} title={action.tooltip}>
              <ToolbarButton onClick={action.action} size="small">
                {action.icon}
              </ToolbarButton>
            </Tooltip>
          ))}
          
          {customActions.length > 3 && (
            <Tooltip title="More actions">
              <ToolbarButton onClick={handleCustomMenuOpen} size="small">
                <MoreHoriz fontSize="small" />
              </ToolbarButton>
            </Tooltip>
          )}
        </>
      )}

      {/* User Section */}
      <UserSection>
        <Tooltip title="Help & Documentation (F1)">
          <ToolbarButton onClick={onHelp} size="small">
            <Help fontSize="small" />
          </ToolbarButton>
        </Tooltip>

        <Tooltip title={`Notifications (${notificationCount})`}>
          <ToolbarButton size="small">
            <Box position="relative">
              <Notifications fontSize="small" />
              {notificationCount > 0 && (
                <Box
                  sx={{
                    position: "absolute",
                    top: "-2px",
                    right: "-2px",
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    backgroundColor: technicalColorSystem.status.error,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "8px",
                    color: "white",
                    fontWeight: "bold",
                  }}
                >
                  {notificationCount > 9 ? '9+' : notificationCount}
                </Box>
              )}
            </Box>
          </ToolbarButton>
        </Tooltip>

        <Tooltip title={`${userName} - User menu`}>
          <ToolbarButton onClick={handleUserMenuOpen} size="small">
            <Avatar 
              sx={{ 
                width: 24, 
                height: 24, 
                fontSize: '12px',
                backgroundColor: technicalColorSystem.accent.primary,
                color: 'white'
              }}
            >
              {userName.charAt(0).toUpperCase()}
            </Avatar>
          </ToolbarButton>
        </Tooltip>
      </UserSection>

      {/* User Menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleUserMenuClose}
        PaperProps={{
          sx: {
            mt: 1,
            minWidth: '180px',
            border: `1px solid ${technicalColorSystem.background.border}`,
          }
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {userName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            System Administrator
          </Typography>
        </Box>
        <Divider />
        <MenuItem onClick={() => { handleUserMenuClose(); onSettings?.(); }}>
          <Settings fontSize="small" sx={{ mr: 1 }} />
          Settings
        </MenuItem>
        <MenuItem onClick={() => { handleUserMenuClose(); onHelp?.(); }}>
          <Help fontSize="small" sx={{ mr: 1 }} />
          Help & Support
        </MenuItem>
        <Divider />
        <MenuItem 
          onClick={() => { handleUserMenuClose(); onLogout?.(); }}
          sx={{ color: technicalColorSystem.status.error }}
        >
          <Logout fontSize="small" sx={{ mr: 1 }} />
          Sign Out
        </MenuItem>
      </Menu>

      {/* Custom Actions Menu */}
      <Menu
        anchorEl={customMenuAnchor}
        open={Boolean(customMenuAnchor)}
        onClose={handleCustomMenuClose}
        PaperProps={{
          sx: {
            mt: 1,
            minWidth: '160px',
            border: `1px solid ${technicalColorSystem.background.border}`,
          }
        }}
      >
        {customActions.slice(3).map((action) => (
          <MenuItem 
            key={action.id} 
            onClick={() => { 
              handleCustomMenuClose(); 
              action.action(); 
            }}
          >
            {action.icon}
            <Box component="span" sx={{ ml: 1 }}>
              {action.tooltip}
            </Box>
          </MenuItem>
        ))}
      </Menu>
    </ToolbarContainer>
  );
};

export default QuickAccessToolbar;