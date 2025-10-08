/**
 * Enhanced Professional Layout
 * Main layout component with collapsible sidebar and content area
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  ListItemIcon,
  Chip,
  styled,
  alpha,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  AccountCircle,
  Settings,
  ExitToApp,
  Brightness4,
  Brightness7,
  Fullscreen,
  FullscreenExit,
  Search,
  HelpOutline,
} from '@mui/icons-material';
import ProfessionalSidebar from './ProfessionalSidebar';

interface EnhancedLayoutProps {
  children: React.ReactNode;
  activeSection: string;
  onSectionChange: (section: string) => void;
  notificationCount?: number;
  alertCount?: number;
  userRole?: string;
  userName?: string;
  onLogout?: () => void;
  onSettings?: () => void;
  onNotifications?: () => void;
}

const LayoutContainer = styled(Box)({
  display: 'flex',
  minHeight: '100vh',
  backgroundColor: '#f5f6fa',
});

const TopAppBar = styled(AppBar, {
  shouldForwardProp: (prop) => prop !== 'sidebarCollapsed',
})<{ sidebarCollapsed: boolean }>(({ theme, sidebarCollapsed }) => ({
  zIndex: theme.zIndex.drawer + 1,
  backgroundColor: '#fff',
  color: '#1a1a2e',
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  borderBottom: `1px solid ${alpha(theme.palette.divider, 0.12)}`,
  marginLeft: sidebarCollapsed ? 60 : 280,
  width: sidebarCollapsed ? 'calc(100% - 60px)' : 'calc(100% - 280px)',
  transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
}));

const MainContent = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'sidebarCollapsed',
})<{ sidebarCollapsed: boolean }>(({ sidebarCollapsed }) => ({
  flex: 1,
  marginLeft: sidebarCollapsed ? 60 : 280,
  marginTop: 64, // AppBar height
  padding: '24px',
  transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  minHeight: 'calc(100vh - 64px)',
  backgroundColor: '#f5f6fa',
}));

const HeaderActions = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
});

const StatusChip = styled(Chip)(({ theme }) => ({
  height: 24,
  fontSize: '0.75rem',
  fontWeight: 600,
  '&.online': {
    backgroundColor: alpha('#4caf50', 0.1),
    color: '#2e7d32',
    border: '1px solid #4caf50',
  },
  '&.processing': {
    backgroundColor: alpha('#ff9800', 0.1),
    color: '#f57c00',
    border: '1px solid #ff9800',
  },
}));

const SearchContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  borderRadius: 20,
  backgroundColor: alpha('#1a1a2e', 0.05),
  border: `1px solid ${alpha('#1a1a2e', 0.1)}`,
  marginRight: 16,
  width: 300,
  display: 'flex',
  alignItems: 'center',
  '&:hover': {
    backgroundColor: alpha('#1a1a2e', 0.08),
  },
  '&:focus-within': {
    backgroundColor: '#fff',
    border: '1px solid #667eea',
    boxShadow: '0 0 0 3px rgba(102, 126, 234, 0.1)',
  },
}));

const SearchInput = styled('input')({
  border: 'none',
  outline: 'none',
  backgroundColor: 'transparent',
  padding: '8px 12px 8px 40px',
  width: '100%',
  fontSize: '0.875rem',
  fontFamily: '"Segoe UI", "Roboto", sans-serif',
  '&::placeholder': {
    color: alpha('#1a1a2e', 0.6),
  },
});

export default function EnhancedLayout({
  children,
  activeSection,
  onSectionChange,
  notificationCount = 0,
  alertCount = 0,
  userRole = 'Analyst',
  userName = 'John Doe',
  onLogout,
  onSettings,
  onNotifications,
}: EnhancedLayoutProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [sidebarCollapsed, setSidebarCollapsed] = useState(isMobile);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [systemStatus, setSystemStatus] = useState<'online' | 'processing' | 'offline'>('online');

  // Handle responsive behavior
  useEffect(() => {
    setSidebarCollapsed(isMobile);
  }, [isMobile]);

  // Handle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleSearch = (event: React.FormEvent) => {
    event.preventDefault();
    if (searchQuery.trim()) {
      // Implement search functionality
      console.log('Searching for:', searchQuery);
    }
  };

  const getPageTitle = () => {
    const titles: Record<string, string> = {
      dashboard: 'Dashboard Overview',
      'live-alerts': 'Live Alerts',
      'network-activity': 'Network Activity', 
      'fraud-detection': 'Fraud Detection',
      'system-health': 'System Health',
      'fraud-patterns': 'Fraud Patterns',
      'risk-scoring': 'Risk Scoring',
      'behavioral-analysis': 'Behavioral Analysis',
      'trend-analysis': 'Trend Analysis',
      'geographical-analysis': 'Geographic Analysis',
      'case-management': 'Case Management',
      'call-analysis': 'Call Analysis',
      'user-profiling': 'User Profiling',
      'network-visualization': 'Network Visualization',
      'timeline-analysis': 'Timeline Analysis',
      'fraud-reports': 'Fraud Reports',
      'compliance-reports': 'Compliance Reports',
      'custom-reports': 'Custom Reports',
      'data-export': 'Data Export',
      'scheduled-reports': 'Scheduled Reports',
      'user-management': 'User Management',
      'system-settings': 'System Settings',
      'audit-logs': 'Audit Logs',
      notifications: 'Notifications',
      integrations: 'Integrations',
    };
    return titles[activeSection] || 'FraudGuard 360';
  };

  return (
    <LayoutContainer>
      {/* Professional Sidebar */}
      <ProfessionalSidebar
        collapsed={sidebarCollapsed}
        onCollapse={setSidebarCollapsed}
        activeSection={activeSection}
        onSectionChange={onSectionChange}
        notificationCount={notificationCount}
        alertCount={alertCount}
        userRole={userRole}
      />

      {/* Top AppBar */}
      <TopAppBar position="fixed" sidebarCollapsed={sidebarCollapsed}>
        <Toolbar sx={{ minHeight: '64px !important' }}>
          {/* Mobile menu button */}
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}

          {/* Page title and status */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#1a1a2e' }}>
              {getPageTitle()}
            </Typography>
            <StatusChip 
              label={systemStatus.charAt(0).toUpperCase() + systemStatus.slice(1)} 
              className={systemStatus}
              size="small"
            />
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          {/* Search */}
          <SearchContainer>
            <Search 
              sx={{ 
                position: 'absolute', 
                left: 12, 
                color: alpha('#1a1a2e', 0.6),
                fontSize: 20,
              }} 
            />
            <form onSubmit={handleSearch} style={{ width: '100%' }}>
              <SearchInput
                placeholder="Search users, cases, patterns..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </form>
          </SearchContainer>

          {/* Header Actions */}
          <HeaderActions>
            {/* Fullscreen toggle */}
            <IconButton color="inherit" onClick={toggleFullscreen}>
              {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>

            {/* Help */}
            <IconButton color="inherit">
              <HelpOutline />
            </IconButton>

            {/* Notifications */}
            <IconButton color="inherit" onClick={onNotifications}>
              <Badge badgeContent={notificationCount} color="error" max={99}>
                <Notifications />
              </Badge>
            </IconButton>

            {/* User Menu */}
            <IconButton
              color="inherit"
              onClick={handleUserMenuOpen}
              sx={{ ml: 1 }}
            >
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  bgcolor: '#667eea',
                  fontSize: '0.875rem',
                }}
              >
                {userName.charAt(0)}
              </Avatar>
            </IconButton>
          </HeaderActions>
        </Toolbar>
      </TopAppBar>

      {/* User Menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleUserMenuClose}
        onClick={handleUserMenuClose}
        PaperProps={{
          elevation: 8,
          sx: {
            mt: 1.5,
            minWidth: 200,
            '& .MuiAvatar-root': {
              width: 24,
              height: 24,
              ml: -0.5,
              mr: 1,
            },
          },
        }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            {userName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {userRole}
          </Typography>
        </Box>
        <Divider />
        <MenuItem onClick={onSettings}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        <MenuItem>
          <ListItemIcon>
            <AccountCircle fontSize="small" />
          </ListItemIcon>
          Profile
        </MenuItem>
        <Divider />
        <MenuItem onClick={onLogout}>
          <ListItemIcon>
            <ExitToApp fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>

      {/* Main Content Area */}
      <MainContent sidebarCollapsed={sidebarCollapsed}>
        {children}
      </MainContent>
    </LayoutContainer>
  );
}