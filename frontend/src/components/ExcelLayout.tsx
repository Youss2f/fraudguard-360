/**
 * Excel-Style Professional Layout
 * Modern, clean layout inspired by Microsoft Excel's interface
 */

import React, { useState, memo, lazy } from 'react';
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
  Paper,
  Container,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  AccountCircle,
  Settings,
  ExitToApp,
  Search,
  HelpOutline,
  Refresh,
  Fullscreen,
  MoreVert,
  Home,
  Assessment,
  Security,
  Report,
  MonitorHeart,
} from '@mui/icons-material';
import { excelColors } from '../theme/excelTheme';

const RibbonNavigation = lazy(() => import('./RibbonNavigation'));

interface ExcelLayoutProps {
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
  onActionClick: (action: string) => void;
  currentTab: string;
}

// Main layout container with Excel-style structure
const LayoutRoot = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  minHeight: '100vh',
  backgroundColor: excelColors.background.default,
  
  // Responsive adjustments
  [theme.breakpoints.down('md')]: {
    minHeight: '100dvh', // Use dynamic viewport height on mobile
  },
}));

// Excel-style ribbon/toolbar
const ExcelRibbon = styled(AppBar)(({ theme }) => ({
  backgroundColor: excelColors.background.ribbon,
  color: excelColors.text.primary,
  boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
  borderBottom: `1px solid ${excelColors.background.border}`,
  position: 'sticky',
  top: 0,
  zIndex: 1200,
  
  // Responsive toolbar height
  [theme.breakpoints.down('sm')]: {
    minHeight: 56, // Shorter on mobile
  },
}));

// Navigation tabs (Excel ribbon tabs)
const NavigationTabs = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  backgroundColor: excelColors.background.paper,
  borderBottom: `1px solid ${excelColors.background.border}`,
  padding: '0 24px',
  minHeight: 48,
  overflowX: 'auto', // Allow horizontal scrolling on small screens
  
  // Mobile responsive
  [theme.breakpoints.down('md')]: {
    padding: '0 16px',
    minHeight: 40,
    
    // Hide scrollbar but keep functionality
    '&::-webkit-scrollbar': {
      display: 'none',
    },
    scrollbarWidth: 'none',
  },
  
  [theme.breakpoints.down('sm')]: {
    padding: '0 12px',
    gap: '4px',
  },
}));

const NavTab = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'active',
})<{ active?: boolean }>(({ active, theme }) => ({
  padding: '8px 16px',
  cursor: 'pointer',
  borderRadius: '4px 4px 0 0',
  fontSize: '0.875rem',
  fontWeight: 500,
  color: active ? excelColors.primary.main : excelColors.text.secondary,
  backgroundColor: active ? excelColors.background.paper : 'transparent',
  border: active ? `1px solid ${excelColors.background.border}` : '1px solid transparent',
  borderBottom: active ? 'none' : '1px solid transparent',
  transition: 'all 0.2s ease',
  whiteSpace: 'nowrap',
  flexShrink: 0,
  
  '&:hover': {
    backgroundColor: active ? excelColors.background.paper : excelColors.background.hover,
    color: excelColors.primary.main,
  },
  
  // Mobile responsive
  [theme.breakpoints.down('md')]: {
    padding: '6px 12px',
    fontSize: '0.8rem',
  },
  
  [theme.breakpoints.down('sm')]: {
    padding: '4px 8px',
    fontSize: '0.75rem',
    minWidth: 'auto',
  },
}));

// Main content area
const ContentArea = styled(Box)(({ theme }) => ({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: excelColors.background.default,
  
  // Responsive adjustments
  [theme.breakpoints.down('md')]: {
    minHeight: 'calc(100vh - 120px)', // Account for smaller toolbar on mobile
  },
}));

const MainContent = styled(Container)(({ theme }) => ({
  flex: 1,
  padding: '24px',
  maxWidth: '100% !important',
  
  // Responsive padding
  [theme.breakpoints.down('lg')]: {
    padding: '20px',
  },
  
  [theme.breakpoints.down('md')]: {
    padding: '16px',
  },
  
  [theme.breakpoints.down('sm')]: {
    padding: '12px',
  },
}));

// Quick access toolbar (Excel-style)
const QuickAccessBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '0 16px',
  
  // Mobile responsive
  [theme.breakpoints.down('md')]: {
    padding: '0 12px',
    gap: 4,
  },
  
  [theme.breakpoints.down('sm')]: {
    padding: '0 8px',
    gap: 2,
    // Hide some buttons on mobile
    '& .desktop-only': {
      display: 'none',
    },
  },
}));

const QuickAccessButton = styled(IconButton)(({ theme }) => ({
  width: 32,
  height: 32,
  color: excelColors.text.secondary,
  '&:hover': {
    backgroundColor: excelColors.background.hover,
    color: excelColors.primary.main,
  },
  
  // Mobile responsive
  [theme.breakpoints.down('sm')]: {
    width: 28,
    height: 28,
    '& .MuiSvgIcon-root': {
      fontSize: '1rem',
    },
  },
}));

// Status bar (Excel-style bottom bar)
const StatusBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  backgroundColor: excelColors.background.ribbon,
  borderTop: `1px solid ${excelColors.background.border}`,
  padding: '4px 16px',
  minHeight: 24,
  fontSize: '0.75rem',
  color: excelColors.text.secondary,
  
  // Mobile responsive
  [theme.breakpoints.down('md')]: {
    padding: '4px 12px',
    fontSize: '0.7rem',
  },
  
  [theme.breakpoints.down('sm')]: {
    padding: '2px 8px',
    minHeight: 20,
    fontSize: '0.65rem',
    // Stack on mobile if needed
    flexWrap: 'wrap',
    gap: '4px',
  },
}));

// Navigation sections configuration
const navigationSections = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'monitoring', label: 'Real-time Monitoring', icon: MonitorHeart },
  { id: 'investigations', label: 'Fraud Detection', icon: Security },
  { id: 'analytics', label: 'Analytics', icon: Assessment },
  { id: 'reports', label: 'Reports', icon: Report },
  { id: 'settings', label: 'Settings', icon: Settings },
];

const ExcelLayout: React.FC<ExcelLayoutProps> = ({
  children,
  activeSection,
  onSectionChange,
  notificationCount = 0,
  alertCount = 0,
  userRole = 'Administrator',
  userName = 'John Doe',
  onLogout,
  onSettings,
  onNotifications,
  onActionClick,
  currentTab,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  const handleUserMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setUserMenuAnchor(null);
  };

  const handleLogout = () => {
    handleUserMenuClose();
    onLogout?.();
  };

  const handleSettings = () => {
    handleUserMenuClose();
    onSettings?.();
  };

  return (
    <LayoutRoot>
      {/* Excel-style ribbon header */}
      <RibbonNavigation
        currentTab={currentTab}
        onTabChange={onSectionChange}
        onActionClick={onActionClick}
        notificationCount={notificationCount}
        userName={userName}
      />

      {/* Main content area */}
      <ContentArea role="main" aria-label="Main content">
        <MainContent 
          id={`panel-${activeSection}`}
          role="tabpanel"
          aria-labelledby={`tab-${activeSection}`}
        >
          {children}
        </MainContent>
      </ContentArea>

      {/* Excel-style status bar */}
      <StatusBar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="caption">
            Ready • Last updated: {new Date().toLocaleTimeString()}
          </Typography>
          <Typography variant="caption">
            Connected to FraudGuard Analytics Engine
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="caption">
            Processing: {Math.floor(Math.random() * 1000)} transactions/min
          </Typography>
          <Typography variant="caption">
            Memory: {Math.floor(Math.random() * 40 + 60)}%
          </Typography>
        </Box>
      </StatusBar>
    </LayoutRoot>
  );
};

export default memo(ExcelLayout);