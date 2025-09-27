/**
 * Professional Windows-style Ribbon Navigation
 * Inspired by Microsoft Office 2010 interface
 */

import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Tabs,
  Tab,
  Button,
  ButtonGroup,
  IconButton,
  Badge,
  Divider,
  Tooltip,
  Menu,
  MenuItem,
  Avatar,
  Paper,
  styled,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Search as SearchIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Notifications as NotificationIcon,
  AccountCircle as AccountIcon,
  Refresh as RefreshIcon,
  GetApp as ExportIcon,
  Print as PrintIcon,
  Share as ShareIcon,
  FilterList as FilterIcon,
  ViewModule as ViewIcon,
  Timeline as TimelineIcon,
  TravelExplore as InvestigateIcon,
  Report as ReportIcon,
  AdminPanelSettings as AdminIcon,
} from '@mui/icons-material';
import { customColors } from '../theme/enterpriseTheme';

// Styled components for professional ribbon appearance
const RibbonContainer = styled(Paper)(({ theme }) => ({
  backgroundColor: customColors.background.ribbon,
  borderBottom: `2px solid ${customColors.primary[500]}`,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
  position: 'sticky',
  top: 0,
  zIndex: 1000,
}));

const TitleBar = styled(Box)(({ theme }) => ({
  backgroundColor: customColors.primary[600],
  color: 'white',
  padding: '8px 24px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  boxShadow: 'inset 0 -1px 0 rgba(255, 255, 255, 0.1)',
}));

const RibbonTabs = styled(Tabs)(({ theme }) => ({
  minHeight: '32px',
  '& .MuiTabs-flexContainer': {
    height: '32px',
  },
  '& .MuiTab-root': {
    minHeight: '32px',
    fontSize: '0.75rem',
    fontWeight: 500,
    padding: '6px 16px',
    textTransform: 'none',
    color: customColors.neutral[700],
    '&.Mui-selected': {
      color: customColors.primary[700],
      backgroundColor: customColors.background.paper,
      borderRadius: '4px 4px 0 0',
    },
  },
  '& .MuiTabs-indicator': {
    display: 'none',
  },
}));

const RibbonContent = styled(Box)(({ theme }) => ({
  backgroundColor: customColors.background.paper,
  padding: '12px 24px',
  borderBottom: `1px solid ${customColors.neutral[200]}`,
  display: 'flex',
  gap: '24px',
  alignItems: 'center',
  minHeight: '80px',
}));

const RibbonGroup = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: '8px',
  padding: '8px 12px',
  borderRight: `1px solid ${customColors.neutral[200]}`,
  '&:last-child': {
    borderRight: 'none',
  },
}));

const GroupLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.7rem',
  fontWeight: 600,
  color: customColors.neutral[600],
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  marginTop: '4px',
}));

const RibbonButton = styled(Button)(({ theme }) => ({
  minWidth: '60px',
  height: '50px',
  flexDirection: 'column',
  gap: '4px',
  padding: '8px',
  fontSize: '0.7rem',
  textTransform: 'none',
  borderRadius: '4px',
  color: customColors.neutral[700],
  '&:hover': {
    backgroundColor: customColors.primary[50],
    color: customColors.primary[700],
  },
  '& .MuiButton-startIcon': {
    margin: 0,
    fontSize: '1.5rem',
  },
}));

interface RibbonNavigationProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  onActionClick: (action: string) => void;
  notificationCount?: number;
  userName?: string;
}

const RibbonNavigation: React.FC<RibbonNavigationProps> = ({
  currentTab,
  onTabChange,
  onActionClick,
  notificationCount = 0,
  userName = 'Analyst',
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  const handleProfileClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleProfileClose = () => {
    setAnchorEl(null);
  };

  const ribbonGroups = {
    dashboard: [
      {
        group: 'View',
        actions: [
          { id: 'refresh', label: 'Refresh', icon: <RefreshIcon /> },
          { id: 'filter', label: 'Filter', icon: <FilterIcon /> },
          { id: 'view', label: 'View', icon: <ViewIcon /> },
        ],
      },
      {
        group: 'Analytics',
        actions: [
          { id: 'timeline', label: 'Timeline', icon: <TimelineIcon /> },
          { id: 'reports', label: 'Reports', icon: <ReportIcon /> },
          { id: 'export', label: 'Export', icon: <ExportIcon /> },
        ],
      },
      {
        group: 'Actions',
        actions: [
          { id: 'investigate', label: 'Investigate', icon: <InvestigateIcon /> },
          { id: 'share', label: 'Share', icon: <ShareIcon /> },
          { id: 'print', label: 'Print', icon: <PrintIcon /> },
        ],
      },
    ],
    investigation: [
      {
        group: 'Analysis',
        actions: [
          { id: 'expand', label: 'Expand', icon: <SearchIcon /> },
          { id: 'timeline', label: 'Timeline', icon: <TimelineIcon /> },
          { id: 'patterns', label: 'Patterns', icon: <AssessmentIcon /> },
        ],
      },
      {
        group: 'Case Management',
        actions: [
          { id: 'assign', label: 'Assign', icon: <AccountIcon /> },
          { id: 'escalate', label: 'Escalate', icon: <ReportIcon /> },
          { id: 'close', label: 'Close', icon: <SecurityIcon /> },
        ],
      },
      {
        group: 'Export',
        actions: [
          { id: 'export-case', label: 'Export', icon: <ExportIcon /> },
          { id: 'report', label: 'Report', icon: <PrintIcon /> },
          { id: 'share-case', label: 'Share', icon: <ShareIcon /> },
        ],
      },
    ],
    reports: [
      {
        group: 'Generate',
        actions: [
          { id: 'daily', label: 'Daily', icon: <AssessmentIcon /> },
          { id: 'weekly', label: 'Weekly', icon: <TimelineIcon /> },
          { id: 'custom', label: 'Custom', icon: <ReportIcon /> },
        ],
      },
      {
        group: 'Export',
        actions: [
          { id: 'pdf', label: 'PDF', icon: <PrintIcon /> },
          { id: 'excel', label: 'Excel', icon: <ExportIcon /> },
          { id: 'email', label: 'Email', icon: <ShareIcon /> },
        ],
      },
    ],
    admin: [
      {
        group: 'Users',
        actions: [
          { id: 'manage-users', label: 'Manage', icon: <AccountIcon /> },
          { id: 'roles', label: 'Roles', icon: <AdminIcon /> },
          { id: 'permissions', label: 'Permissions', icon: <SecurityIcon /> },
        ],
      },
      {
        group: 'System',
        actions: [
          { id: 'settings', label: 'Settings', icon: <SettingsIcon /> },
          { id: 'monitoring', label: 'Monitor', icon: <AssessmentIcon /> },
          { id: 'logs', label: 'Logs', icon: <ReportIcon /> },
        ],
      },
    ],
  };

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'investigation', label: 'Investigation', icon: <InvestigateIcon /> },
    { id: 'reports', label: 'Reports', icon: <ReportIcon /> },
    { id: 'admin', label: 'Administration', icon: <AdminIcon /> },
  ];

  const currentGroups = ribbonGroups[currentTab as keyof typeof ribbonGroups] || ribbonGroups.dashboard;

  return (
    <RibbonContainer elevation={0}>
      {/* Title Bar */}
      <TitleBar>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
            FraudGuard 360 - Professional Fraud Detection Platform
          </Typography>
        </Box>
        
        <Box display="flex" alignItems="center" gap={1}>
          <Tooltip title="Notifications">
            <IconButton color="inherit" onClick={() => onActionClick('notifications')}>
              <Badge badgeContent={notificationCount} color="error">
                <NotificationIcon />
              </Badge>
            </IconButton>
          </Tooltip>
          
          <Tooltip title="User Profile">
            <IconButton color="inherit" onClick={handleProfileClick}>
              <Avatar sx={{ width: 28, height: 28, bgcolor: 'rgba(255,255,255,0.2)' }}>
                {userName.charAt(0)}
              </Avatar>
            </IconButton>
          </Tooltip>
          
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleProfileClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          >
            <MenuItem onClick={() => { onActionClick('profile'); handleProfileClose(); }}>
              Profile Settings
            </MenuItem>
            <MenuItem onClick={() => { onActionClick('logout'); handleProfileClose(); }}>
              Logout
            </MenuItem>
          </Menu>
        </Box>
      </TitleBar>

      {/* Tab Navigation */}
      <Box sx={{ backgroundColor: customColors.background.ribbon, px: 3 }}>
        <RibbonTabs
          value={currentTab}
          onChange={(_, newValue) => onTabChange(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          {tabs.map((tab) => (
            <Tab
              key={tab.id}
              value={tab.id}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
            />
          ))}
        </RibbonTabs>
      </Box>

      {/* Ribbon Content */}
      <RibbonContent>
        {currentGroups.map((group, groupIndex) => (
          <RibbonGroup key={groupIndex}>
            <Box display="flex" gap={1}>
              {group.actions.map((action) => (
                <RibbonButton
                  key={action.id}
                  startIcon={action.icon}
                  onClick={() => onActionClick(action.id)}
                  size="small"
                >
                  {action.label}
                </RibbonButton>
              ))}
            </Box>
            <GroupLabel>{group.group}</GroupLabel>
          </RibbonGroup>
        ))}
      </RibbonContent>
    </RibbonContainer>
  );
};

export default RibbonNavigation;