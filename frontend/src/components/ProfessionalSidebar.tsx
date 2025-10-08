/**
 * Professional Collapsible Sidebar
 * Enhanced navigation panel with modern design and complete functionality
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  IconButton,
  Tooltip,
  Typography,
  Divider,
  Badge,
  Avatar,
  Chip,
  styled,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Dashboard,
  Security,
  Timeline,
  Assessment,
  Settings,
  Help,
  ChevronLeft,
  ChevronRight,
  ExpandLess,
  ExpandMore,
  Circle,
  AccountTree,
  Visibility,
  Warning,
  Search,
  Storage,
  NetworkCheck,
  TrendingUp,
  Groups,
  Description,
  Notifications,
  ManageAccounts,
  Analytics,
  ReportProblem,
  Shield,
  Speed,
  Assignment,
  Person,
  LocationOn,
  QueryStats,
  Tune,
  History,
  WorkOutline,
  SignalCellular4Bar,
  BarChart,
  PieChart,
  ShowChart,
  TableChart,
  Map,
  CloudUpload,
  Download,
  Share,
  Print,
  FilterList,
  Schedule,
} from '@mui/icons-material';

interface SidebarProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
  activeSection: string;
  onSectionChange: (section: string) => void;
  notificationCount?: number;
  alertCount?: number;
  userRole?: string;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ElementType;
  badge?: number;
  children?: NavigationItem[];
  onClick?: () => void;
  disabled?: boolean;
}

const SidebarContainer = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed: boolean }>(({ theme, collapsed }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  bottom: 0,
  width: collapsed ? 60 : 280,
  backgroundColor: theme.palette.mode === 'dark' 
    ? theme.palette.grey[900] 
    : '#1a1a2e',
  borderRight: `1px solid ${alpha(theme.palette.divider, 0.12)}`,
  transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  zIndex: 1200,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '2px 0 10px rgba(0,0,0,0.1)',
  overflow: 'hidden',
}));

const SidebarHeader = styled(Box)(({ theme }) => ({
  height: 64,
  padding: '12px 16px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  borderBottom: `1px solid ${alpha('#fff', 0.1)}`,
  backgroundColor: alpha('#000', 0.2),
}));

const LogoContainer = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed: boolean }>(({ collapsed }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  overflow: 'hidden',
  transition: 'opacity 0.2s ease',
  opacity: collapsed ? 0 : 1,
}));

const CollapseButton = styled(IconButton)(({ theme }) => ({
  width: 36,
  height: 36,
  color: '#fff',
  '&:hover': {
    backgroundColor: alpha('#fff', 0.1),
  },
}));

const NavigationList = styled(List)(({ theme }) => ({
  flex: 1,
  padding: '8px 0',
  overflow: 'auto',
  '&::-webkit-scrollbar': {
    width: 6,
  },
  '&::-webkit-scrollbar-track': {
    background: 'transparent',
  },
  '&::-webkit-scrollbar-thumb': {
    background: alpha('#fff', 0.2),
    borderRadius: 3,
  },
}));

const StyledListItem = styled(ListItem)({
  padding: 0,
  margin: '2px 8px',
});

const StyledListItemButton = styled(ListItemButton, {
  shouldForwardProp: (prop) => prop !== 'level' && prop !== 'collapsed' && prop !== 'active',
})<{ level?: number; collapsed?: boolean; active?: boolean }>(({ theme, level = 0, collapsed = false, active = false }) => ({
  minHeight: 44,
  paddingLeft: collapsed ? 16 : 16 + level * 20,
  paddingRight: 16,
  borderRadius: 8,
  margin: '2px 0',
  color: active ? '#fff' : alpha('#fff', 0.8),
  backgroundColor: active ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'transparent',
  transition: 'all 0.2s ease',
  '&:hover': {
    backgroundColor: active 
      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' 
      : alpha('#fff', 0.08),
    color: '#fff',
    transform: 'translateX(2px)',
  },
  '&::before': {
    content: '""',
    position: 'absolute',
    left: 0,
    top: '50%',
    transform: 'translateY(-50%)',
    width: 3,
    height: active ? 20 : 0,
    backgroundColor: '#667eea',
    borderRadius: '0 2px 2px 0',
    transition: 'height 0.2s ease',
  },
}));

const StyledListItemIcon = styled(ListItemIcon, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed }) => ({
  minWidth: collapsed ? 'auto' : 40,
  color: 'inherit',
  justifyContent: 'center',
}));

const StyledListItemText = styled(ListItemText, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed }) => ({
  opacity: collapsed ? 0 : 1,
  transition: 'opacity 0.2s ease',
  '& .MuiListItemText-primary': {
    fontSize: '0.875rem',
    fontWeight: 500,
    fontFamily: '"Segoe UI", "Roboto", sans-serif',
  },
}));

const SectionDivider = styled(Divider)(({ theme }) => ({
  margin: '12px 16px',
  backgroundColor: alpha('#fff', 0.1),
}));

const SectionTitle = styled(Typography, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed }) => ({
  fontSize: '0.75rem',
  fontWeight: 600,
  color: alpha('#fff', 0.6),
  textTransform: 'uppercase',
  letterSpacing: 1,
  padding: '16px 16px 8px',
  opacity: collapsed ? 0 : 1,
  transition: 'opacity 0.2s ease',
}));

const UserSection = styled(Box)(({ theme }) => ({
  padding: '16px',
  borderTop: `1px solid ${alpha('#fff', 0.1)}`,
  backgroundColor: alpha('#000', 0.2),
}));

const UserInfo = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  '& .user-details': {
    opacity: collapsed ? 0 : 1,
    transition: 'opacity 0.2s ease',
  },
}));

export default function ProfessionalSidebar({
  collapsed,
  onCollapse,
  activeSection,
  onSectionChange,
  notificationCount = 0,
  alertCount = 0,
  userRole = 'Analyst',
}: SidebarProps) {
  const theme = useTheme();
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    monitoring: true,
    analytics: false,
    investigations: false,
    reports: false,
    administration: false,
  });

  const handleSectionToggle = (sectionId: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  const navigationItems: NavigationItem[] = [
    // Main Dashboard
    {
      id: 'dashboard',
      label: 'Dashboard Overview',
      icon: Dashboard,
    },
    
    // Real-time Monitoring Section
    {
      id: 'monitoring',
      label: 'Real-time Monitoring',
      icon: Speed,
      children: [
        { id: 'live-alerts', label: 'Live Alerts', icon: Warning, badge: alertCount },
        { id: 'network-activity', label: 'Network Activity', icon: NetworkCheck },
        { id: 'fraud-detection', label: 'Fraud Detection', icon: Security },
        { id: 'system-health', label: 'System Health', icon: SignalCellular4Bar },
      ],
    },

    // Analytics & Insights
    {
      id: 'analytics',
      label: 'Analytics & Insights',
      icon: Analytics,
      children: [
        { id: 'fraud-patterns', label: 'Fraud Patterns', icon: ShowChart },
        { id: 'risk-scoring', label: 'Risk Scoring', icon: TrendingUp },
        { id: 'behavioral-analysis', label: 'Behavioral Analysis', icon: QueryStats },
        { id: 'trend-analysis', label: 'Trend Analysis', icon: Timeline },
        { id: 'geographical-analysis', label: 'Geographic Analysis', icon: Map },
      ],
    },

    // Investigation Tools
    {
      id: 'investigations',
      label: 'Investigation Tools',
      icon: Search,
      children: [
        { id: 'case-management', label: 'Case Management', icon: Assignment },
        { id: 'call-analysis', label: 'Call Analysis', icon: AccountTree },
        { id: 'user-profiling', label: 'User Profiling', icon: Person },
        { id: 'network-visualization', label: 'Network Visualization', icon: Visibility },
        { id: 'timeline-analysis', label: 'Timeline Analysis', icon: History },
      ],
    },

    // Reports & Export
    {
      id: 'reports',
      label: 'Reports & Export',
      icon: Description,
      children: [
        { id: 'fraud-reports', label: 'Fraud Reports', icon: ReportProblem },
        { id: 'compliance-reports', label: 'Compliance Reports', icon: Shield },
        { id: 'custom-reports', label: 'Custom Reports', icon: BarChart },
        { id: 'data-export', label: 'Data Export', icon: CloudUpload },
        { id: 'scheduled-reports', label: 'Scheduled Reports', icon: Schedule },
      ],
    },

    // Administration
    {
      id: 'administration',
      label: 'Administration',
      icon: ManageAccounts,
      children: [
        { id: 'user-management', label: 'User Management', icon: Groups },
        { id: 'system-settings', label: 'System Settings', icon: Settings },
        { id: 'audit-logs', label: 'Audit Logs', icon: History },
        { id: 'notifications', label: 'Notifications', icon: Notifications, badge: notificationCount },
        { id: 'integrations', label: 'Integrations', icon: Storage },
      ],
    },
  ];

  const renderNavigationItem = (item: NavigationItem, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedSections[item.id];
    const isActive = activeSection === item.id;

    return (
      <React.Fragment key={item.id}>
        <StyledListItem>
          <Tooltip 
            title={collapsed ? item.label : ''} 
            placement="right"
            disableHoverListener={!collapsed}
          >
            <StyledListItemButton
              level={level}
              collapsed={collapsed}
              active={isActive}
              onClick={() => {
                if (hasChildren) {
                  handleSectionToggle(item.id);
                } else {
                  onSectionChange(item.id);
                }
                if (item.onClick) {
                  item.onClick();
                }
              }}
              disabled={item.disabled}
            >
              <StyledListItemIcon collapsed={collapsed}>
                <Badge badgeContent={item.badge} color="error" max={99}>
                  <item.icon fontSize="small" />
                </Badge>
              </StyledListItemIcon>
              
              <StyledListItemText 
                collapsed={collapsed}
                primary={item.label}
              />
              
              {hasChildren && !collapsed && (
                <IconButton size="small" sx={{ color: 'inherit' }}>
                  {isExpanded ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
              )}
            </StyledListItemButton>
          </Tooltip>
        </StyledListItem>

        {hasChildren && (
          <Collapse in={isExpanded && !collapsed} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map((child) => renderNavigationItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <SidebarContainer collapsed={collapsed}>
      {/* Header */}
      <SidebarHeader>
        <LogoContainer collapsed={collapsed}>
          <Shield sx={{ color: '#667eea', fontSize: 28 }} />
          <Typography 
            variant="h6" 
            sx={{ 
              color: '#fff', 
              fontWeight: 700,
              fontSize: '1.1rem',
            }}
          >
            FraudGuard 360
          </Typography>
        </LogoContainer>
        
        <CollapseButton onClick={() => onCollapse(!collapsed)}>
          {collapsed ? <ChevronRight /> : <ChevronLeft />}
        </CollapseButton>
      </SidebarHeader>

      {/* Navigation */}
      <NavigationList>
        {/* Main Dashboard */}
        {renderNavigationItem(navigationItems[0])}
        
        <SectionDivider />
        
        {/* Core Sections */}
        {navigationItems.slice(1).map((section) => (
          <React.Fragment key={section.id}>
            <SectionTitle collapsed={collapsed}>
              {section.label}
            </SectionTitle>
            {renderNavigationItem(section)}
          </React.Fragment>
        ))}
      </NavigationList>

      {/* User Section */}
      <UserSection>
        <UserInfo collapsed={collapsed}>
          <Avatar 
            sx={{ 
              width: 36, 
              height: 36, 
              bgcolor: '#667eea',
              fontSize: '0.875rem',
            }}
          >
            U
          </Avatar>
          <Box className="user-details">
            <Typography 
              variant="body2" 
              sx={{ color: '#fff', fontWeight: 600, lineHeight: 1.2 }}
            >
              User Name
            </Typography>
            <Chip 
              label={userRole}
              size="small"
              sx={{ 
                height: 20,
                fontSize: '0.75rem',
                backgroundColor: alpha('#667eea', 0.2),
                color: '#667eea',
                border: '1px solid #667eea',
              }}
            />
          </Box>
        </UserInfo>
      </UserSection>
    </SidebarContainer>
  );
}