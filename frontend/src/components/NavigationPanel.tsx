/**
 * Navigation Panel
 * Collapsible left-hand navigation with hierarchical tree structure
 */

import React, { useState } from 'react';
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
  styled,
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
} from '@mui/icons-material';
import { technicalColorSystem } from '../theme/technicalTheme';

const NavigationContainer = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed: boolean }>(({ collapsed }) => ({
  position: 'fixed',
  top: '40px', // Below toolbar
  left: 0,
  bottom: 0,
  width: collapsed ? '48px' : '240px',
  backgroundColor: technicalColorSystem.background.secondary,
  borderRight: `1px solid ${technicalColorSystem.background.border}`,
  transition: 'width 0.3s ease',
  zIndex: 1100,
  display: 'flex',
  flexDirection: 'column',
}));

const CollapseButton = styled(IconButton)({
  position: 'absolute',
  top: '8px',
  right: '8px',
  width: '24px',
  height: '24px',
  color: technicalColorSystem.text.muted,
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    color: technicalColorSystem.text.primary,
  },
});

const NavigationList = styled(List)({
  padding: '8px 0',
  flex: 1,
  overflow: 'auto',
});

const NavigationListItem = styled(ListItem)({
  padding: 0,
});

const NavigationButton = styled(ListItemButton, {
  shouldForwardProp: (prop) => prop !== 'level' && prop !== 'collapsed',
})<{ level?: number; collapsed?: boolean }>(({ level = 0, collapsed = false }) => ({
  minHeight: '32px',
  paddingLeft: collapsed ? '12px' : `${12 + level * 16}px`,
  paddingRight: '8px',
  paddingTop: '4px',
  paddingBottom: '4px',
  color: technicalColorSystem.text.primary,
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  '&.Mui-selected': {
    backgroundColor: technicalColorSystem.accent.primary,
    color: 'white',
    '&:hover': {
      backgroundColor: technicalColorSystem.accent.primary,
    },
  },
}));

const NavigationIcon = styled(ListItemIcon, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed = false }) => ({
  minWidth: collapsed ? 'auto' : '24px',
  marginRight: collapsed ? 0 : '8px',
  color: 'inherit',
  '& .MuiSvgIcon-root': {
    fontSize: '16px',
  },
}));

const NavigationText = styled(ListItemText, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed = false }) => ({
  display: collapsed ? 'none' : 'block',
  '& .MuiTypography-root': {
    fontSize: '13px',
    fontWeight: 500,
  },
}));

const SectionHeader = styled(Typography, {
  shouldForwardProp: (prop) => prop !== 'collapsed',
})<{ collapsed?: boolean }>(({ collapsed = false }) => ({
  display: collapsed ? 'none' : 'block',
  fontSize: '11px',
  fontWeight: 600,
  color: technicalColorSystem.text.muted,
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  padding: '16px 12px 8px 12px',
  marginTop: '16px',
  '&:first-of-type': {
    marginTop: '8px',
  },
}));

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  children?: NavigationItem[];
  badge?: number;
  onClick?: () => void;
}

interface NavigationPanelProps {
  selectedItem?: string;
  onItemSelect?: (itemId: string) => void;
  navigationItems?: NavigationItem[];
}

const defaultNavigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Overview',
    icon: <Dashboard />,
    onClick: () => console.log('Dashboard clicked'),
  },
  {
    id: 'fraud-detection',
    label: 'Fraud Detection',
    icon: <Security />,
    children: [
      {
        id: 'real-time-monitoring',
        label: 'Real-time Monitoring',
        icon: <Visibility />,
      },
      {
        id: 'alerts',
        label: 'Alerts & Warnings',
        icon: <Warning />,
        badge: 3,
      },
      {
        id: 'investigations',
        label: 'Investigations',
        icon: <Search />,
      },
    ],
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: <Assessment />,
    children: [
      {
        id: 'fraud-patterns',
        label: 'Fraud Patterns',
        icon: <TrendingUp />,
      },
      {
        id: 'network-analysis',
        label: 'Network Analysis',
        icon: <AccountTree />,
      },
      {
        id: 'risk-scoring',
        label: 'Risk Scoring',
        icon: <Timeline />,
      },
    ],
  },
  {
    id: 'data-management',
    label: 'Data Management',
    icon: <Storage />,
    children: [
      {
        id: 'data-sources',
        label: 'Data Sources',
        icon: <NetworkCheck />,
      },
      {
        id: 'data-quality',
        label: 'Data Quality',
        icon: <Circle />,
      },
    ],
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: <Description />,
  },
  {
    id: 'user-management',
    label: 'User Management',
    icon: <ManageAccounts />,
    children: [
      {
        id: 'users',
        label: 'Users',
        icon: <Groups />,
      },
      {
        id: 'permissions',
        label: 'Permissions',
        icon: <Security />,
      },
    ],
  },
];

const NavigationPanel: React.FC<NavigationPanelProps> = ({
  selectedItem,
  onItemSelect,
  navigationItems = defaultNavigationItems,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  // Auto-expand parent items when a child is selected
  React.useEffect(() => {
    if (selectedItem) {
      const newExpanded = new Set(expandedItems);
      
      // Find parent items that contain the selected item as a child
      navigationItems.forEach(item => {
        if (item.children) {
          const hasSelectedChild = item.children.some(child => child.id === selectedItem);
          if (hasSelectedChild) {
            newExpanded.add(item.id);
          }
        }
      });
      
      setExpandedItems(newExpanded);
    }
  }, [selectedItem]);

  const handleCollapse = () => {
    setCollapsed(!collapsed);
  };

  const handleExpand = (itemId: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemId)) {
      newExpanded.delete(itemId);
    } else {
      newExpanded.add(itemId);
    }
    setExpandedItems(newExpanded);
  };

  const handleItemClick = (item: NavigationItem) => {
    if (item.children) {
      // For parent items, expand/collapse and also trigger navigation if they have a direct route
      handleExpand(item.id);
      // Still trigger navigation for parent items that might have their own routes
      onItemSelect?.(item.id);
    } else {
      // For leaf items, trigger navigation
      onItemSelect?.(item.id);
      item.onClick?.();
    }
  };

  const renderNavigationItem = (item: NavigationItem, level: number = 0) => {
    const isExpanded = expandedItems.has(item.id);
    const isSelected = selectedItem === item.id;
    const hasChildren = item.children && item.children.length > 0;

    return (
      <React.Fragment key={item.id}>
        <NavigationListItem>
          <NavigationButton
            level={level}
            collapsed={collapsed}
            selected={isSelected}
            onClick={() => handleItemClick(item)}
          >
            <NavigationIcon collapsed={collapsed}>
              {item.icon}
            </NavigationIcon>
            <NavigationText 
              collapsed={collapsed}
              primary={item.label}
            />
            {!collapsed && item.badge && (
              <Box
                sx={{
                  minWidth: '16px',
                  height: '16px',
                  borderRadius: '8px',
                  backgroundColor: technicalColorSystem.status.error,
                  color: 'white',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginLeft: 'auto',
                  marginRight: '8px',
                }}
              >
                {item.badge}
              </Box>
            )}
            {!collapsed && hasChildren && (
              <Box sx={{ marginLeft: 'auto' }}>
                {isExpanded ? (
                  <ExpandLess fontSize="small" />
                ) : (
                  <ExpandMore fontSize="small" />
                )}
              </Box>
            )}
          </NavigationButton>
        </NavigationListItem>
        
        {hasChildren && !collapsed && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map((child) => renderNavigationItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  return (
    <NavigationContainer collapsed={collapsed}>
      <CollapseButton onClick={handleCollapse}>
        {collapsed ? <ChevronRight /> : <ChevronLeft />}
      </CollapseButton>

      <NavigationList>
        <SectionHeader collapsed={collapsed}>
          Main Navigation
        </SectionHeader>
        
        {navigationItems.slice(0, 4).map((item) => renderNavigationItem(item))}
        
        <SectionHeader collapsed={collapsed}>
          Tools
        </SectionHeader>
        
        {navigationItems.slice(4, 6).map((item) => renderNavigationItem(item))}
        
        <SectionHeader collapsed={collapsed}>
          Administration
        </SectionHeader>
        
        {navigationItems.slice(6).map((item) => renderNavigationItem(item))}
      </NavigationList>

      {!collapsed && (
        <Box sx={{ p: 2, borderTop: `1px solid ${technicalColorSystem.background.border}` }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            FraudGuard 360
          </Typography>
          <Typography variant="caption" color="text.secondary">
            v2.1.0 Enterprise
          </Typography>
        </Box>
      )}
    </NavigationContainer>
  );
};

export default NavigationPanel;