import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  IconButton,
  Typography,
  Tooltip,
  Divider,
  Card,
  CardContent,
  Switch,
  Slider,
  TextField,
  Button,
  Badge,
  Chip,
  Avatar,
  Paper,
  alpha,
  useTheme,
  styled,
  keyframes,
} from '@mui/material';
import {
  Dashboard,
  Analytics,
  Assessment,
  MonitorHeart,
  Settings,
  Notifications,
  TrendingUp,
  Security,
  DataObject,
  Timeline,
  ExpandLess,
  ExpandMore,
  ChevronLeft,
  ChevronRight,
  Search,
  FilterList,
  Bookmark,
  History,
  Help,
  Person,
  Logout,
  Menu,
  Close,
  Brightness4,
  Brightness7,
  Language,
  Palette,
  Speed,
  ViewModule,
  ViewList,
  GridView,
} from '@mui/icons-material';

const excelColors = {
  primary: { main: '#0078d4', dark: '#106ebe', light: '#40e0ff' },
  secondary: { main: '#107c10', dark: '#0b5e0b', light: '#6bb26b' },
  accent: { 
    orange: '#ff8c00', 
    red: '#d83b01', 
    blue: '#0078d4',
    green: '#107c10',
    purple: '#5c2e91',
    teal: '#008272'
  },
  background: {
    default: '#faf9f8',
    paper: '#ffffff',
    sidebar: '#f8f9fa',
    border: '#e1dfdd',
    hover: '#f3f2f1',
  },
  text: {
    primary: '#323130',
    secondary: '#605e5c',
  }
};

// Animated slide-in effect
const slideIn = keyframes`
  from {
    transform: translateX(-100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
`;

const pulseGlow = keyframes`
  0%, 100% { box-shadow: 0 0 5px ${alpha(excelColors.primary.main, 0.3)}; }
  50% { box-shadow: 0 0 20px ${alpha(excelColors.primary.main, 0.6)}, 0 0 30px ${alpha(excelColors.primary.main, 0.4)}; }
`;

const RevolutionarySidebar = styled(Box)(({ theme, collapsed }: { theme?: any, collapsed: boolean }) => ({
  width: collapsed ? 80 : 320,
  transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  height: '100vh',
  backgroundColor: excelColors.background.sidebar,
  borderRight: `1px solid ${excelColors.background.border}`,
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
  overflow: 'hidden',
}));

const AnimatedListItem = styled(ListItemButton)(({ theme }) => ({
  margin: '4px 8px',
  borderRadius: '8px',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  position: 'relative',
  overflow: 'hidden',
  '&:hover': {
    backgroundColor: alpha(excelColors.primary.main, 0.08),
    transform: 'translateX(4px)',
    boxShadow: `0 4px 12px ${alpha(excelColors.primary.main, 0.15)}`,
    '&::before': {
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      height: '100%',
      width: '3px',
      backgroundColor: excelColors.primary.main,
      borderRadius: '0 2px 2px 0',
    }
  },
  '&.active': {
    backgroundColor: alpha(excelColors.primary.main, 0.12),
    color: excelColors.primary.main,
    fontWeight: 600,
    animation: `${pulseGlow} 2s infinite`,
    '&::before': {
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      height: '100%',
      width: '4px',
      backgroundColor: excelColors.primary.main,
      borderRadius: '0 2px 2px 0',
    }
  },
}));

const SlidingPanel = styled(Box)(({ theme, visible }: { theme?: any, visible: boolean }) => ({
  position: 'absolute',
  top: 0,
  right: visible ? 0 : -400,
  width: 400,
  height: '100%',
  backgroundColor: excelColors.background.paper,
  borderLeft: `1px solid ${excelColors.background.border}`,
  transition: 'right 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)',
  zIndex: 1000,
  boxShadow: visible ? `0 0 50px ${alpha('#000', 0.15)}` : 'none',
  display: 'flex',
  flexDirection: 'column',
}));

interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  badge?: number;
  children?: MenuItem[];
  panel?: React.ReactNode;
  color?: string;
}

interface InteractiveSidebarProps {
  menuItems: MenuItem[];
  activeItem: string;
  onItemClick: (itemId: string) => void;
  user?: {
    name: string;
    avatar?: string;
    role?: string;
  };
  notifications?: number;
  onSettingsChange?: (setting: string, value: any) => void;
}

export const InteractiveSidebar: React.FC<InteractiveSidebarProps> = ({
  menuItems,
  activeItem,
  onItemClick,
  user,
  notifications = 0,
  onSettingsChange,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());
  const [activePanelId, setActivePanelId] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [settings, setSettings] = useState({
    darkMode: false,
    animations: true,
    notifications: true,
    autoCollapse: false,
    viewMode: 'list',
  });

  const theme = useTheme();

  const handleItemClick = (item: MenuItem) => {
    onItemClick(item.id);
    
    // Toggle panel if item has one
    if (item.panel) {
      setActivePanelId(activePanelId === item.id ? null : item.id);
    }
    
    // Handle expandable items
    if (item.children) {
      const newExpanded = new Set(expandedItems);
      if (newExpanded.has(item.id)) {
        newExpanded.delete(item.id);
      } else {
        newExpanded.add(item.id);
      }
      setExpandedItems(newExpanded);
    }
  };

  const handleSettingChange = (setting: string, value: any) => {
    setSettings(prev => ({ ...prev, [setting]: value }));
    onSettingsChange?.(setting, value);
  };

  const filteredItems = menuItems.filter(item =>
    item.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const renderMenuItem = (item: MenuItem, level = 0) => (
    <Box key={item.id}>
      <AnimatedListItem
        className={activeItem === item.id ? 'active' : ''}
        onClick={() => handleItemClick(item)}
        sx={{ 
          pl: collapsed ? 1 : 2 + level * 2,
          minHeight: 48,
        }}
      >
        <ListItemIcon 
          sx={{ 
            color: activeItem === item.id ? excelColors.primary.main : excelColors.text.secondary,
            minWidth: collapsed ? 'auto' : 40,
            justifyContent: 'center'
          }}
        >
          <Badge badgeContent={item.badge} color="error">
            {item.icon}
          </Badge>
        </ListItemIcon>
        
        {!collapsed && (
          <>
            <ListItemText
              primary={item.label}
              sx={{
                '& .MuiListItemText-primary': {
                  fontSize: '0.875rem',
                  fontWeight: activeItem === item.id ? 600 : 400,
                }
              }}
            />
            
            {item.children && (
              <IconButton size="small">
                {expandedItems.has(item.id) ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            )}
            
            {item.panel && (
              <Chip
                size="small"
                label="Panel"
                sx={{ ml: 1, height: 20 }}
              />
            )}
          </>
        )}
      </AnimatedListItem>
      
      {item.children && !collapsed && (
        <Collapse in={expandedItems.has(item.id)} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {item.children.map(child => renderMenuItem(child, level + 1))}
          </List>
        </Collapse>
      )}
    </Box>
  );

  const activeMenuItem = menuItems.find(item => item.id === activePanelId);

  return (
    <>
      <RevolutionarySidebar collapsed={collapsed}>
        {/* Header */}
        <Box sx={{ 
          p: 2, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          borderBottom: `1px solid ${excelColors.background.border}`,
          minHeight: 64,
        }}>
          {!collapsed && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Avatar 
                sx={{ 
                  width: 32, 
                  height: 32, 
                  bgcolor: excelColors.primary.main 
                }}
              >
                <Dashboard />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 700, color: excelColors.primary.main }}>
                FraudGuard 360
              </Typography>
            </Box>
          )}
          
          <Tooltip title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}>
            <IconButton
              onClick={() => setCollapsed(!collapsed)}
              sx={{ 
                color: excelColors.primary.main,
                '&:hover': {
                  backgroundColor: alpha(excelColors.primary.main, 0.1),
                  transform: 'scale(1.1)',
                }
              }}
            >
              {collapsed ? <ChevronRight /> : <ChevronLeft />}
            </IconButton>
          </Tooltip>
        </Box>

        {/* Search */}
        {!collapsed && (
          <Box sx={{ p: 2, borderBottom: `1px solid ${excelColors.background.border}` }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search navigation..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: excelColors.text.secondary }} />,
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 3,
                  backgroundColor: alpha(excelColors.primary.main, 0.02),
                }
              }}
            />
          </Box>
        )}

        {/* Navigation Menu */}
        <Box sx={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
          <List sx={{ p: 1 }}>
            {filteredItems.map(item => renderMenuItem(item))}
          </List>
        </Box>

        {/* User Profile */}
        {user && (
          <Box sx={{ 
            p: 2, 
            borderTop: `1px solid ${excelColors.background.border}`,
            backgroundColor: alpha(excelColors.primary.main, 0.02)
          }}>
            {!collapsed ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Avatar src={user.avatar} sx={{ width: 40, height: 40 }}>
                  <Person />
                </Avatar>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="subtitle2" noWrap>
                    {user.name}
                  </Typography>
                  <Typography variant="caption" color="textSecondary" noWrap>
                    {user.role || 'User'}
                  </Typography>
                </Box>
                <Tooltip title="Logout">
                  <IconButton size="small" color="error">
                    <Logout />
                  </IconButton>
                </Tooltip>
              </Box>
            ) : (
              <Tooltip title={`${user.name} - Click to expand`}>
                <IconButton onClick={() => setCollapsed(false)}>
                  <Avatar src={user.avatar} sx={{ width: 32, height: 32 }}>
                    <Person />
                  </Avatar>
                </IconButton>
              </Tooltip>
            )}
          </Box>
        )}

        {/* Notifications Badge */}
        {notifications > 0 && (
          <Badge
            badgeContent={notifications}
            color="error"
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              '& .MuiBadge-badge': {
                animation: `${pulseGlow} 2s infinite`,
              }
            }}
          />
        )}
      </RevolutionarySidebar>

      {/* Revolutionary Sliding Panel */}
      <SlidingPanel visible={!!activePanelId}>
        {activeMenuItem && (
          <>
            <Box sx={{ 
              p: 2, 
              borderBottom: `1px solid ${excelColors.background.border}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {activeMenuItem.icon}
                <Typography variant="h6">{activeMenuItem.label}</Typography>
              </Box>
              <IconButton onClick={() => setActivePanelId(null)}>
                <Close />
              </IconButton>
            </Box>
            
            <Box sx={{ flex: 1, p: 2, overflowY: 'auto' }}>
              {activeMenuItem.panel}
            </Box>
          </>
        )}
      </SlidingPanel>
    </>
  );
};

// Sample Settings Panel Component
export const SettingsPanel: React.FC<{ onSettingChange?: (setting: string, value: any) => void }> = ({
  onSettingChange
}) => {
  const [settings, setSettings] = useState({
    darkMode: false,
    animations: true,
    notifications: true,
    autoSave: true,
    dataRefresh: 30,
    chartAnimation: true,
    soundEnabled: false,
  });

  const handleChange = (setting: string, value: any) => {
    setSettings(prev => ({ ...prev, [setting]: value }));
    onSettingChange?.(setting, value);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Appearance
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Brightness7 />
              <Typography>Dark Mode</Typography>
            </Box>
            <Switch
              checked={settings.darkMode}
              onChange={(e) => handleChange('darkMode', e.target.checked)}
            />
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Speed />
              <Typography>Animations</Typography>
            </Box>
            <Switch
              checked={settings.animations}
              onChange={(e) => handleChange('animations', e.target.checked)}
            />
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data & Performance
          </Typography>
          
          <Typography gutterBottom>Auto-refresh interval (seconds)</Typography>
          <Slider
            value={settings.dataRefresh}
            onChange={(e, value) => handleChange('dataRefresh', value)}
            min={5}
            max={300}
            step={5}
            marks={[
              { value: 5, label: '5s' },
              { value: 60, label: '1m' },
              { value: 300, label: '5m' },
            ]}
          />
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Notifications
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography>Enable Notifications</Typography>
            <Switch
              checked={settings.notifications}
              onChange={(e) => handleChange('notifications', e.target.checked)}
            />
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography>Sound Alerts</Typography>
            <Switch
              checked={settings.soundEnabled}
              onChange={(e) => handleChange('soundEnabled', e.target.checked)}
            />
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default InteractiveSidebar;