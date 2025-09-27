/**
 * FraudGuard 360 - Professional Fraud Detection Platform
 * Enterprise-grade real-time fraud prevention system with Windows-style interface
 */

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { CssBaseline, Box } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { technicalTheme } from './theme/technicalTheme';
import RibbonNavigation from './components/RibbonNavigation';
import EnterpriseDashboard from './components/EnterpriseDashboard';
import FraudNetworkVisualization from './components/FraudNetworkVisualization';
import Reports from './components/Reports';
import Settings from './components/Settings';
import LoginForm from './components/LoginForm';
import FilterPanel from './components/FilterPanel';
import ViewOptionsPanel from './components/ViewOptionsPanel';
import DataExportPanel from './components/DataExportPanel';
import NotificationsPanel from './components/NotificationsPanel';
import PlaceholderPage from './components/PlaceholderPage';
import InvestigationToolsPanel from './components/InvestigationToolsPanel';
import AdminPanel from './components/AdminPanel';
import ReportsPanel from './components/ReportsPanel';
import TimelineViewPanel from './components/TimelineViewPanel';
import ShareDashboardPanel from './components/ShareDashboardPanel';
import ProfileManagementPanel from './components/ProfileManagementPanel';
import QuickAccessToolbar from './components/QuickAccessToolbar';
import NavigationPanel from './components/NavigationPanel';
import StatusBar from './components/StatusBar';
import KeyboardShortcutsDialog from './components/KeyboardShortcutsDialog';
import CaseManagementPanel from './components/CaseManagementPanel';
import AdvancedExportSystem from './components/AdvancedExportSystem';
import SystemMonitoringPanel from './components/SystemMonitoringPanel';
import AlertManagementSystem from './components/AlertManagementSystem';
import InvestigationViews from './components/InvestigationViews';

// Main App Component with Router
function App() {
  return (
    <Router>
      <ThemeProvider theme={technicalTheme}>
        <CssBaseline />
        <AppContent />
      </ThemeProvider>
    </Router>
  );
}

// App Content Component (handles authentication and routing)
function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [notificationCount] = useState(5);
  const [user, setUser] = useState<any>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  // Panel states
  const [filterPanelOpen, setFilterPanelOpen] = useState(false);
  const [viewOptionsOpen, setViewOptionsOpen] = useState(false);
  const [dataExportOpen, setDataExportOpen] = useState(false);
  const [notificationsPanelOpen, setNotificationsPanelOpen] = useState(false);
  const [investigationToolsPanelOpen, setInvestigationToolsPanelOpen] = useState(false);
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [reportsPanelOpen, setReportsPanelOpen] = useState(false);
  const [timelineViewOpen, setTimelineViewOpen] = useState(false);
  const [shareDashboardOpen, setShareDashboardOpen] = useState(false);
  const [profileManagementOpen, setProfileManagementOpen] = useState(false);
  const [keyboardShortcutsOpen, setKeyboardShortcutsOpen] = useState(false);
  const [caseManagementOpen, setCaseManagementOpen] = useState(false);
  const [advancedExportOpen, setAdvancedExportOpen] = useState(false);
  const [systemMonitoringOpen, setSystemMonitoringOpen] = useState(false);
  const [alertManagementOpen, setAlertManagementOpen] = useState(false);
  const [investigationViewsOpen, setInvestigationViewsOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    // Check for existing authentication
    const token = localStorage.getItem('fraudguard_token');
    const userData = localStorage.getItem('fraudguard_user');
    
    if (token && userData) {
      setUser(JSON.parse(userData));
      setIsAuthenticated(true);
    }
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      // F1 for help
      if (event.key === 'F1') {
        event.preventDefault();
        setKeyboardShortcutsOpen(true);
      }
      
      // Ctrl+K for global search
      if (event.ctrlKey && event.key === 'k') {
        event.preventDefault();
        const searchTerm = prompt('Enter search term:');
        if (searchTerm) {
          console.log('Searching for:', searchTerm);
          alert(`Global search for "${searchTerm}" - Feature ready for implementation!`);
        }
      }
      
      // Tab navigation shortcuts
      if (event.ctrlKey && !event.shiftKey) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleTabChange('dashboard');
            break;
          case '2':
            event.preventDefault();
            handleTabChange('investigations');
            break;
          case '3':
            event.preventDefault();
            handleTabChange('reports');
            break;
          case '4':
            event.preventDefault();
            handleTabChange('settings');
            break;
        }
      }
    };

    if (isAuthenticated) {
      window.addEventListener('keydown', handleGlobalKeyDown);
    }
    
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, [isAuthenticated]);

  // Update current tab based on URL and page title
  useEffect(() => {
    const path = location.pathname.split('/')[1] || 'dashboard';
    setCurrentTab(path);
    
    // Update page title based on current route
    const pageTitles: { [key: string]: string } = {
      'dashboard': 'Dashboard - FraudGuard 360',
      'real-time-monitoring': 'Real-time Monitoring - FraudGuard 360',
      'alerts': 'Alerts & Warnings - FraudGuard 360',
      'investigations': 'Investigations - FraudGuard 360',
      'fraud-patterns': 'Fraud Patterns - FraudGuard 360',
      'network-analysis': 'Network Analysis - FraudGuard 360',
      'risk-scoring': 'Risk Scoring - FraudGuard 360',
      'data-sources': 'Data Sources - FraudGuard 360',
      'data-quality': 'Data Quality - FraudGuard 360',
      'reports': 'Reports - FraudGuard 360',
      'users': 'User Management - FraudGuard 360',
      'permissions': 'Permissions - FraudGuard 360',
      'settings': 'Settings - FraudGuard 360',
    };
    
    document.title = pageTitles[path] || 'FraudGuard 360 - Professional Fraud Detection';
  }, [location]);

  const handleTabChange = (tab: string) => {
    setCurrentTab(tab);
    navigate(`/${tab}`);
  };

  const handleLogin = (userData: any) => {
    setUser(userData);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('fraudguard_token');
    localStorage.removeItem('fraudguard_user');
    setUser(null);
    setIsAuthenticated(false);
  };

  const handleActionClick = (action: string) => {
    console.log('Action clicked:', action);
    
    // Handle various ribbon and quick access actions
    switch (action) {
      // Dashboard actions
      case 'save':
        // Save current dashboard state
        localStorage.setItem('fraudguard_dashboard_state', JSON.stringify({
          currentTab,
          timestamp: new Date().toISOString()
        }));
        alert('Dashboard state saved successfully!');
        break;
      case 'refresh':
        setLastUpdate(new Date());
        // In a real app, this would refresh data from the API
        alert('Data refreshed successfully!');
        break;
      case 'filter':
        setFilterPanelOpen(true);
        break;
      case 'view':
        setViewOptionsOpen(true);
        break;
      case 'timeline':
        setTimelineViewOpen(true);
        break;
      case 'reports':
        setReportsPanelOpen(true);
        break;
      case 'export':
        setAdvancedExportOpen(true);
        break;
      case 'investigate':
        setInvestigationViewsOpen(true);
        break;
      case 'search':
        // Global search functionality
        const searchTerm = prompt('Enter search term:');
        if (searchTerm) {
          console.log('Searching for:', searchTerm);
          alert(`Global search for "${searchTerm}" - Feature ready for implementation!`);
        }
        break;
      case 'share':
        setShareDashboardOpen(true);
        break;
      case 'print':
        window.print();
        break;
        
      // Investigation actions
      case 'expand':
        setInvestigationViewsOpen(true);
        break;
      case 'patterns':
        setInvestigationViewsOpen(true);
        break;
      case 'assign':
        setAlertManagementOpen(true);
        break;
      case 'escalate':
        setCaseManagementOpen(true);
        break;
      case 'close':
        setCaseManagementOpen(true);
        break;
      case 'export-case':
        setCaseManagementOpen(true);
        break;
      case 'report':
        setReportsPanelOpen(true);
        break;
      case 'share-case':
        setCaseManagementOpen(true);
        break;
        
      // Reports actions
      case 'daily':
        setReportsPanelOpen(true);
        break;
      case 'weekly':
        setReportsPanelOpen(true);
        break;
      case 'custom':
        setReportsPanelOpen(true);
        break;
      case 'pdf':
        setAdvancedExportOpen(true);
        break;
      case 'excel':
        setAdvancedExportOpen(true);
        break;
      case 'email':
        setAdvancedExportOpen(true);
        break;
        
      // Admin actions
      case 'manage-users':
        setAdminPanelOpen(true);
        break;
      case 'roles':
        setAdminPanelOpen(true);
        break;
      case 'permissions':
        setAdminPanelOpen(true);
        break;
      case 'settings':
        setAdminPanelOpen(true);
        break;
      case 'monitoring':
        setSystemMonitoringOpen(true);
        break;
      case 'logs':
        setSystemMonitoringOpen(true);
        break;
        
      // User actions
      case 'notifications':
        setNotificationsPanelOpen(true);
        break;
      case 'profile':
        setProfileManagementOpen(true);
        break;
      case 'help':
        setKeyboardShortcutsOpen(true);
        break;
      case 'logout':
        handleLogout();
        break;
        
      default:
        console.log('Unhandled action:', action);
        window.alert(`Action "${action}" - Feature coming soon!`);
    }
  };



  // Show login form if not authenticated
  if (!isAuthenticated) {
    return <LoginForm onLogin={handleLogin} />;
  }

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      backgroundColor: technicalTheme.palette.background.default
    }}>
      {/* Top Toolbar */}
        <QuickAccessToolbar 
          userName={user?.username || 'User'}
          notificationCount={notificationCount}
          onSave={() => handleActionClick('save')}
          onUndo={() => handleActionClick('undo')}
          onRedo={() => handleActionClick('redo')}
          onSearch={(query) => console.log('Search:', query)}
          onHelp={() => handleActionClick('help')}
          onSettings={() => handleActionClick('settings')}
          onLogout={() => setIsAuthenticated(false)}
          onActionClick={handleActionClick}
        />
        
        {/* Main Layout - Navigation + Content */}
        <Box sx={{ 
          display: 'flex',
          flex: 1,
          overflow: 'hidden',
          paddingTop: '40px', // Space for fixed toolbar
        }}>
          {/* Left Navigation Panel */}
          <NavigationPanel
            selectedItem={currentTab}
            onItemSelect={handleTabChange}
          />
          
          {/* Main Content Area */}
          <Box sx={{ 
            flex: 1,
            marginLeft: '240px', // Space for navigation panel
            overflow: 'auto',
            backgroundColor: technicalTheme.palette.background.paper,
            padding: '16px',
          }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={
                <EnterpriseDashboard 
                  onOpenInvestigation={() => setInvestigationViewsOpen(true)}
                  onOpenAlertManagement={() => setAlertManagementOpen(true)}
                  onOpenUserInvestigation={() => setInvestigationViewsOpen(true)}
                />
              } />
              <Route path="/real-time-monitoring" element={
                <PlaceholderPage
                  title="Real-time Monitoring"
                  description="Monitor fraud detection systems and transactions in real-time with advanced alerting and visualization."
                  features={['Live Transaction Stream', 'Real-time Alerts', 'System Health Monitoring', 'Performance Metrics']}
                />
              } />
              <Route path="/alerts" element={
                <PlaceholderPage
                  title="Alerts & Warnings"
                  description="Manage and respond to fraud alerts with intelligent prioritization and automated workflows."
                  features={['Alert Management', 'Risk Prioritization', 'Automated Responses', 'Alert History']}
                />
              } />
              <Route path="/investigations" element={
                <PlaceholderPage
                  title="Fraud Investigations"
                  description="Comprehensive investigation tools for analyzing suspected fraudulent activities and building cases."
                  features={['Case Management', 'Evidence Collection', 'Timeline Analysis', 'Investigation Reports']}
                />
              } />
              <Route path="/fraud-patterns" element={<FraudNetworkVisualization />} />
              <Route path="/network-analysis" element={<FraudNetworkVisualization />} />
              <Route path="/risk-scoring" element={
                <PlaceholderPage
                  title="Risk Scoring"
                  description="Advanced risk assessment and scoring algorithms for transaction and entity evaluation."
                  features={['ML Risk Models', 'Score Calibration', 'Risk Thresholds', 'Historical Analysis']}
                />
              } />
              <Route path="/data-sources" element={
                <PlaceholderPage
                  title="Data Sources"
                  description="Manage and configure data sources, connections, and integration pipelines."
                  features={['Source Configuration', 'Data Connectors', 'Pipeline Management', 'Data Quality Checks']}
                />
              } />
              <Route path="/data-quality" element={
                <PlaceholderPage
                  title="Data Quality"
                  description="Monitor and maintain data quality with automated validation and cleansing processes."
                  features={['Quality Metrics', 'Data Validation', 'Cleansing Rules', 'Quality Reports']}
                />
              } />
              <Route path="/reports" element={<Reports />} />
              <Route path="/users" element={
                <PlaceholderPage
                  title="User Management"
                  description="Manage user accounts, access levels, and authentication settings."
                  features={['User Accounts', 'Role Management', 'Access Control', 'Activity Tracking']}
                />
              } />
              <Route path="/permissions" element={
                <PlaceholderPage
                  title="Permissions & Access Control"
                  description="Configure detailed permissions and access controls for system security."
                  features={['Role-based Access', 'Feature Permissions', 'Data Access Rules', 'Audit Logs']}
                />
              } />
              <Route path="/admin" element={<Settings />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Box>
        </Box>
        
        {/* Feature Panels */}
        <FilterPanel
          open={filterPanelOpen}
          onClose={() => setFilterPanelOpen(false)}
          onApplyFilters={(filters) => {
            console.log('Applied filters:', filters);
            // Here you would apply the filters to your data
          }}
        />
        
        <ViewOptionsPanel
          open={viewOptionsOpen}
          onClose={() => setViewOptionsOpen(false)}
          onApplyView={(viewOptions) => {
            console.log('Applied view options:', viewOptions);
            // Here you would apply the view settings
          }}
        />
        
        <DataExportPanel
          open={dataExportOpen}
          onClose={() => setDataExportOpen(false)}
          exportType={currentTab as any}
        />
        
        <NotificationsPanel
          open={notificationsPanelOpen}
          onClose={() => setNotificationsPanelOpen(false)}
          notificationCount={notificationCount}
        />
        
        <InvestigationToolsPanel
          open={investigationToolsPanelOpen}
          onClose={() => setInvestigationToolsPanelOpen(false)}
        />
        
        <AdminPanel
          open={adminPanelOpen}
          onClose={() => setAdminPanelOpen(false)}
        />
        
        <ReportsPanel
          open={reportsPanelOpen}
          onClose={() => setReportsPanelOpen(false)}
        />
        
        <TimelineViewPanel
          open={timelineViewOpen}
          onClose={() => setTimelineViewOpen(false)}
        />
        
        <ShareDashboardPanel
          open={shareDashboardOpen}
          onClose={() => setShareDashboardOpen(false)}
        />
        
        <ProfileManagementPanel
          open={profileManagementOpen}
          onClose={() => setProfileManagementOpen(false)}
        />
        
        <KeyboardShortcutsDialog
          open={keyboardShortcutsOpen}
          onClose={() => setKeyboardShortcutsOpen(false)}
        />
        
        <CaseManagementPanel
          open={caseManagementOpen}
          onClose={() => setCaseManagementOpen(false)}
        />
        
        <AdvancedExportSystem
          open={advancedExportOpen}
          onClose={() => setAdvancedExportOpen(false)}
        />
        
        <SystemMonitoringPanel
          open={systemMonitoringOpen}
          onClose={() => setSystemMonitoringOpen(false)}
        />
        
        <AlertManagementSystem
          open={alertManagementOpen}
          onClose={() => setAlertManagementOpen(false)}
        />
        
        <InvestigationViews
          open={investigationViewsOpen}
          onClose={() => setInvestigationViewsOpen(false)}
        />
        
        <StatusBar
          currentUser={user?.username || 'User'}
          systemStatus="online"
          lastUpdate={lastUpdate}
          processingTasks={0}
          zoomLevel={zoomLevel}
          onZoomChange={setZoomLevel}
        />
      </Box>
  );
}

export default App;
