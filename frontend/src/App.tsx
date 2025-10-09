/**
 * FraudGuard 360 - Professional Fraud Detection Platform
 * Enterprise-grade real-time fraud prevention system with Windows-style interface
 * Enhanced with performance optimizations and lazy loading
 */

import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { CssBaseline, Box, CircularProgress, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { excelTheme } from './theme/excelTheme';
import ErrorBoundary from './components/ErrorBoundary';
import { setupGlobalErrorHandlers } from './utils/errorHandling';

// Lazy load components for better performance
const RibbonNavigation = lazy(() => import('./components/RibbonNavigation'));
const EnterpriseDashboard = lazy(() => import('./components/EnterpriseDashboard'));
const FraudNetworkVisualization = lazy(() => import('./components/FraudNetworkVisualization'));
const Reports = lazy(() => import('./components/Reports'));
const Settings = lazy(() => import('./components/Settings'));
const LoginForm = lazy(() => import('./components/LoginForm'));
const ExcelLayout = lazy(() => import('./components/ExcelLayout'));
const ExcelDashboard = lazy(() => import('./components/ExcelDashboard'));
const ExcelRealTimeMonitoring = lazy(() => import('./components/ExcelRealTimeMonitoring'));
const ExcelFraudDetection = lazy(() => import('./components/ExcelFraudDetection'));
const ExcelAnalytics = lazy(() => import('./components/ExcelAnalytics'));
const ExcelReports = lazy(() => import('./components/ExcelReports'));

// Lazy load panels and modals
const FilterPanel = lazy(() => import('./components/FilterPanel'));
const ViewOptionsPanel = lazy(() => import('./components/ViewOptionsPanel'));
const DataExportPanel = lazy(() => import('./components/DataExportPanel'));
const NotificationsPanel = lazy(() => import('./components/NotificationsPanel'));
const PlaceholderPage = lazy(() => import('./components/PlaceholderPage'));
const InvestigationToolsPanel = lazy(() => import('./components/InvestigationToolsPanel'));
const AdminPanel = lazy(() => import('./components/AdminPanel'));
const ReportsPanel = lazy(() => import('./components/ReportsPanel'));
const TimelineViewPanel = lazy(() => import('./components/TimelineViewPanel'));
const ShareDashboardPanel = lazy(() => import('./components/ShareDashboardPanel'));
const ProfileManagementPanel = lazy(() => import('./components/ProfileManagementPanel'));
const QuickAccessToolbar = lazy(() => import('./components/QuickAccessToolbar'));
const NavigationPanel = lazy(() => import('./components/NavigationPanel'));
const StatusBar = lazy(() => import('./components/StatusBar'));
const KeyboardShortcutsDialog = lazy(() => import('./components/KeyboardShortcutsDialog'));
const CaseManagementPanel = lazy(() => import('./components/CaseManagementPanel'));
const AdvancedExportSystem = lazy(() => import('./components/AdvancedExportSystem'));
const SystemMonitoringPanel = lazy(() => import('./components/SystemMonitoringPanel'));
const AlertManagementSystem = lazy(() => import('./components/AlertManagementSystem'));
const InvestigationViews = lazy(() => import('./components/InvestigationViews'));
const RealTimeMonitoring = lazy(() => import('./components/RealTimeMonitoring'));
const AlertsAndWarnings = lazy(() => import('./components/AlertsAndWarnings'));
const FraudInvestigations = lazy(() => import('./components/FraudInvestigations'));
const RiskScoring = lazy(() => import('./components/RiskScoring'));
const TestComponent = lazy(() => import('./components/TestComponent'));
const SearchResultsModal = lazy(() => import('./components/SearchResultsModal'));
const NotificationSystem = lazy(() => import('./components/NotificationSystem'));

// Loading component
const LoadingFallback: React.FC<{ message?: string }> = ({ message = "Loading..." }) => (
  <Box
    display="flex"
    flexDirection="column"
    justifyContent="center"
    alignItems="center"
    minHeight="200px"
    gap={2}
  >
    <CircularProgress size={40} />
    <Typography variant="body2" color="textSecondary">
      {message}
    </Typography>
  </Box>
);

// Main App Component with Router
function App() {
  // Setup global error handlers on app initialization
  useEffect(() => {
    setupGlobalErrorHandlers();
  }, []);

  return (
    <Router>
      <ThemeProvider theme={excelTheme}>
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
  const [currentSection, setCurrentSection] = useState('dashboard');
  const [notificationCount] = useState(5);
  const [alertCount] = useState(12);
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
  
  // Enhanced state management
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [notifications, setNotifications] = useState<any[]>([]);
  const [activeSearchModal, setActiveSearchModal] = useState(false);
  const [dashboardData, setDashboardData] = useState<any>(null);

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
        if (searchTerm?.trim()) {
          performSearch(searchTerm.trim());
        }
      }
      
      // Section navigation shortcuts
      if (event.ctrlKey && !event.shiftKey) {
        switch (event.key) {
          case '1':
            event.preventDefault();
            handleSectionChange('dashboard');
            break;
          case '2':
            event.preventDefault();
            handleSectionChange('investigations');
            break;
          case '3':
            event.preventDefault();
            handleSectionChange('reports');
            break;
          case '4':
            event.preventDefault();
            handleSectionChange('settings');
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

  // Enhanced utility functions
  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const notification = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toISOString()
    };
    setNotifications(prev => [notification, ...prev.slice(0, 9)]); // Keep last 10
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== notification.id));
    }, 5000);
  };

  const performSearch = async (searchTerm: string) => {
    if (!searchTerm.trim()) return;
    
    setIsLoading(true);
    setSearchQuery(searchTerm);
    
    try {
      // Mock search results with comprehensive data
      await new Promise(resolve => setTimeout(resolve, 800)); // Simulate API delay
      
      const mockAlerts = [
        { id: 1, type: 'alert', title: `High-Risk Transaction Alert`, content: `Suspicious $5,000 transaction from ${searchTerm}`, score: 0.95, severity: 'critical', timestamp: new Date(Date.now() - 5 * 60 * 1000) },
        { id: 2, type: 'alert', title: `Identity Verification Failed`, content: `Multiple failed login attempts for user ${searchTerm}`, score: 0.87, severity: 'high', timestamp: new Date(Date.now() - 15 * 60 * 1000) },
        { id: 3, type: 'alert', title: `Card Fraud Detection`, content: `Unusual spending pattern detected for ${searchTerm}`, score: 0.73, severity: 'medium', timestamp: new Date(Date.now() - 30 * 60 * 1000) }
      ];

      const mockTransactions = [
        { id: 4, type: 'transaction', title: `Transaction #TX-${Math.floor(Math.random() * 10000)}`, content: `Payment of $${(Math.random() * 1000 + 100).toFixed(2)} by ${searchTerm}`, score: 0.82, amount: Math.random() * 1000 + 100, status: 'completed' },
        { id: 5, type: 'transaction', title: `Transfer #TF-${Math.floor(Math.random() * 10000)}`, content: `Bank transfer involving ${searchTerm}`, score: 0.69, amount: Math.random() * 5000 + 500, status: 'pending' },
        { id: 6, type: 'transaction', title: `Refund #RF-${Math.floor(Math.random() * 10000)}`, content: `Refund processed for ${searchTerm}`, score: 0.51, amount: Math.random() * 200 + 50, status: 'completed' }
      ];

      const mockUsers = [
        { id: 7, type: 'user', title: `User Profile: ${searchTerm}`, content: `Account details and activity history for ${searchTerm}`, score: 0.91, riskLevel: 'medium', joinDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000) },
        { id: 8, type: 'user', title: `Related Account: ${searchTerm.split(' ')[0]}`, content: `Linked account with similar patterns to ${searchTerm}`, score: 0.65, riskLevel: 'low', joinDate: new Date(Date.now() - 200 * 24 * 60 * 60 * 1000) }
      ];

      const mockCases = [
        { id: 9, type: 'case', title: `Investigation Case #C-${Math.floor(Math.random() * 1000)}`, content: `Active fraud investigation involving ${searchTerm}`, score: 0.88, status: 'active', priority: 'high' },
        { id: 10, type: 'case', title: `Closed Case #C-${Math.floor(Math.random() * 1000)}`, content: `Resolved fraud case related to ${searchTerm}`, score: 0.76, status: 'closed', priority: 'medium' }
      ];

      // Filter results based on search term
      const allResults = [...mockAlerts, ...mockTransactions, ...mockUsers, ...mockCases];
      const filteredResults = allResults.filter(item => 
        item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.content.toLowerCase().includes(searchTerm.toLowerCase())
      );

      // Sort by relevance score
      const sortedResults = filteredResults.sort((a, b) => b.score - a.score);
      
      setSearchResults(sortedResults);
      setActiveSearchModal(true);
      showNotification(`Found ${sortedResults.length} results for "${searchTerm}"`, 'success');
    } catch (error) {
      console.error('Search error:', error);
      showNotification('Search failed. Please try again.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshDashboardData = async () => {
    setIsLoading(true);
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Generate new mock dashboard data
      const newDashboardData = {
        totalTransactions: Math.floor(Math.random() * 10000) + 50000,
        fraudDetected: Math.floor(Math.random() * 100) + 247,
        totalUsers: Math.floor(Math.random() * 1000) + 15000,
        systemHealth: Math.floor(Math.random() * 20) + 80,
        activeAlerts: Math.floor(Math.random() * 20) + 5,
        recentActivity: [
          { type: 'transaction', message: 'High-value transaction processed', timestamp: new Date() },
          { type: 'alert', message: 'New fraud pattern detected', timestamp: new Date(Date.now() - 5 * 60 * 1000) },
          { type: 'user', message: 'New user registered', timestamp: new Date(Date.now() - 10 * 60 * 1000) }
        ]
      };
      
      setDashboardData(newDashboardData);
      setLastUpdate(new Date());
      showNotification('Dashboard data refreshed successfully!', 'success');
    } catch (error) {
      console.error('Refresh error:', error);
      setLastUpdate(new Date());
      showNotification('Refresh completed with cached data', 'info');
    } finally {
      setIsLoading(false);
    }
  };

  const saveDashboardState = () => {
    try {
      const state = {
        currentSection,
        timestamp: new Date().toISOString(),
        user: user?.username,
        preferences: {
          zoomLevel,
          openPanels: {
            filterPanelOpen,
            viewOptionsOpen,
            notificationsPanelOpen
          }
        }
      };
      
      localStorage.setItem('fraudguard_dashboard_state', JSON.stringify(state));
      showNotification('Dashboard state saved successfully!', 'success');
    } catch (error) {
      console.error('Save error:', error);
      showNotification('Failed to save dashboard state', 'error');
    }
  };

  // Update current section based on URL and page title
  useEffect(() => {
    const path = location.pathname.split('/')[1] || 'dashboard';
    setCurrentSection(path);
    
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

  const handleSectionChange = (section: string) => {
    setCurrentSection(section);
    navigate(`/${section}`);
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
    
    // Handle various ribbon and quick access actions with real implementations
    switch (action) {
      // Dashboard actions
      case 'save':
        saveDashboardState();
        break;
        
      case 'refresh':
        refreshDashboardData();
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
        // Enhanced global search functionality
        const searchTerm = prompt('Enter search term:');
        if (searchTerm?.trim()) {
          performSearch(searchTerm.trim());
        }
        break;
        
      case 'share':
        setShareDashboardOpen(true);
        break;
        
      case 'print':
        try {
          window.print();
          showNotification('Print dialog opened', 'info');
        } catch (error) {
          showNotification('Print failed. Please try again.', 'error');
        }
        break;
        
      // Investigation actions
      case 'expand':
        setInvestigationViewsOpen(true);
        showNotification('Opening investigation views...', 'info');
        break;
        
      case 'patterns':
        setInvestigationViewsOpen(true);
        showNotification('Loading fraud pattern analysis...', 'info');
        break;
        
      case 'assign':
        setAlertManagementOpen(true);
        showNotification('Opening case assignment panel...', 'info');
        break;
        
      case 'escalate':
        setCaseManagementOpen(true);
        showNotification('Opening case escalation...', 'info');
        break;
        
      case 'close':
        setCaseManagementOpen(true);
        showNotification('Opening case closure workflow...', 'info');
        break;
        
      case 'export-case':
        setCaseManagementOpen(true);
        showNotification('Preparing case export...', 'info');
        break;
        
      case 'report':
        setReportsPanelOpen(true);
        showNotification('Loading report generator...', 'info');
        break;
        
      case 'share-case':
        setCaseManagementOpen(true);
        showNotification('Opening case sharing options...', 'info');
        break;
        
      // Reports actions
      case 'daily':
        setReportsPanelOpen(true);
        showNotification('Generating daily report...', 'info');
        break;
        
      case 'weekly':
        setReportsPanelOpen(true);
        showNotification('Generating weekly report...', 'info');
        break;
        
      case 'custom':
        setReportsPanelOpen(true);
        showNotification('Opening custom report builder...', 'info');
        break;
        
      case 'pdf':
        setAdvancedExportOpen(true);
        showNotification('Preparing PDF export...', 'info');
        break;
        
      case 'excel':
        setAdvancedExportOpen(true);
        showNotification('Preparing Excel export...', 'info');
        break;
        
      case 'email':
        setAdvancedExportOpen(true);
        showNotification('Opening email delivery options...', 'info');
        break;
        
      // Admin actions
      case 'manage-users':
        setAdminPanelOpen(true);
        showNotification('Loading user management...', 'info');
        break;
        
      case 'roles':
        setAdminPanelOpen(true);
        showNotification('Loading role management...', 'info');
        break;
        
      case 'permissions':
        setAdminPanelOpen(true);
        showNotification('Loading permission settings...', 'info');
        break;
        
      case 'settings':
        setAdminPanelOpen(true);
        showNotification('Opening system settings...', 'info');
        break;
        
      case 'monitoring':
        setSystemMonitoringOpen(true);
        showNotification('Loading system monitoring...', 'info');
        break;
        
      case 'logs':
        setSystemMonitoringOpen(true);
        showNotification('Loading system logs...', 'info');
        break;
        
      // User actions
      case 'notifications':
        setNotificationsPanelOpen(true);
        break;
        
      case 'profile':
        setProfileManagementOpen(true);
        showNotification('Loading profile settings...', 'info');
        break;
        
      case 'help':
        setKeyboardShortcutsOpen(true);
        break;
        
      case 'logout':
        handleLogout();
        showNotification('Logged out successfully', 'success');
        break;
        
      default:
        console.log('Unhandled action:', action);
        showNotification(`Feature "${action}" is being prepared for release`, 'info');
    }
  };



  // Show login form if not authenticated
  if (!isAuthenticated) {
    return (
      <Suspense fallback={<LoadingFallback message="Loading login..." />}>
        <LoginForm onLogin={handleLogin} />
      </Suspense>
    );
  }

  const renderContent = () => {
    switch (currentSection) {
      case 'dashboard':
        return (
          <Suspense fallback={<LoadingFallback message="Loading dashboard..." />}>
            <ExcelDashboard />
          </Suspense>
        );
      case 'monitoring':
      case 'live-alerts':
      case 'network-activity':
      case 'system-health':
      case 'real-time-monitoring':
        return (
          <Suspense fallback={<LoadingFallback message="Loading monitoring..." />}>
            <ExcelRealTimeMonitoring />
          </Suspense>
        );
      case 'fraud-detection':
        return (
          <Suspense fallback={<LoadingFallback message="Loading fraud detection..." />}>
            <ExcelFraudDetection />
          </Suspense>
        );
      case 'analytics':
      case 'investigations':
      case 'fraud-patterns':
      case 'network-analysis':
      case 'risk-scoring':
        return (
          <Suspense fallback={<LoadingFallback message="Loading analytics..." />}>
            <ExcelAnalytics />
          </Suspense>
        );
      case 'data-sources':
        return (
          <Suspense fallback={<LoadingFallback message="Loading data sources..." />}>
            <PlaceholderPage
              title="Data Sources"
              description="Manage and configure data sources, connections, and integration pipelines."
              features={['Source Configuration', 'Data Connectors', 'Pipeline Management', 'Data Quality Checks']}
            />
          </Suspense>
        );
      case 'data-quality':
        return (
          <Suspense fallback={<LoadingFallback message="Loading data quality..." />}>
            <PlaceholderPage
              title="Data Quality"
              description="Monitor and maintain data quality with automated validation and cleansing processes."
              features={['Quality Metrics', 'Data Validation', 'Cleansing Rules', 'Quality Reports']}
            />
          </Suspense>
        );
      case 'reports':
        return (
          <Suspense fallback={<LoadingFallback message="Loading reports..." />}>
            <ExcelReports />
          </Suspense>
        );
      case 'users':
      case 'user-management':
        return (
          <Suspense fallback={<LoadingFallback message="Loading user management..." />}>
            <PlaceholderPage
              title="User Management"
              description="Manage user accounts, access levels, and authentication settings."
              features={['User Accounts', 'Role Management', 'Access Control', 'Activity Tracking']}
            />
          </Suspense>
        );
      case 'permissions':
        return (
          <Suspense fallback={<LoadingFallback message="Loading permissions..." />}>
            <PlaceholderPage
              title="Permissions & Access Control"
              description="Configure detailed permissions and access controls for system security."
              features={['Role-based Access', 'Feature Permissions', 'Data Access Rules', 'Audit Logs']}
            />
          </Suspense>
        );
      case 'settings':
      case 'system-settings':
        return (
          <Suspense fallback={<LoadingFallback message="Loading settings..." />}>
            <Settings />
          </Suspense>
        );
      case 'test':
        return (
          <Suspense fallback={<LoadingFallback message="Loading test component..." />}>
            <TestComponent />
          </Suspense>
        );
      default:
        return (
          <Suspense fallback={<LoadingFallback message="Loading dashboard..." />}>
            <ExcelDashboard />
          </Suspense>
        );
    }
  };

  return (
    <Suspense fallback={<LoadingFallback message="Loading FraudGuard 360..." />}>
      <ExcelLayout
        activeSection={currentSection}
        onSectionChange={setCurrentSection}
        notificationCount={notificationCount}
        alertCount={alertCount}
        userRole={user?.role || 'Analyst'}
        userName={user?.username || 'John Doe'}
        onLogout={handleLogout}
        onSettings={() => setCurrentSection('settings')}
        onNotifications={() => setNotificationsPanelOpen(true)}
      >
        {renderContent()}
        
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
              exportType={currentSection as any}
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
          
          {/* Enhanced Components */}
          <SearchResultsModal
            open={activeSearchModal}
            onClose={() => setActiveSearchModal(false)}
            results={searchResults}
            query={searchQuery}
            loading={isLoading && searchQuery !== ''}
          />
          
          <NotificationSystem
            notifications={notifications}
            onClose={(id) => setNotifications(prev => prev.filter(n => n.id !== id))}
          />
          
      </ExcelLayout>
    </Suspense>
  );
}

// Enhanced App component with error boundary
const AppWithErrorBoundary = () => (
  <ErrorBoundary level="app">
    <App />
  </ErrorBoundary>
);

export default AppWithErrorBoundary;
