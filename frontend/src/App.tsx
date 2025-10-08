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
const { RevolutionaryModal } = lazy(() => import('./components/InteractiveModals'));
const { ReportViewer } = lazy(() => import('./components/ReportViewer'));
const { SchedulingModal } = lazy(() => import('./components/SchedulingModal'));


// Sample data generators from ExcelDashboard
const generateTransactionData = () => {
  const hours = Array.from({ length: 24 }, (_, i) => {
    const hour = i.toString().padStart(2, '0');
    return {
      time: `${hour}:00`,
      transactions: Math.floor(Math.random() * 1000) + 500,
      fraudulent: Math.floor(Math.random() * 50) + 10,
    };
  });
  return hours;
};

const generateRecentAlerts = () => [
  {
    id: 1,
    type: 'Suspicious Transaction',
    user: 'John Smith',
    amount: '$2,547.00',
    location: 'New York, NY',
    time: '2 min ago',
    severity: 'high',
    status: 'investigating',
  },
  {
    id: 2,
    type: 'Identity Theft',
    user: 'Sarah Johnson',
    amount: '$892.50',
    location: 'Los Angeles, CA',
    time: '5 min ago',
    severity: 'medium',
    status: 'reviewed',
  },
  {
    id: 3,
    type: 'Card Fraud',
    user: 'Michael Brown',
    amount: '$1,234.75',
    location: 'Chicago, IL',
    time: '8 min ago',
    severity: 'high',
    status: 'blocked',
  },
  {
    id: 4,
    type: 'Account Takeover',
    user: 'Emily Davis',
    amount: '$567.25',
    location: 'Houston, TX',
    time: '12 min ago',
    severity: 'low',
    status: 'cleared',
  },
];

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
  const [notificationCount, setNotificationCount] = useState(5);
  const [alertCount, setAlertCount] = useState(12);
  const [user, setUser] = useState<any>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  // CONSOLIDATED STATE FROM ExcelDashboard
  const [transactionData, setTransactionData] = useState(generateTransactionData());
  const [alerts, setAlerts] = useState(generateRecentAlerts());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [appliedFilters, setAppliedFilters] = useState<any>({
    riskLevel: 'all',
    dateRange: '7days',
    amount: [0, 100000],
    status: [],
    region: 'all'
  });
  const [filteredAlerts, setFilteredAlerts] = useState(generateRecentAlerts());
  const [filteredTransactionData, setFilteredTransactionData] = useState(generateTransactionData());
  const [selectedAlert, setSelectedAlert] = useState<any>(null);
  const [showAdvancedGrid, setShowAdvancedGrid] = useState(false);

  // Panel states (now includes states from ExcelDashboard)
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
  const [actionsModalOpen, setActionsModalOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  // New states for reporting
  const [reportViewerOpen, setReportViewerOpen] = useState(false);
  const [schedulingModalOpen, setSchedulingModalOpen] = useState(false);
  const [reportData, setReportData] = useState<any[]>([]);
  const [reportTitle, setReportTitle] = useState('');

  // Enhanced state management
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [notifications, setNotifications] = useState<any[]>([]);
  const [activeSearchModal, setActiveSearchModal] = useState(false);
  const [dashboardData, setDashboardData] = useState<any>(null);

  // CONSOLIDATED LOGIC FROM ExcelDashboard
  const applyFiltersToData = (transData: any[], alertsData: any[], filters: any) => {
    let filtered = alertsData;
    if (filters.riskLevel !== 'all') {
      filtered = filtered.filter((alert: any) => alert.severity === filters.riskLevel);
    }
    if (filters.amount) {
      const amountNum = parseFloat(filters.amount.toString().replace(/[$,]/g, '') || '0');
      filtered = filtered.filter((alert: any) => {
        const alertAmount = parseFloat(alert.amount?.replace(/[$,]/g, '') || '0');
        return alertAmount >= amountNum;
      });
    }
    if (filters.status && filters.status.length > 0) {
      filtered = filtered.filter((alert: any) => filters.status.includes(alert.status));
    }
    setFilteredAlerts(filtered);

    let filteredTrans = transData;
    if (filters.dateRange !== 'all') {
      const days = filters.dateRange === '7days' ? 7 : filters.dateRange === '30days' ? 30 : 365;
      filteredTrans = transData.slice(-days);
    }
    setFilteredTransactionData(filteredTrans);
  };

  const handleFilterApply = (filters: any) => {
    setAppliedFilters(filters);
    applyFiltersToData(transactionData, alerts, filters);
    setFilterPanelOpen(false);
  };

  const handleExport = (format: string, options: any) => {
    const exportData = { alerts: filteredAlerts, transactions: filteredTransactionData };
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `fraudguard-export-${Date.now()}.${format === 'excel' ? 'xlsx' : format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setDataExportOpen(false);
  };

  const handleActionExecute = (actionType: string, alertData: any) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertData?.id 
        ? { ...alert, status: actionType === 'block' ? 'blocked' : actionType === 'resolve' ? 'resolved' : 'investigating' }
        : alert
    ));
    setActionsModalOpen(false);
  };

  const handleAlertClick = (alertData: any) => {
    setSelectedAlert(alertData);
    setActionsModalOpen(true);
  };

  const handleSchedule = (schedule: any) => {
    console.log('Scheduled report:', schedule);
    showNotification(`Report "${schedule.reportType}" scheduled successfully!`, 'success');
  };

  useEffect(() => {
    const token = localStorage.getItem('fraudguard_token');
    const userData = localStorage.getItem('fraudguard_user');
    if (token && userData) {
      setUser(JSON.parse(userData));
      setIsAuthenticated(true);
    }
  }, []);

  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'F1') { event.preventDefault(); setKeyboardShortcutsOpen(true); }
      if (event.ctrlKey && event.key === 'k') {
        event.preventDefault();
        const searchTerm = prompt('Enter search term:');
        if (searchTerm?.trim()) { performSearch(searchTerm.trim()); }
      }
    };
    if (isAuthenticated) { window.addEventListener('keydown', handleGlobalKeyDown); }
    return () => { window.removeEventListener('keydown', handleGlobalKeyDown); };
  }, [isAuthenticated]);

  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const notification = { id: Date.now(), message, type, timestamp: new Date().toISOString() };
    setNotifications(prev => [notification, ...prev.slice(0, 9)]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== notification.id));
    }, 5000);
  };

  const performSearch = async (searchTerm: string) => { /* ... existing search logic ... */ };

  const refreshDashboardData = async () => {
    setIsRefreshing(true);
    setTimeout(() => {
      const newTransactionData = generateTransactionData();
      const newAlerts = generateRecentAlerts();
      setTransactionData(newTransactionData);
      setAlerts(newAlerts);
      applyFiltersToData(newTransactionData, newAlerts, appliedFilters);
      setIsRefreshing(false);
      showNotification('Dashboard data refreshed successfully!', 'success');
    }, 1000);
  };

  const saveDashboardState = () => { /* ... existing save logic ... */ };

  useEffect(() => {
    const path = location.pathname.split('/')[1] || 'dashboard';
    setCurrentSection(path);
    document.title = `${path.charAt(0).toUpperCase() + path.slice(1)} - FraudGuard 360`;
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

  // MASTER ACTION HANDLER
  const handleActionClick = (action: string, data?: any) => {
    console.log('Action clicked:', action, data);
    switch (action) {
      case 'refresh': refreshDashboardData(); break;
      case 'filter': setFilterPanelOpen(true); break;
      case 'view': setViewOptionsOpen(true); break;
      case 'timeline': setTimelineViewOpen(true); break;
      case 'reports': setReportsPanelOpen(true); break;
      case 'export': setDataExportOpen(true); break;
      case 'investigate': setInvestigationViewsOpen(true); break;
      case 'share': setShareDashboardOpen(true); break;
      case 'print': window.print(); break;
      case 'search': const term = prompt('Enter search term:'); if (term) performSearch(term); break;
      case 'assign': case 'escalate': case 'close': setCaseManagementOpen(true); break;
      case 'manage-users': case 'roles': case 'permissions': case 'settings': setAdminPanelOpen(true); break;
      case 'monitoring': case 'logs': setSystemMonitoringOpen(true); break;
      case 'notifications': setNotificationsPanelOpen(true); break;
      case 'profile': setProfileManagementOpen(true); break;
      case 'help': setKeyboardShortcutsOpen(true); break;
      case 'logout': handleLogout(); break;
      case 'alertClick': handleAlertClick(data); break;
      case 'showAdvancedGrid': setShowAdvancedGrid(data); break;
      case 'daily':
        setReportTitle('Daily Fraud Report');
        setReportData(alerts.slice(0, 10)); // Show top 10 alerts for daily report
        setReportViewerOpen(true);
        break;
      case 'weekly':
        setReportTitle('Weekly Fraud Summary');
        setReportData(transactionData.slice(0, 7).map(d => ({ day: d.time, ...d }))); // Show last 7 days of transactions
        setReportViewerOpen(true);
        break;
      case 'custom':
        setReportTitle('Custom Report Builder');
        setReportData(alerts); // Show all alerts for custom report
        setReportViewerOpen(true);
        break;
      case 'add-schedule':
        setSchedulingModalOpen(true);
        break;
      default:
        console.log('Unhandled action:', action);
        showNotification(`Feature "${action}" is being prepared for release`, 'info');
    }
  };

  if (!isAuthenticated) {
    return <Suspense fallback={<LoadingFallback />}><LoginForm onLogin={handleLogin} /></Suspense>;
  }

  const renderContent = () => {
    switch (currentSection) {
      case 'dashboard':
        return (
          <Suspense fallback={<LoadingFallback message="Loading dashboard..." />}>
            <ExcelDashboard 
              transactionData={filteredTransactionData}
              alerts={filteredAlerts}
              isRefreshing={isRefreshing}
              onRefresh={refreshDashboardData}
              onFilter={() => handleActionClick('filter')}
              onExport={() => handleActionClick('export')}
              onAlertClick={(alert) => handleActionClick('alertClick', alert)}
              onShowAdvancedGrid={setShowAdvancedGrid}
            />
          </Suspense>
        );
      // ... other cases from original App.tsx
      default:
        return (
          <Suspense fallback={<LoadingFallback message="Loading dashboard..." />}>
            <ExcelDashboard 
              transactionData={filteredTransactionData}
              alerts={filteredAlerts}
              isRefreshing={isRefreshing}
              onRefresh={refreshDashboardData}
              onFilter={() => handleActionClick('filter')}
              onExport={() => handleActionClick('export')}
              onAlertClick={(alert) => handleActionClick('alertClick', alert)}
              onShowAdvancedGrid={setShowAdvancedGrid}
            />
          </Suspense>
        );
    }
  };

  return (
    <Suspense fallback={<LoadingFallback message="Loading FraudGuard 360..." />}>
      <ExcelLayout
        activeSection={currentSection}
        onSectionChange={handleSectionChange}
        notificationCount={notificationCount}
        alertCount={alertCount}
        userRole={user?.role || 'Analyst'}
        userName={user?.username || 'John Doe'}
        onLogout={handleLogout}
        onSettings={() => handleActionClick('settings')}
        onNotifications={() => handleActionClick('notifications')}
        onActionClick={handleActionClick}
        currentTab={currentSection}
      >
        {renderContent()}
        
        {/* MODALS - Now controlled by App.tsx state */}
        <Suspense fallback={<div />}>
          <RevolutionaryModal
            open={filterPanelOpen}
            onClose={() => setFilterPanelOpen(false)}
            title="Advanced Fraud Filters"
            type="filter"
            onApplyFilter={handleFilterApply}
          />
          <RevolutionaryModal
            open={dataExportOpen}
            onClose={() => setDataExportOpen(false)}
            title="Export Dashboard Data"
            type="export"
            onExportData={handleExport}
          />
          <RevolutionaryModal
            open={actionsModalOpen}
            onClose={() => setActionsModalOpen(false)}
            title="Fraud Investigation Actions"
            type="actions"
            data={selectedAlert}
            onActionExecute={handleActionExecute}
          />
          <NotificationsPanel
            open={notificationsPanelOpen}
            onClose={() => setNotificationsPanelOpen(false)}
            notificationCount={notificationCount}
          />
          <ReportsPanel
            open={reportsPanelOpen}
            onClose={() => setReportsPanelOpen(false)}
            onActionClick={handleActionClick}
          />
          <ReportViewer
            open={reportViewerOpen}
            onClose={() => setReportViewerOpen(false)}
            title={reportTitle}
            data={reportData}
            onPrint={() => window.print()}
            onExport={() => handleExport('csv', {})}
          />
          <SchedulingModal
            open={schedulingModalOpen}
            onClose={() => setSchedulingModalOpen(false)}
            onSchedule={handleSchedule}
          />
        </Suspense>
        
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