import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Popper,
  Paper,
  MenuList,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Chip,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  List,
  ListItem,
  ListItemButton,
  Snackbar,
  Alert,
  Card,
  CardContent,
  alpha,
  styled,
  keyframes,
  useTheme,
} from '@mui/material';
import {
  ContentCopy,
  ContentPaste,
  Undo,
  Redo,
  Save,
  Search,
  FilterList,
  Sort,
  Visibility,
  Edit,
  Delete,
  Add,
  Download,
  Upload,
  Share,
  Print,
  Bookmark,
  Star,
  Flag,
  Archive,
  Refresh,
  Settings,
  Help,
  Keyboard,
  Close,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Info,
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
    border: '#e1dfdd',
    hover: '#f3f2f1',
  },
  text: {
    primary: '#323130',
    secondary: '#605e5c',
  }
};

// Command palette animations
const slideIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(-20px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
`;

const CommandPalette = styled(Paper)(({ theme }) => ({
  width: 600,
  maxHeight: 400,
  overflow: 'hidden',
  borderRadius: 12,
  boxShadow: `0 20px 60px ${alpha('#000', 0.3)}`,
  border: `1px solid ${excelColors.background.border}`,
  animation: `${slideIn} 0.2s cubic-bezier(0.25, 0.8, 0.25, 1)`,
}));

const ContextMenuPaper = styled(Paper)(({ theme }) => ({
  minWidth: 200,
  borderRadius: 8,
  boxShadow: `0 8px 30px ${alpha('#000', 0.2)}`,
  border: `1px solid ${excelColors.background.border}`,
  overflow: 'hidden',
}));

interface Command {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  shortcut?: string;
  category: string;
  action: () => void;
  disabled?: boolean;
}

interface ContextMenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  shortcut?: string;
  action: () => void;
  disabled?: boolean;
  divider?: boolean;
  submenu?: ContextMenuItem[];
}

interface PowerUserHubProps {
  commands?: Command[];
  contextMenuItems?: ContextMenuItem[];
  onCommandExecute?: (commandId: string) => void;
  enableTooltips?: boolean;
  enableNotifications?: boolean;
}

export const PowerUserHub: React.FC<PowerUserHubProps> = ({
  commands = [],
  contextMenuItems = [],
  onCommandExecute,
  enableTooltips = true,
  enableNotifications = true,
}) => {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [shortcutsDialog, setShortcutsDialog] = useState(false);
  const [notification, setNotification] = useState<{
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  } | null>(null);
  
  const commandInputRef = useRef<HTMLInputElement>(null);
  const theme = useTheme();

  // Default commands
  const defaultCommands: Command[] = [
    {
      id: 'search',
      label: 'Search Data',
      description: 'Search across all data tables',
      icon: <Search />,
      shortcut: 'Ctrl+F',
      category: 'Navigation',
      action: () => showNotification('Search activated', 'info'),
    },
    {
      id: 'filter',
      label: 'Advanced Filter',
      description: 'Open advanced filtering options',
      icon: <FilterList />,
      shortcut: 'Ctrl+Shift+F',
      category: 'Data',
      action: () => showNotification('Filters opened', 'info'),
    },
    {
      id: 'sort',
      label: 'Sort Data',
      description: 'Sort data by multiple criteria',
      icon: <Sort />,
      shortcut: 'Ctrl+Shift+S',
      category: 'Data',
      action: () => showNotification('Sort options available', 'info'),
    },
    {
      id: 'export',
      label: 'Export Data',
      description: 'Export current view to various formats',
      icon: <Download />,
      shortcut: 'Ctrl+E',
      category: 'File',
      action: () => showNotification('Export initiated', 'success'),
    },
    {
      id: 'save',
      label: 'Save Changes',
      description: 'Save all pending changes',
      icon: <Save />,
      shortcut: 'Ctrl+S',
      category: 'File',
      action: () => showNotification('Changes saved', 'success'),
    },
    {
      id: 'undo',
      label: 'Undo',
      description: 'Undo last action',
      icon: <Undo />,
      shortcut: 'Ctrl+Z',
      category: 'Edit',
      action: () => showNotification('Action undone', 'info'),
    },
    {
      id: 'redo',
      label: 'Redo',
      description: 'Redo last undone action',
      icon: <Redo />,
      shortcut: 'Ctrl+Y',
      category: 'Edit',
      action: () => showNotification('Action redone', 'info'),
    },
    {
      id: 'copy',
      label: 'Copy Selected',
      description: 'Copy selected items to clipboard',
      icon: <ContentCopy />,
      shortcut: 'Ctrl+C',
      category: 'Edit',
      action: () => showNotification('Items copied', 'success'),
    },
    {
      id: 'paste',
      label: 'Paste',
      description: 'Paste from clipboard',
      icon: <ContentPaste />,
      shortcut: 'Ctrl+V',
      category: 'Edit',
      action: () => showNotification('Items pasted', 'success'),
    },
    {
      id: 'shortcuts',
      label: 'Keyboard Shortcuts',
      description: 'View all available keyboard shortcuts',
      icon: <Keyboard />,
      shortcut: 'Ctrl+?',
      category: 'Help',
      action: () => setShortcutsDialog(true),
    },
  ];

  const allCommands = [...defaultCommands, ...commands];

  const showNotification = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    if (enableNotifications) {
      setNotification({ message, severity });
    }
  };

  // Keyboard shortcuts handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command palette
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
      }

      // Escape to close command palette
      if (e.key === 'Escape') {
        setCommandPaletteOpen(false);
        setContextMenu(null);
      }

      // Help shortcuts
      if ((e.ctrlKey || e.metaKey) && e.key === '?') {
        e.preventDefault();
        setShortcutsDialog(true);
      }

      // Execute commands by shortcut
      if (e.ctrlKey || e.metaKey) {
        const command = allCommands.find(cmd => {
          if (!cmd.shortcut) return false;
          const shortcut = cmd.shortcut.toLowerCase();
          const key = e.key.toLowerCase();
          
          if (e.shiftKey) {
            return shortcut.includes('shift') && shortcut.includes(key);
          }
          return shortcut.includes(key) && !shortcut.includes('shift');
        });

        if (command && !command.disabled) {
          e.preventDefault();
          command.action();
          onCommandExecute?.(command.id);
          showNotification(`Executed: ${command.label}`, 'success');
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [allCommands, onCommandExecute]);

  // Focus command input when palette opens
  useEffect(() => {
    if (commandPaletteOpen && commandInputRef.current) {
      commandInputRef.current.focus();
    }
  }, [commandPaletteOpen]);

  // Filter commands based on search
  const filteredCommands = allCommands.filter(command =>
    command.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    command.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    command.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group commands by category
  const commandsByCategory = filteredCommands.reduce((acc, command) => {
    if (!acc[command.category]) acc[command.category] = [];
    acc[command.category].push(command);
    return acc;
  }, {} as Record<string, Command[]>);

  const handleCommandExecute = (command: Command) => {
    command.action();
    onCommandExecute?.(command.id);
    setCommandPaletteOpen(false);
    setSearchTerm('');
    showNotification(`Executed: ${command.label}`, 'success');
  };

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const renderContextMenu = () => {
    if (!contextMenu) return null;

    const defaultContextItems: ContextMenuItem[] = [
      {
        id: 'copy',
        label: 'Copy',
        icon: <ContentCopy />,
        shortcut: 'Ctrl+C',
        action: () => showNotification('Copied to clipboard', 'success'),
      },
      {
        id: 'paste',
        label: 'Paste',
        icon: <ContentPaste />,
        shortcut: 'Ctrl+V',
        action: () => showNotification('Pasted from clipboard', 'success'),
      },
      { id: 'divider1', label: '', icon: <></>, action: () => {}, divider: true },
      {
        id: 'edit',
        label: 'Edit',
        icon: <Edit />,
        action: () => showNotification('Edit mode activated', 'info'),
      },
      {
        id: 'delete',
        label: 'Delete',
        icon: <Delete />,
        action: () => showNotification('Item deleted', 'warning'),
      },
      { id: 'divider2', label: '', icon: <></>, action: () => {}, divider: true },
      {
        id: 'refresh',
        label: 'Refresh',
        icon: <Refresh />,
        shortcut: 'F5',
        action: () => showNotification('Refreshed', 'info'),
      },
    ];

    const allContextItems = [...defaultContextItems, ...contextMenuItems];

    return (
      <Popper
        open={true}
        anchorEl={{
          getBoundingClientRect: () => ({
            x: contextMenu.x,
            y: contextMenu.y,
            width: 0,
            height: 0,
            top: contextMenu.y,
            left: contextMenu.x,
            right: contextMenu.x,
            bottom: contextMenu.y,
            toJSON: () => ({})
          })
        } as any}
        placement="bottom-start"
        style={{ zIndex: 1300 }}
      >
        <ContextMenuPaper>
          <MenuList dense>
            {allContextItems.map((item) => {
              if (item.divider) {
                return <Divider key={item.id} />;
              }
              
              return (
                <MenuItem
                  key={item.id}
                  onClick={() => {
                    item.action();
                    setContextMenu(null);
                  }}
                  disabled={item.disabled}
                  sx={{
                    '&:hover': {
                      backgroundColor: alpha(excelColors.primary.main, 0.08),
                    }
                  }}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText>{item.label}</ListItemText>
                  {item.shortcut && (
                    <Typography variant="caption" sx={{ ml: 2, color: excelColors.text.secondary }}>
                      {item.shortcut}
                    </Typography>
                  )}
                </MenuItem>
              );
            })}
          </MenuList>
        </ContextMenuPaper>
      </Popper>
    );
  };

  return (
    <>
      {/* Command Palette Trigger - Hidden div for context menu */}
      <Box
        onContextMenu={handleContextMenu}
        sx={{ 
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          pointerEvents: 'none',
          zIndex: -1
        }}
      />

      {/* Command Palette */}
      <Dialog
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            background: 'transparent',
            boxShadow: 'none',
            overflow: 'visible'
          }
        }}
      >
        <CommandPalette>
          <Box sx={{ p: 2, borderBottom: `1px solid ${excelColors.background.border}` }}>
            <TextField
              ref={commandInputRef}
              fullWidth
              placeholder="Type a command or search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <Search sx={{ mr: 1, color: excelColors.text.secondary }} />,
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                  backgroundColor: excelColors.background.default,
                }
              }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
              <Typography variant="caption" color="textSecondary">
                {filteredCommands.length} commands available
              </Typography>
              <Chip
                label="Ctrl+K to open"
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem' }}
              />
            </Box>
          </Box>

          <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
            {Object.entries(commandsByCategory).map(([category, commands]) => (
              <Box key={category}>
                <Typography
                  variant="overline"
                  sx={{
                    px: 2,
                    py: 1,
                    display: 'block',
                    color: excelColors.text.secondary,
                    fontWeight: 600,
                    backgroundColor: alpha(excelColors.background.default, 0.5)
                  }}
                >
                  {category}
                </Typography>
                {commands.map((command) => (
                  <MenuItem
                    key={command.id}
                    onClick={() => handleCommandExecute(command)}
                    disabled={command.disabled}
                    sx={{
                      px: 2,
                      py: 1.5,
                      '&:hover': {
                        backgroundColor: alpha(excelColors.primary.main, 0.08),
                      }
                    }}
                  >
                    <ListItemIcon>{command.icon}</ListItemIcon>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {command.label}
                      </Typography>
                      {command.description && (
                        <Typography variant="caption" color="textSecondary">
                          {command.description}
                        </Typography>
                      )}
                    </Box>
                    {command.shortcut && (
                      <Chip
                        label={command.shortcut}
                        size="small"
                        variant="outlined"
                        sx={{ fontSize: '0.7rem', ml: 1 }}
                      />
                    )}
                  </MenuItem>
                ))}
              </Box>
            ))}
          </Box>
        </CommandPalette>
      </Dialog>

      {/* Context Menu */}
      {renderContextMenu()}

      {/* Keyboard Shortcuts Dialog */}
      <Dialog
        open={shortcutsDialog}
        onClose={() => setShortcutsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="h6">Keyboard Shortcuts</Typography>
            <IconButton onClick={() => setShortcutsDialog(false)}>
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 3 }}>
            {Object.entries(commandsByCategory).map(([category, commands]) => (
              <Card key={category} variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom color="primary">
                    {category}
                  </Typography>
                  <List dense>
                    {commands
                      .filter(cmd => cmd.shortcut)
                      .map((command) => (
                        <ListItem key={command.id} sx={{ px: 0 }}>
                          <ListItemIcon sx={{ minWidth: 40 }}>
                            {command.icon}
                          </ListItemIcon>
                          <ListItemText
                            primary={command.label}
                            secondary={command.description}
                          />
                          <Chip
                            label={command.shortcut}
                            size="small"
                            variant="outlined"
                          />
                        </ListItem>
                      ))}
                  </List>
                </CardContent>
              </Card>
            ))}
          </Box>
        </DialogContent>
      </Dialog>

      {/* Floating Command Palette Trigger */}
      <Tooltip title="Command Palette (Ctrl+K)" placement="left">
        <IconButton
          onClick={() => setCommandPaletteOpen(true)}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            bgcolor: excelColors.primary.main,
            color: 'white',
            width: 56,
            height: 56,
            boxShadow: `0 8px 25px ${alpha(excelColors.primary.main, 0.3)}`,
            '&:hover': {
              bgcolor: excelColors.primary.dark,
              transform: 'scale(1.1)',
            },
            transition: 'all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1)',
            zIndex: 1000,
          }}
        >
          <Keyboard />
        </IconButton>
      </Tooltip>

      {/* Notifications */}
      {notification && (
        <Snackbar
          open={true}
          autoHideDuration={3000}
          onClose={() => setNotification(null)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert
            severity={notification.severity}
            onClose={() => setNotification(null)}
            variant="filled"
          >
            {notification.message}
          </Alert>
        </Snackbar>
      )}
    </>
  );
};

export default PowerUserHub;