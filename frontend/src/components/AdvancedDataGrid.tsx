import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Select,
  MenuItem,
  Checkbox,
  IconButton,
  Tooltip,
  Chip,
  Button,
  ButtonGroup,
  Menu,
  ListItemIcon,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Switch,
  Slider,
  Typography,
  Paper,
  Card,
  CardContent,
  alpha,
  styled,
  keyframes,
  useTheme,
} from '@mui/material';
import {
  Edit,
  Save,
  Cancel,
  Delete,
  Add,
  FilterList,
  Sort,
  MoreVert,
  Visibility,
  VisibilityOff,
  Download,
  Upload,
  Search,
  Clear,
  ContentCopy,
  ContentPaste,
  Undo,
  Redo,
  SelectAll,
  Deselect,
  KeyboardArrowUp,
  KeyboardArrowDown,
  DragIndicator,
  Lock,
  LockOpen,
  Star,
  StarBorder,
  Flag,
  Archive,
  Refresh,
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
    selected: '#e3f2fd',
  },
  text: {
    primary: '#323130',
    secondary: '#605e5c',
  }
};

// Revolutionary animations
const editHighlight = keyframes`
  0% { background-color: transparent; }
  50% { background-color: ${alpha(excelColors.primary.main, 0.1)}; }
  100% { background-color: transparent; }
`;

const saveSuccess = keyframes`
  0% { box-shadow: 0 0 0 0 ${alpha(excelColors.accent.green, 0.7)}; }
  70% { box-shadow: 0 0 0 10px ${alpha(excelColors.accent.green, 0)}; }
  100% { box-shadow: 0 0 0 0 ${alpha(excelColors.accent.green, 0)}; }
`;

const AdvancedTableRow = styled(TableRow)(({ theme }) => ({
  cursor: 'pointer',
  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    backgroundColor: excelColors.background.hover,
    transform: 'translateX(2px)',
    '& .row-actions': { opacity: 1 },
    '& .drag-handle': { opacity: 1 },
  },
  '&.selected': {
    backgroundColor: excelColors.background.selected,
    '&:hover': {
      backgroundColor: alpha(excelColors.background.selected, 0.8),
    }
  },
  '&.editing': {
    backgroundColor: alpha(excelColors.primary.main, 0.05),
    border: `2px solid ${excelColors.primary.main}`,
  },
  '&.saving': {
    animation: `${saveSuccess} 0.6s ease-out`,
  }
}));

const EditableCell = styled(TableCell)(({ theme }) => ({
  position: 'relative',
  '&.editing': {
    padding: 4,
    animation: `${editHighlight} 0.5s ease-in-out`,
  },
  '&:hover .edit-overlay': {
    opacity: 1,
  },
  '& .edit-overlay': {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: alpha(excelColors.primary.main, 0.05),
    opacity: 0,
    transition: 'opacity 0.2s ease',
    pointerEvents: 'none',
    border: `1px dashed ${alpha(excelColors.primary.main, 0.3)}`,
  }
}));

interface Column {
  id: string;
  label: string;
  type: 'text' | 'number' | 'select' | 'boolean' | 'date' | 'currency';
  width?: number;
  editable?: boolean;
  sortable?: boolean;
  filterable?: boolean;
  required?: boolean;
  options?: string[];
  format?: (value: any) => string;
  validate?: (value: any) => string | null;
}

interface AdvancedDataGridProps {
  data: any[];
  columns: Column[];
  onDataChange?: (newData: any[]) => void;
  onRowEdit?: (rowId: any, changes: any) => void;
  onRowDelete?: (rowId: any) => void;
  onRowAdd?: (newRow: any) => void;
  enableInlineEdit?: boolean;
  enableBulkEdit?: boolean;
  enableFiltering?: boolean;
  enableSorting?: boolean;
  enableSelection?: boolean;
  enableKeyboardShortcuts?: boolean;
  title?: string;
}

export const AdvancedDataGrid: React.FC<AdvancedDataGridProps> = ({
  data: initialData,
  columns,
  onDataChange,
  onRowEdit,
  onRowDelete,
  onRowAdd,
  enableInlineEdit = true,
  enableBulkEdit = true,
  enableFiltering = true,
  enableSorting = true,
  enableSelection = true,
  enableKeyboardShortcuts = true,
  title = "Advanced Data Grid",
}) => {
  const [data, setData] = useState(initialData);
  const [editingCells, setEditingCells] = useState<Set<string>>(new Set());
  const [editValues, setEditValues] = useState<Record<string, any>>({});
  const [selectedRows, setSelectedRows] = useState<Set<any>>(new Set());
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [filters, setFilters] = useState<Record<string, any>>({});
  const [columnVisibility, setColumnVisibility] = useState<Record<string, boolean>>(
    Object.fromEntries(columns.map(col => [col.id, true]))
  );
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; rowId?: any } | null>(null);
  const [bulkEditDialog, setBulkEditDialog] = useState(false);
  const [clipboard, setClipboard] = useState<any>(null);
  const [undoStack, setUndoStack] = useState<any[]>([]);
  const [redoStack, setRedoStack] = useState<any[]>([]);
  
  const tableRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  // Keyboard shortcuts
  useEffect(() => {
    if (!enableKeyboardShortcuts) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'z':
            e.preventDefault();
            handleUndo();
            break;
          case 'y':
            e.preventDefault();
            handleRedo();
            break;
          case 'c':
            if (selectedRows.size > 0) {
              e.preventDefault();
              handleCopy();
            }
            break;
          case 'v':
            if (clipboard) {
              e.preventDefault();
              handlePaste();
            }
            break;
          case 'a':
            e.preventDefault();
            handleSelectAll();
            break;
          case 's':
            e.preventDefault();
            handleSaveAll();
            break;
        }
      }
      
      if (e.key === 'Delete' && selectedRows.size > 0) {
        e.preventDefault();
        handleBulkDelete();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [selectedRows, clipboard, undoStack, redoStack]);

  // Save state for undo/redo
  const saveState = useCallback(() => {
    setUndoStack(prev => [...prev.slice(-9), data]); // Keep last 10 states
    setRedoStack([]);
  }, [data]);

  // Data manipulation functions
  const handleCellEdit = (rowId: any, columnId: string, value: any) => {
    const cellKey = `${rowId}-${columnId}`;
    setEditingCells(prev => {
      const newSet = new Set(prev);
      newSet.add(cellKey);
      return newSet;
    });
    setEditValues(prev => ({ ...prev, [cellKey]: value }));
  };

  const handleCellSave = (rowId: any, columnId: string) => {
    const cellKey = `${rowId}-${columnId}`;
    const value = editValues[cellKey];
    
    // Validate
    const column = columns.find(col => col.id === columnId);
    if (column?.validate) {
      const error = column.validate(value);
      if (error) {
        alert(error); // Replace with toast notification in production
        return;
      }
    }

    saveState();
    
    const newData = data.map(row => 
      row.id === rowId ? { ...row, [columnId]: value } : row
    );
    
    setData(newData);
    setEditingCells(prev => {
      const newSet = new Set(prev);
      newSet.delete(cellKey);
      return newSet;
    });
    
    const newEditValues = { ...editValues };
    delete newEditValues[cellKey];
    setEditValues(newEditValues);
    
    onDataChange?.(newData);
    onRowEdit?.(rowId, { [columnId]: value });
    
    // Trigger save animation
    const rowElement = document.querySelector(`[data-row-id="${rowId}"]`);
    if (rowElement) {
      rowElement.classList.add('saving');
      setTimeout(() => rowElement.classList.remove('saving'), 600);
    }
  };

  const handleCellCancel = (rowId: any, columnId: string) => {
    const cellKey = `${rowId}-${columnId}`;
    setEditingCells(prev => {
      const newSet = new Set(prev);
      newSet.delete(cellKey);
      return newSet;
    });
    
    const newEditValues = { ...editValues };
    delete newEditValues[cellKey];
    setEditValues(newEditValues);
  };

  const handleSort = (columnId: string) => {
    if (!enableSorting) return;
    
    setSortConfig(current => ({
      key: columnId,
      direction: current?.key === columnId && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleUndo = () => {
    if (undoStack.length === 0) return;
    setRedoStack(prev => [data, ...prev]);
    const previousState = undoStack[undoStack.length - 1];
    setUndoStack(prev => prev.slice(0, -1));
    setData(previousState);
  };

  const handleRedo = () => {
    if (redoStack.length === 0) return;
    setUndoStack(prev => [...prev, data]);
    const nextState = redoStack[0];
    setRedoStack(prev => prev.slice(1));
    setData(nextState);
  };

  const handleCopy = () => {
    const selectedData = data.filter(row => selectedRows.has(row.id));
    setClipboard(selectedData);
  };

  const handlePaste = () => {
    if (!clipboard) return;
    saveState();
    const newData = [...data, ...clipboard.map((row: any) => ({ ...row, id: Date.now() + Math.random() }))];
    setData(newData);
  };

  const handleSelectAll = () => {
    if (selectedRows.size === sortedData.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(sortedData.map(row => row.id)));
    }
  };

  const handleSaveAll = () => {
    // Save all editing cells
    editingCells.forEach(cellKey => {
      const [rowId, columnId] = cellKey.split('-');
      handleCellSave(parseInt(rowId), columnId);
    });
  };

  const handleBulkDelete = () => {
    if (selectedRows.size === 0) return;
    saveState();
    const newData = data.filter(row => !selectedRows.has(row.id));
    setData(newData);
    setSelectedRows(new Set());
    onDataChange?.(newData);
  };

  const handleContextMenu = (e: React.MouseEvent, rowId?: any) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, rowId });
  };

  // Filter and sort data
  const filteredData = useMemo(() => {
    return data.filter(row => {
      return Object.entries(filters).every(([columnId, filterValue]) => {
        if (!filterValue) return true;
        const cellValue = String(row[columnId]).toLowerCase();
        return cellValue.includes(String(filterValue).toLowerCase());
      });
    });
  }, [data, filters]);

  const sortedData = useMemo(() => {
    if (!sortConfig) return filteredData;
    
    return [...filteredData].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredData, sortConfig]);

  const renderEditableCell = (row: any, column: Column) => {
    const cellKey = `${row.id}-${column.id}`;
    const isEditing = editingCells.has(cellKey);
    const value = isEditing ? editValues[cellKey] : row[column.id];

    if (isEditing) {
      return (
        <EditableCell className="editing">
          {column.type === 'select' ? (
            <Select
              value={value || ''}
              onChange={(e) => setEditValues(prev => ({ ...prev, [cellKey]: e.target.value }))}
              size="small"
              fullWidth
              autoFocus
            >
              {column.options?.map(option => (
                <MenuItem key={option} value={option}>{option}</MenuItem>
              ))}
            </Select>
          ) : column.type === 'boolean' ? (
            <Checkbox
              checked={value || false}
              onChange={(e) => setEditValues(prev => ({ ...prev, [cellKey]: e.target.checked }))}
              autoFocus
            />
          ) : (
            <TextField
              value={value || ''}
              onChange={(e) => setEditValues(prev => ({ ...prev, [cellKey]: e.target.value }))}
              onBlur={() => handleCellSave(row.id, column.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCellSave(row.id, column.id);
                if (e.key === 'Escape') handleCellCancel(row.id, column.id);
              }}
              size="small"
              fullWidth
              autoFocus
              type={column.type === 'number' ? 'number' : column.type === 'date' ? 'date' : 'text'}
            />
          )}
          
          <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
            <IconButton size="small" onClick={() => handleCellSave(row.id, column.id)}>
              <Save fontSize="small" />
            </IconButton>
            <IconButton size="small" onClick={() => handleCellCancel(row.id, column.id)}>
              <Cancel fontSize="small" />
            </IconButton>
          </Box>
        </EditableCell>
      );
    }

    return (
      <EditableCell
        onDoubleClick={() => column.editable && handleCellEdit(row.id, column.id, value)}
        onContextMenu={(e) => handleContextMenu(e, row.id)}
      >
        <div className="edit-overlay" />
        {column.format ? column.format(value) : String(value || '')}
        {column.editable && (
          <IconButton
            size="small"
            sx={{ 
              position: 'absolute',
              top: 2,
              right: 2,
              opacity: 0,
              transition: 'opacity 0.2s',
              '.MuiTableCell-root:hover &': { opacity: 1 }
            }}
            onClick={() => handleCellEdit(row.id, column.id, value)}
          >
            <Edit fontSize="small" />
          </IconButton>
        )}
      </EditableCell>
    );
  };

  return (
    <Card sx={{ overflow: 'hidden' }}>
      <CardContent sx={{ p: 0 }}>
        {/* Advanced Toolbar */}
        <Box sx={{ 
          p: 2, 
          bgcolor: alpha(excelColors.primary.main, 0.02), 
          borderBottom: `1px solid ${excelColors.background.border}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 2
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
              {title}
            </Typography>
            {selectedRows.size > 0 && (
              <Chip 
                label={`${selectedRows.size} selected`}
                color="primary"
                size="small"
                onDelete={() => setSelectedRows(new Set())}
              />
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Add Row (Ctrl+N)">
                <Button startIcon={<Add />}>Add</Button>
              </Tooltip>
              <Tooltip title="Bulk Edit">
                <Button 
                  startIcon={<Edit />}
                  onClick={() => setBulkEditDialog(true)}
                  disabled={selectedRows.size === 0}
                >
                  Edit
                </Button>
              </Tooltip>
              <Tooltip title="Delete Selected (Del)">
                <Button 
                  startIcon={<Delete />}
                  onClick={handleBulkDelete}
                  disabled={selectedRows.size === 0}
                  color="error"
                >
                  Delete
                </Button>
              </Tooltip>
            </ButtonGroup>

            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Undo (Ctrl+Z)">
                <Button 
                  startIcon={<Undo />}
                  onClick={handleUndo}
                  disabled={undoStack.length === 0}
                >
                  Undo
                </Button>
              </Tooltip>
              <Tooltip title="Redo (Ctrl+Y)">
                <Button 
                  startIcon={<Redo />}
                  onClick={handleRedo}
                  disabled={redoStack.length === 0}
                >
                  Redo
                </Button>
              </Tooltip>
            </ButtonGroup>

            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Copy (Ctrl+C)">
                <Button 
                  startIcon={<ContentCopy />}
                  onClick={handleCopy}
                  disabled={selectedRows.size === 0}
                >
                  Copy
                </Button>
              </Tooltip>
              <Tooltip title="Paste (Ctrl+V)">
                <Button 
                  startIcon={<ContentPaste />}
                  onClick={handlePaste}
                  disabled={!clipboard}
                >
                  Paste
                </Button>
              </Tooltip>
            </ButtonGroup>

            <Button startIcon={<Download />} size="small" variant="outlined">
              Export
            </Button>
          </Box>
        </Box>

        {/* Revolutionary Data Table */}
        <TableContainer ref={tableRef} sx={{ maxHeight: 600 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {enableSelection && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedRows.size === sortedData.length && sortedData.length > 0}
                      indeterminate={selectedRows.size > 0 && selectedRows.size < sortedData.length}
                      onChange={handleSelectAll}
                    />
                  </TableCell>
                )}
                
                {columns.filter(col => columnVisibility[col.id]).map((column) => (
                  <TableCell
                    key={column.id}
                    style={{ width: column.width }}
                    sx={{ 
                      cursor: column.sortable && enableSorting ? 'pointer' : 'default',
                      userSelect: 'none',
                      position: 'relative',
                      '&:hover': column.sortable && enableSorting ? {
                        bgcolor: alpha(excelColors.primary.main, 0.05)
                      } : undefined
                    }}
                    onClick={() => column.sortable && handleSort(column.id)}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {column.label}
                        {column.required && <span style={{ color: excelColors.accent.red }}>*</span>}
                      </Typography>
                      
                      {column.sortable && enableSorting && sortConfig?.key === column.id && (
                        <IconButton size="small">
                          {sortConfig.direction === 'asc' ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
                        </IconButton>
                      )}
                    </Box>
                    
                    {column.filterable && enableFiltering && (
                      <TextField
                        size="small"
                        placeholder="Filter..."
                        value={filters[column.id] || ''}
                        onChange={(e) => setFilters(prev => ({ ...prev, [column.id]: e.target.value }))}
                        onClick={(e) => e.stopPropagation()}
                        sx={{ mt: 1, width: '100%' }}
                      />
                    )}
                  </TableCell>
                ))}
                
                <TableCell width={120}>Actions</TableCell>
              </TableRow>
            </TableHead>
            
            <TableBody>
              {sortedData.map((row) => (
                <AdvancedTableRow
                  key={row.id}
                  data-row-id={row.id}
                  className={selectedRows.has(row.id) ? 'selected' : ''}
                  onContextMenu={(e) => handleContextMenu(e, row.id)}
                >
                  {enableSelection && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedRows.has(row.id)}
                        onChange={() => {
                          const newSelected = new Set(selectedRows);
                          if (newSelected.has(row.id)) {
                            newSelected.delete(row.id);
                          } else {
                            newSelected.add(row.id);
                          }
                          setSelectedRows(newSelected);
                        }}
                      />
                    </TableCell>
                  )}
                  
                  {columns.filter(col => columnVisibility[col.id]).map((column) => (
                    <React.Fragment key={column.id}>
                      {renderEditableCell(row, column)}
                    </React.Fragment>
                  ))}
                  
                  <TableCell>
                    <Box className="row-actions" sx={{ opacity: 0, transition: 'opacity 0.2s', display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Edit Row">
                        <IconButton size="small">
                          <Edit fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete Row">
                        <IconButton size="small" color="error">
                          <Delete fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <IconButton size="small">
                        <MoreVert fontSize="small" />
                      </IconButton>
                    </Box>
                  </TableCell>
                </AdvancedTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Context Menu */}
        <Menu
          open={!!contextMenu}
          onClose={() => setContextMenu(null)}
          anchorReference="anchorPosition"
          anchorPosition={contextMenu ? { top: contextMenu.y, left: contextMenu.x } : undefined}
        >
          <MenuItem onClick={() => handleCopy()}>
            <ListItemIcon><ContentCopy fontSize="small" /></ListItemIcon>
            <ListItemText>Copy</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => handlePaste()}>
            <ListItemIcon><ContentPaste fontSize="small" /></ListItemIcon>
            <ListItemText>Paste</ListItemText>
          </MenuItem>
          <Divider />
          <MenuItem>
            <ListItemIcon><Edit fontSize="small" /></ListItemIcon>
            <ListItemText>Edit Row</ListItemText>
          </MenuItem>
          <MenuItem>
            <ListItemIcon><Delete fontSize="small" /></ListItemIcon>
            <ListItemText>Delete Row</ListItemText>
          </MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default AdvancedDataGrid;