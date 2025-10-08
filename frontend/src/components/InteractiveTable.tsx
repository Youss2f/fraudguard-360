import React, { useState, useMemo } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Chip,
  TextField,
  InputAdornment,
  Button,
  ButtonGroup,
  Menu,
  MenuItem,
  Checkbox,
  Select,
  FormControl,
  InputLabel,
  Typography,
  Card,
  CardContent,
  alpha,
  useTheme,
  styled,
  keyframes,
} from '@mui/material';
import {
  DragIndicator,
  Search,
  FilterList,
  Sort,
  ViewColumn,
  Download,
  Refresh,
  MoreVert,
  Edit,
  Delete,
  Visibility,
  Add,
  KeyboardArrowUp,
  KeyboardArrowDown,
  DragHandle,
} from '@mui/icons-material';
// Revolutionary table without external drag-drop dependency

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

// Animated row hover effect
const hoverGlow = keyframes`
  0% { box-shadow: none; }
  100% { box-shadow: 0 0 20px ${alpha(excelColors.primary.main, 0.3)}; }
`;

const InteractiveTableRow = styled(TableRow)(({ theme }) => ({
  cursor: 'pointer',
  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  '&:hover': {
    backgroundColor: alpha(excelColors.primary.main, 0.05),
    transform: 'translateX(4px)',
    animation: `${hoverGlow} 0.3s ease-in-out`,
    '& .drag-handle': {
      opacity: 1,
    },
    '& .row-actions': {
      opacity: 1,
    },
  },
  '&.dragging': {
    backgroundColor: alpha(excelColors.primary.main, 0.1),
    boxShadow: `0 8px 25px ${alpha(excelColors.primary.main, 0.3)}`,
    transform: 'rotate(2deg) scale(1.02)',
  },
}));

const EditableCell = styled(TableCell)(({ theme }) => ({
  position: 'relative',
  '&:hover .edit-icon': {
    opacity: 1,
  },
  '& .edit-icon': {
    opacity: 0,
    position: 'absolute',
    top: '50%',
    right: 8,
    transform: 'translateY(-50%)',
    transition: 'opacity 0.2s ease',
  },
}));

interface Column {
  id: string;
  label: string;
  minWidth?: number;
  align?: 'right' | 'left' | 'center';
  format?: (value: any) => string;
  sortable?: boolean;
  filterable?: boolean;
  editable?: boolean;
}

interface InteractiveTableProps {
  columns: Column[];
  data: any[];
  title?: string;
  enableDragDrop?: boolean;
  enableInlineEdit?: boolean;
  enableFiltering?: boolean;
  enableSorting?: boolean;
  enableSelection?: boolean;
  onRowClick?: (row: any) => void;
  onRowEdit?: (row: any, field: string, value: any) => void;
  onRowDelete?: (row: any) => void;
  onRowsReorder?: (newOrder: any[]) => void;
}

export const InteractiveTable: React.FC<InteractiveTableProps> = ({
  columns,
  data: initialData,
  title,
  enableDragDrop = true,
  enableInlineEdit = true,
  enableFiltering = true,
  enableSorting = true,
  enableSelection = true,
  onRowClick,
  onRowEdit,
  onRowDelete,
  onRowsReorder,
}) => {
  const [data, setData] = useState(initialData);
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [filterText, setFilterText] = useState('');
  const [selectedRows, setSelectedRows] = useState<Set<any>>(new Set());
  const [editingCell, setEditingCell] = useState<{ rowId: any; field: string } | null>(null);
  const [editValue, setEditValue] = useState('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(new Set(columns.map(col => col.id)));

  const theme = useTheme();

  // Filtering
  const filteredData = useMemo(() => {
    if (!filterText) return data;
    return data.filter(row =>
      Object.values(row).some(value =>
        String(value).toLowerCase().includes(filterText.toLowerCase())
      )
    );
  }, [data, filterText]);

  // Sorting
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

  const handleSort = (columnId: string) => {
    if (!enableSorting) return;
    
    setSortConfig(current => ({
      key: columnId,
      direction: current?.key === columnId && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const [draggedRow, setDraggedRow] = useState<any>(null);

  const handleDragStart = (e: React.DragEvent, row: any, index: number) => {
    if (!enableDragDrop) return;
    setDraggedRow({ row, index });
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, targetIndex: number) => {
    e.preventDefault();
    if (!draggedRow || !enableDragDrop) return;

    const newData = Array.from(data);
    const [reorderedItem] = newData.splice(draggedRow.index, 1);
    newData.splice(targetIndex, 0, reorderedItem);

    setData(newData);
    onRowsReorder?.(newData);
    setDraggedRow(null);
  };

  const handleRowSelect = (row: any) => {
    if (!enableSelection) return;
    
    const newSelected = new Set(selectedRows);
    if (newSelected.has(row.id)) {
      newSelected.delete(row.id);
    } else {
      newSelected.add(row.id);
    }
    setSelectedRows(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedRows.size === sortedData.length) {
      setSelectedRows(new Set());
    } else {
      setSelectedRows(new Set(sortedData.map(row => row.id)));
    }
  };

  const handleEdit = (row: any, field: string) => {
    if (!enableInlineEdit) return;
    
    setEditingCell({ rowId: row.id, field });
    setEditValue(row[field]);
  };

  const handleSaveEdit = () => {
    if (!editingCell) return;
    
    const newData = data.map(row => 
      row.id === editingCell.rowId 
        ? { ...row, [editingCell.field]: editValue }
        : row
    );
    
    setData(newData);
    onRowEdit?.(editingCell.rowId, editingCell.field, editValue);
    setEditingCell(null);
    setEditValue('');
  };

  const handleCancelEdit = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const renderCell = (row: any, column: Column) => {
    const isEditing = editingCell?.rowId === row.id && editingCell?.field === column.id;
    const value = column.format ? column.format(row[column.id]) : row[column.id];

    if (isEditing) {
      return (
        <TextField
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleSaveEdit}
          onKeyPress={(e) => {
            if (e.key === 'Enter') handleSaveEdit();
            if (e.key === 'Escape') handleCancelEdit();
          }}
          size="small"
          autoFocus
          sx={{ width: '100%' }}
        />
      );
    }

    return (
      <EditableCell
        align={column.align}
        onDoubleClick={() => column.editable && handleEdit(row, column.id)}
      >
        <Typography variant="body2">{value}</Typography>
        {column.editable && (
          <IconButton
            className="edit-icon"
            size="small"
            onClick={() => handleEdit(row, column.id)}
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
        {/* Table Header with Controls */}
        <Box sx={{ p: 2, bgcolor: alpha(excelColors.primary.main, 0.02), borderBottom: `1px solid ${excelColors.background.border}` }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            {title && (
              <Typography variant="h6" sx={{ color: excelColors.text.primary }}>
                {title}
                {selectedRows.size > 0 && (
                  <Chip 
                    label={`${selectedRows.size} selected`}
                    size="small"
                    color="primary"
                    sx={{ ml: 1 }}
                  />
                )}
              </Typography>
            )}
            
            <ButtonGroup size="small" variant="outlined">
              <Tooltip title="Add Row">
                <Button startIcon={<Add />}>Add</Button>
              </Tooltip>
              <Tooltip title="Export">
                <Button startIcon={<Download />}>Export</Button>
              </Tooltip>
              <Tooltip title="Refresh">
                <Button startIcon={<Refresh />}>Refresh</Button>
              </Tooltip>
              <Tooltip title="More Options">
                <Button onClick={(e) => setAnchorEl(e.currentTarget)}>
                  <MoreVert />
                </Button>
              </Tooltip>
            </ButtonGroup>
          </Box>

          {/* Search and Filters */}
          {enableFiltering && (
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <TextField
                placeholder="Search table..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                size="small"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search />
                    </InputAdornment>
                  ),
                }}
                sx={{ minWidth: 300 }}
              />
              
              <Button
                startIcon={<FilterList />}
                variant="outlined"
                size="small"
              >
                Advanced Filters
              </Button>

              <Button
                startIcon={<ViewColumn />}
                variant="outlined"
                size="small"
              >
                Columns
              </Button>
            </Box>
          )}
        </Box>

        {/* Revolutionary Interactive Table */}
        <TableContainer sx={{ maxHeight: 600 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {enableDragDrop && (
                  <TableCell width={40} />
                )}
                {enableSelection && (
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedRows.size === sortedData.length && sortedData.length > 0}
                      indeterminate={selectedRows.size > 0 && selectedRows.size < sortedData.length}
                      onChange={handleSelectAll}
                    />
                  </TableCell>
                )}
                {columns.filter(col => visibleColumns.has(col.id)).map((column) => (
                  <TableCell
                    key={column.id}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                    sx={{ 
                      cursor: column.sortable && enableSorting ? 'pointer' : 'default',
                      userSelect: 'none',
                      '&:hover': column.sortable && enableSorting ? {
                        bgcolor: alpha(excelColors.primary.main, 0.05)
                      } : undefined
                    }}
                    onClick={() => column.sortable && handleSort(column.id)}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {column.label}
                      </Typography>
                      {column.sortable && enableSorting && sortConfig?.key === column.id && (
                        <IconButton size="small" sx={{ ml: 0.5 }}>
                          {sortConfig.direction === 'asc' ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
                        </IconButton>
                      )}
                    </Box>
                  </TableCell>
                ))}
                <TableCell width={100}>Actions</TableCell>
              </TableRow>
            </TableHead>
            
            <TableBody>
              {sortedData.map((row, index) => (
                <InteractiveTableRow
                  key={row.id}
                  draggable={enableDragDrop}
                  onDragStart={(e) => handleDragStart(e, row, index)}
                  onDragOver={handleDragOver}
                  onDrop={(e) => handleDrop(e, index)}
                  className={draggedRow?.row.id === row.id ? 'dragging' : ''}
                  onClick={() => onRowClick?.(row)}
                >
                  {enableDragDrop && (
                    <TableCell>
                      <IconButton
                        className="drag-handle"
                        size="small"
                        sx={{ opacity: 0, transition: 'opacity 0.2s', cursor: 'grab' }}
                      >
                        <DragIndicator />
                      </IconButton>
                    </TableCell>
                  )}
                  
                  {enableSelection && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedRows.has(row.id)}
                        onChange={() => handleRowSelect(row)}
                      />
                    </TableCell>
                  )}
                  
                  {columns.filter(col => visibleColumns.has(col.id)).map((column) => (
                    <React.Fragment key={column.id}>
                      {renderCell(row, column)}
                    </React.Fragment>
                  ))}
                  
                  <TableCell>
                    <Box 
                      className="row-actions"
                      sx={{ 
                        opacity: 0, 
                        transition: 'opacity 0.2s',
                        display: 'flex',
                        gap: 0.5
                      }}
                    >
                      <Tooltip title="View">
                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); }}>
                          <Visibility fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); }}>
                          <Edit fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton 
                          size="small" 
                          onClick={(e) => { 
                            e.stopPropagation(); 
                            onRowDelete?.(row);
                          }}
                          sx={{ color: excelColors.accent.red }}
                        >
                          <Delete fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </InteractiveTableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Options Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={() => setAnchorEl(null)}
        >
          <MenuItem>Column Settings</MenuItem>
          <MenuItem>Export All</MenuItem>
          <MenuItem>Import Data</MenuItem>
          <MenuItem>Table Settings</MenuItem>
        </Menu>
      </CardContent>
    </Card>
  );
};

export default InteractiveTable;