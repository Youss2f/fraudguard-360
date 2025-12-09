'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import {
  Search,
  Filter,
  Download,
  MoreHorizontal,
  Eye,
  Flag,
  CheckCircle,
  XCircle,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  ArrowUpDown,
  Settings2,
  AlertTriangle,
  Shield,
  MapPin,
  Smartphone,
  Clock,
  Phone,
  PhoneCall,
  PhoneOutgoing,
  Radio,
  Globe,
  Timer,
} from 'lucide-react';

// CDR type definition (Call Detail Record)
interface CDR {
  id: string;
  callerMsisdn: string;
  calleeMsisdn: string;
  callType: 'VOICE' | 'SMS';
  duration: number; // seconds
  status: 'approved' | 'flagged' | 'blocked' | 'pending';
  riskScore: number;
  timestamp: string;
  cellTowerId: string;
  cellTowerLocation: string;
  imei: string;
  roaming: boolean;
  fraudType: 'NONE' | 'WANGIRI' | 'SIMBOX' | 'IRSF' | 'PBX_HACKING';
  riskFactors: string[];
  payload: Record<string, any>;
}

// Generate mock CDRs
const generateCDRs = (count: number): CDR[] => {
  const callTypes: CDR['callType'][] = ['VOICE', 'SMS'];
  const fraudTypes: CDR['fraudType'][] = ['NONE', 'WANGIRI', 'SIMBOX', 'IRSF', 'PBX_HACKING'];
  const cellTowers = [
    { id: 'TWR-CAS-001', location: 'Casablanca, Morocco' },
    { id: 'TWR-RBT-001', location: 'Rabat, Morocco' },
    { id: 'TWR-MRK-001', location: 'Marrakech, Morocco' },
    { id: 'TWR-FES-001', location: 'Fes, Morocco' },
    { id: 'TWR-TNG-001', location: 'Tangier, Morocco' },
    { id: 'TWR-AGD-001', location: 'Agadir, Morocco' },
  ];
  const msisdnPrefixes = ['+212661', '+212662', '+212663', '+212670', '+212671'];
  const internationalPrefixes = ['+220', '+236', '+216', '+33', '+34', '+1'];
  const riskFactors = [
    'Short duration call',
    'International premium number',
    'Multiple SIMs same IMEI',
    'Unusual call pattern',
    'High velocity dialing',
    'Known fraud destination',
    'IRSF pattern detected',
    'Static cell tower',
    'SIM Box signature',
    'Wangiri pattern',
  ];

  const cdrs: CDR[] = [];

  for (let i = 0; i < count; i++) {
    const risk = Math.random();
    let status: CDR['status'];
    let fraudType: CDR['fraudType'] = 'NONE';

    if (risk > 0.9) {
      status = 'blocked';
      fraudType = fraudTypes[Math.floor(Math.random() * (fraudTypes.length - 1)) + 1];
    } else if (risk > 0.7) {
      status = 'flagged';
      fraudType = Math.random() > 0.5 ? fraudTypes[Math.floor(Math.random() * (fraudTypes.length - 1)) + 1] : 'NONE';
    } else if (risk > 0.6) {
      status = 'pending';
    } else {
      status = 'approved';
    }

    const numRiskFactors = Math.floor(risk * 4);
    const selectedRiskFactors = riskFactors
      .sort(() => Math.random() - 0.5)
      .slice(0, numRiskFactors);

    const tower = cellTowers[Math.floor(Math.random() * cellTowers.length)];
    const callType = callTypes[Math.floor(Math.random() * callTypes.length)];
    const isInternational = risk > 0.5;
    const callerPrefix = msisdnPrefixes[Math.floor(Math.random() * msisdnPrefixes.length)];
    const calleePrefix = isInternational 
      ? internationalPrefixes[Math.floor(Math.random() * internationalPrefixes.length)]
      : msisdnPrefixes[Math.floor(Math.random() * msisdnPrefixes.length)];

    // Wangiri fraud = very short calls (1-3 seconds)
    let duration = callType === 'VOICE' 
      ? Math.floor(Math.random() * 600) + 10 
      : 0;
    
    if (fraudType === 'WANGIRI') {
      duration = Math.floor(Math.random() * 3) + 1;
    }

    cdrs.push({
      id: `CDR-${String(100000 + i).padStart(6, '0')}`,
      callerMsisdn: `${callerPrefix}${Math.floor(Math.random() * 1000000).toString().padStart(6, '0')}`,
      calleeMsisdn: `${calleePrefix}${Math.floor(Math.random() * 10000000).toString().padStart(7, '0')}`,
      callType,
      duration,
      status,
      riskScore: Math.floor(risk * 100),
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 86400000 * 7)).toISOString(),
      cellTowerId: tower.id,
      cellTowerLocation: tower.location,
      imei: `${Math.floor(Math.random() * 100000000000000).toString().padStart(15, '0')}`,
      roaming: Math.random() > 0.85,
      fraudType,
      riskFactors: selectedRiskFactors,
      payload: {
        mcc: '604', // Morocco
        mnc: ['00', '01', '02'][Math.floor(Math.random() * 3)],
        lac: Math.floor(Math.random() * 10000).toString(),
        cellId: Math.floor(Math.random() * 50000).toString(),
        signalStrength: Math.floor(Math.random() * 30) - 110,
        sessionId: `sess_${Math.random().toString(36).substring(7)}`,
      },
    });
  }

  return cdrs.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
};

const mockCDRs = generateCDRs(100);

// Status badge component
function StatusBadge({ status, riskScore }: { status: CDR['status']; riskScore: number }) {
  const config = {
    approved: { variant: 'secondary' as const, className: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' },
    flagged: { variant: 'secondary' as const, className: 'bg-amber-500/10 text-amber-500 border-amber-500/20' },
    blocked: { variant: 'destructive' as const, className: 'bg-destructive/10 text-destructive border-destructive/20' },
    pending: { variant: 'outline' as const, className: 'bg-blue-500/10 text-blue-500 border-blue-500/20' },
  };
  
  return (
    <Badge variant={config[status].variant} className={cn('font-medium gap-1', config[status].className)}>
      {status === 'approved' && <CheckCircle className="h-3 w-3" />}
      {status === 'flagged' && <Flag className="h-3 w-3" />}
      {status === 'blocked' && <XCircle className="h-3 w-3" />}
      {status === 'pending' && <Clock className="h-3 w-3" />}
      {status.charAt(0).toUpperCase() + status.slice(1)}
      <span className="font-mono text-[10px] opacity-70">{riskScore}</span>
    </Badge>
  );
}

// Fraud type badge
function FraudBadge({ fraudType }: { fraudType: CDR['fraudType'] }) {
  if (fraudType === 'NONE') {
    return <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">Clean</Badge>;
  }
  
  const config: Record<string, string> = {
    WANGIRI: 'bg-orange-500/10 text-orange-500 border-orange-500/20',
    SIMBOX: 'bg-red-500/10 text-red-500 border-red-500/20',
    IRSF: 'bg-purple-500/10 text-purple-500 border-purple-500/20',
    PBX_HACKING: 'bg-pink-500/10 text-pink-500 border-pink-500/20',
  };
  
  return (
    <Badge variant="destructive" className={cn('font-medium', config[fraudType])}>
      {fraudType.replace('_', ' ')}
    </Badge>
  );
}

// Risk score indicator
function RiskIndicator({ score }: { score: number }) {
  const getColor = () => {
    if (score >= 80) return 'bg-destructive';
    if (score >= 60) return 'bg-orange-500';
    if (score >= 40) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-16 rounded-full bg-muted overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${score}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={cn('h-full rounded-full', getColor())}
        />
      </div>
      <span className="font-mono text-xs font-medium">{score}</span>
    </div>
  );
}

// CDR Detail Drawer
function CDRDrawer({ 
  cdr, 
  open, 
  onOpenChange 
}: { 
  cdr: CDR | null; 
  open: boolean; 
  onOpenChange: (open: boolean) => void;
}) {
  if (!cdr) return null;

  const formatDate = (date: string) => {
    return new Date(date).toLocaleString('en-US', {
      dateStyle: 'medium',
      timeStyle: 'medium',
    });
  };

  const formatDuration = (seconds: number) => {
    if (seconds === 0) return 'N/A (SMS)';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[500px] sm:w-[600px] p-0">
        <SheetHeader className="p-6 pb-4 border-b border-border/50">
          <div className="flex items-center justify-between">
            <div>
              <SheetTitle className="flex items-center gap-2 text-lg font-mono">
                {cdr.id}
                <StatusBadge status={cdr.status} riskScore={cdr.riskScore} />
              </SheetTitle>
              <SheetDescription className="mt-1">
                {formatDate(cdr.timestamp)}
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        <Tabs defaultValue="overview" className="flex-1">
          <div className="px-6 pt-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
              <TabsTrigger value="payload">Raw Payload</TabsTrigger>
            </TabsList>
          </div>

          <ScrollArea className="h-[calc(100vh-200px)]">
            <TabsContent value="overview" className="p-6 space-y-6 mt-0">
              {/* Call Info Card */}
              <Card className="bg-primary/5 border-primary/20">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Call Duration</p>
                      <p className="text-3xl font-bold">{formatDuration(cdr.duration)}</p>
                      <Badge variant="outline" className="mt-2">{cdr.callType}</Badge>
                    </div>
                    <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                      {cdr.callType === 'VOICE' ? (
                        <PhoneCall className="h-6 w-6 text-primary" />
                      ) : (
                        <Phone className="h-6 w-6 text-primary" />
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Details Grid */}
              <div className="grid gap-4">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <PhoneOutgoing className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Caller MSISDN</p>
                    <p className="text-sm font-medium font-mono">{cdr.callerMsisdn}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <Phone className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Callee MSISDN</p>
                    <p className="text-sm font-medium font-mono">{cdr.calleeMsisdn}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <Smartphone className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">IMEI</p>
                    <p className="text-sm font-medium font-mono">{cdr.imei}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <Radio className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Cell Tower</p>
                    <p className="text-sm font-medium">{cdr.cellTowerId}</p>
                    <p className="text-xs text-muted-foreground">{cdr.cellTowerLocation}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Roaming Status</p>
                    <p className="text-sm font-medium">{cdr.roaming ? 'Yes - International Roaming' : 'No - Domestic'}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs text-muted-foreground">Fraud Classification</p>
                    <FraudBadge fraudType={cdr.fraudType} />
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="risk" className="p-6 space-y-6 mt-0">
              {/* Risk Score Card */}
              <Card className={cn(
                'border-2',
                cdr.riskScore >= 80 && 'border-destructive/50 bg-destructive/5',
                cdr.riskScore >= 60 && cdr.riskScore < 80 && 'border-orange-500/50 bg-orange-500/5',
                cdr.riskScore >= 40 && cdr.riskScore < 60 && 'border-amber-500/50 bg-amber-500/5',
                cdr.riskScore < 40 && 'border-emerald-500/50 bg-emerald-500/5',
              )}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Risk Score</p>
                      <p className="text-4xl font-bold">{cdr.riskScore}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {cdr.riskScore >= 80 && 'Critical Risk - Likely Fraud'}
                        {cdr.riskScore >= 60 && cdr.riskScore < 80 && 'High Risk - Suspicious Activity'}
                        {cdr.riskScore >= 40 && cdr.riskScore < 60 && 'Medium Risk - Review Recommended'}
                        {cdr.riskScore < 40 && 'Low Risk - Normal Activity'}
                      </p>
                    </div>
                    <div className={cn(
                      'h-16 w-16 rounded-full flex items-center justify-center',
                      cdr.riskScore >= 80 && 'bg-destructive/20',
                      cdr.riskScore >= 60 && cdr.riskScore < 80 && 'bg-orange-500/20',
                      cdr.riskScore >= 40 && cdr.riskScore < 60 && 'bg-amber-500/20',
                      cdr.riskScore < 40 && 'bg-emerald-500/20',
                    )}>
                      <Shield className={cn(
                        'h-8 w-8',
                        cdr.riskScore >= 80 && 'text-destructive',
                        cdr.riskScore >= 60 && cdr.riskScore < 80 && 'text-orange-500',
                        cdr.riskScore >= 40 && cdr.riskScore < 60 && 'text-amber-500',
                        cdr.riskScore < 40 && 'text-emerald-500',
                      )} />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Risk Factors */}
              <div>
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                  Risk Factors ({cdr.riskFactors.length})
                </h4>
                {cdr.riskFactors.length > 0 ? (
                  <div className="space-y-2">
                    {cdr.riskFactors.map((factor, index) => (
                      <motion.div
                        key={factor}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-center gap-2 p-2 rounded-lg bg-amber-500/10 border border-amber-500/20"
                      >
                        <div className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                        <span className="text-sm">{factor}</span>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No risk factors identified</p>
                )}
              </div>

              {/* Actions */}
              <Separator />
              <div className="flex gap-2">
                <Button variant="outline" className="flex-1">
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Approve
                </Button>
                <Button variant="outline" className="flex-1">
                  <Flag className="mr-2 h-4 w-4" />
                  Flag
                </Button>
                <Button variant="destructive" className="flex-1">
                  <XCircle className="mr-2 h-4 w-4" />
                  Block
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="payload" className="p-6 mt-0">
              <div className="rounded-lg bg-muted/50 p-4 overflow-x-auto">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {JSON.stringify({ ...cdr, payload: cdr.payload }, null, 2)}
                </pre>
              </div>
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

// Main component
export default function CDRExplorer() {
  const [data] = useState<CDR[]>(mockCDRs);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [rowSelection, setRowSelection] = useState({});
  const [globalFilter, setGlobalFilter] = useState('');
  const [selectedCDR, setSelectedCDR] = useState<CDR | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const formatDuration = (seconds: number) => {
    if (seconds === 0) return '-';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  const columns: ColumnDef<CDR>[] = useMemo(() => [
    {
      accessorKey: 'id',
      header: 'CDR ID',
      cell: ({ row }) => (
        <span className="font-mono text-sm">{row.getValue('id')}</span>
      ),
    },
    {
      accessorKey: 'timestamp',
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
          className="-ml-4"
        >
          Timestamp
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => {
        const date = new Date(row.getValue('timestamp'));
        return (
          <div className="text-sm">
            <p>{date.toLocaleDateString()}</p>
            <p className="text-xs text-muted-foreground">{date.toLocaleTimeString()}</p>
          </div>
        );
      },
    },
    {
      accessorKey: 'callerMsisdn',
      header: 'Caller',
      cell: ({ row }) => (
        <span className="font-mono text-sm">{row.getValue('callerMsisdn')}</span>
      ),
    },
    {
      accessorKey: 'calleeMsisdn',
      header: 'Callee',
      cell: ({ row }) => (
        <span className="font-mono text-sm">{row.getValue('calleeMsisdn')}</span>
      ),
    },
    {
      accessorKey: 'callType',
      header: 'Type',
      cell: ({ row }) => (
        <Badge variant="outline" className="capitalize">
          {row.getValue('callType')}
        </Badge>
      ),
    },
    {
      accessorKey: 'duration',
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
          className="-ml-4"
        >
          Duration
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => (
        <div className="flex items-center gap-1.5">
          <Timer className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-sm font-mono">{formatDuration(row.getValue('duration'))}</span>
        </div>
      ),
    },
    {
      accessorKey: 'imei',
      header: 'IMEI',
      cell: ({ row }) => (
        <div className="max-w-[120px] truncate">
          <span className="font-mono text-xs">{row.getValue('imei')}</span>
        </div>
      ),
    },
    {
      accessorKey: 'cellTowerLocation',
      header: 'Location',
      cell: ({ row }) => (
        <div className="flex items-center gap-1.5">
          <MapPin className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-sm">{row.getValue('cellTowerLocation')}</span>
        </div>
      ),
    },
    {
      accessorKey: 'fraudType',
      header: 'Fraud Type',
      cell: ({ row }) => <FraudBadge fraudType={row.getValue('fraudType')} />,
    },
    {
      accessorKey: 'riskScore',
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
          className="-ml-4"
        >
          Risk
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => <RiskIndicator score={row.getValue('riskScore')} />,
    },
    {
      accessorKey: 'status',
      header: 'Status',
      cell: ({ row }) => (
        <StatusBadge 
          status={row.getValue('status')} 
          riskScore={row.original.riskScore} 
        />
      ),
    },
    {
      id: 'actions',
      cell: ({ row }) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem onClick={() => {
              setSelectedCDR(row.original);
              setDrawerOpen(true);
            }}>
              <Eye className="mr-2 h-4 w-4" />
              View Details
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              <CheckCircle className="mr-2 h-4 w-4" />
              Approve
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Flag className="mr-2 h-4 w-4" />
              Flag for Review
            </DropdownMenuItem>
            <DropdownMenuItem className="text-destructive">
              <XCircle className="mr-2 h-4 w-4" />
              Block
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
    },
  ], []);

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
      globalFilter,
    },
    globalFilterFn: 'includesString',
    onGlobalFilterChange: setGlobalFilter,
  });

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">CDR Explorer</h1>
          <p className="text-muted-foreground">
            Search and analyze Call Detail Records
          </p>
        </div>
        <Button variant="outline" size="sm">
          <Download className="mr-2 h-4 w-4" />
          Export
        </Button>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search by MSISDN, IMEI, Cell Tower..."
            value={globalFilter ?? ''}
            onChange={(e) => setGlobalFilter(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuLabel>Filter by Status</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => table.getColumn('status')?.setFilterValue('approved')}>
              Approved
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => table.getColumn('status')?.setFilterValue('flagged')}>
              Flagged
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => table.getColumn('status')?.setFilterValue('blocked')}>
              Blocked
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => table.getColumn('status')?.setFilterValue('pending')}>
              Pending
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuLabel>Filter by Fraud Type</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => table.getColumn('fraudType')?.setFilterValue('WANGIRI')}>
              Wangiri
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => table.getColumn('fraudType')?.setFilterValue('SIMBOX')}>
              SIM Box
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => table.getColumn('fraudType')?.setFilterValue('IRSF')}>
              IRSF
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => {
              table.getColumn('status')?.setFilterValue(undefined);
              table.getColumn('fraudType')?.setFilterValue(undefined);
            }}>
              Clear Filters
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm">
              <Settings2 className="mr-2 h-4 w-4" />
              Columns
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {table
              .getAllColumns()
              .filter((column) => column.getCanHide())
              .map((column) => (
                <DropdownMenuCheckboxItem
                  key={column.id}
                  className="capitalize"
                  checked={column.getIsVisible()}
                  onCheckedChange={(value) => column.toggleVisibility(!!value)}
                >
                  {column.id}
                </DropdownMenuCheckboxItem>
              ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Table */}
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(header.column.columnDef.header, header.getContext())}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              <AnimatePresence>
                {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row, index) => (
                    <motion.tr
                      key={row.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: index * 0.02 }}
                      className={cn(
                        'border-b border-border/50 transition-colors hover:bg-muted/30 cursor-pointer',
                        row.getIsSelected() && 'bg-muted'
                      )}
                      onClick={() => {
                        setSelectedCDR(row.original);
                        setDrawerOpen(true);
                      }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell key={cell.id}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      ))}
                    </motion.tr>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={columns.length} className="h-24 text-center">
                      No CDRs found.
                    </TableCell>
                  </TableRow>
                )}
              </AnimatePresence>
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {table.getFilteredSelectedRowModel().rows.length} of{' '}
          {table.getFilteredRowModel().rows.length} CDR(s) selected.
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.setPageIndex(0)}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.setPageIndex(table.getPageCount() - 1)}
            disabled={!table.getCanNextPage()}
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* CDR Drawer */}
      <CDRDrawer
        cdr={selectedCDR}
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
      />
    </div>
  );
}
