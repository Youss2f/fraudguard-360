'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  FileText,
  Download,
  Calendar,
  TrendingUp,
  Shield,
  BarChart3,
  PieChart,
  Clock,
} from 'lucide-react';

const reports = [
  {
    id: 'RPT-001',
    title: 'Weekly Fraud Summary',
    description: 'Comprehensive overview of fraud detection activities',
    type: 'Scheduled',
    lastGenerated: '2024-01-15T10:00:00Z',
    frequency: 'Weekly',
  },
  {
    id: 'RPT-002',
    title: 'Monthly Risk Analysis',
    description: 'Deep dive into risk patterns and model performance',
    type: 'Scheduled',
    lastGenerated: '2024-01-01T00:00:00Z',
    frequency: 'Monthly',
  },
  {
    id: 'RPT-003',
    title: 'Real-time Alert Report',
    description: 'Live summary of all active alerts and responses',
    type: 'On-demand',
    lastGenerated: '2024-01-15T14:30:00Z',
    frequency: 'On-demand',
  },
  {
    id: 'RPT-004',
    title: 'Model Performance Metrics',
    description: 'ML model accuracy, precision, and recall statistics',
    type: 'Scheduled',
    lastGenerated: '2024-01-14T00:00:00Z',
    frequency: 'Daily',
  },
];

export default function ReportsPage() {
  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Reports</h1>
          <p className="text-muted-foreground">
            Generate and download fraud analytics reports
          </p>
        </div>
        <Button>
          <FileText className="mr-2 h-4 w-4" />
          Create Report
        </Button>
      </div>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Reports</p>
                <p className="text-xl font-bold">{reports.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <Calendar className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Scheduled</p>
                <p className="text-xl font-bold">{reports.filter(r => r.type === 'Scheduled').length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <Clock className="h-5 w-5 text-amber-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Generated Today</p>
                <p className="text-xl font-bold">2</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Download className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Downloads</p>
                <p className="text-xl font-bold">156</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Report List */}
      <div className="grid gap-4 md:grid-cols-2">
        {reports.map((report) => (
          <Card key={report.id} className="hover:bg-muted/30 transition-colors cursor-pointer">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-lg">{report.title}</CardTitle>
                  <CardDescription className="mt-1">{report.description}</CardDescription>
                </div>
                <Badge variant="outline">{report.type}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3.5 w-3.5" />
                    {report.frequency}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5" />
                    {new Date(report.lastGenerated).toLocaleDateString()}
                  </span>
                </div>
                <Button size="sm" variant="outline">
                  <Download className="mr-2 h-3.5 w-3.5" />
                  Download
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
