'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { toast } from 'sonner';
import {
  User,
  Bell,
  Shield,
  Key,
  Database,
  Zap,
  Globe,
  Monitor,
  Moon,
  Sun,
  Save,
  RefreshCcw,
  CheckCircle,
  AlertTriangle,
  Mail,
  Smartphone,
} from 'lucide-react';

interface SettingSection {
  id: string;
  label: string;
  icon: React.ElementType;
}

const sections: SettingSection[] = [
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'api', label: 'API & Integrations', icon: Key },
  { id: 'detection', label: 'Detection Settings', icon: Zap },
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('profile');
  const [saving, setSaving] = useState(false);

  // Form states
  const [settings, setSettings] = useState({
    // Profile
    name: 'Security Analyst',
    email: 'analyst@fraudguard.io',
    role: 'analyst',
    timezone: 'America/New_York',
    
    // Notifications
    emailNotifications: true,
    pushNotifications: true,
    highRiskAlerts: true,
    dailyDigest: true,
    weeklyReport: false,
    
    // Security
    twoFactorEnabled: true,
    sessionTimeout: '30',
    ipWhitelisting: false,
    
    // API
    apiKey: 'fg_live_xxxxxxxxxxxxxxxxxx',
    webhookUrl: 'https://example.com/webhook',
    
    // Detection
    riskThreshold: '70',
    autoBlock: true,
    autoBlockThreshold: '90',
    velocityChecks: true,
    geoAnomalyDetection: true,
    deviceFingerprinting: true,
  });

  const handleSave = async () => {
    setSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSaving(false);
    toast.success('Settings saved successfully');
  };

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Sidebar */}
      <div className="w-64 border-r border-border/50 bg-card/30 p-4">
        <h2 className="font-semibold mb-4">Settings</h2>
        <nav className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveTab(section.id)}
              className={cn(
                'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                activeTab === section.id
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <section.icon className="h-4 w-4" />
              {section.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto p-6 space-y-6">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div>
                <h1 className="text-2xl font-bold">Profile Settings</h1>
                <p className="text-muted-foreground">
                  Manage your account information and preferences
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Personal Information</CardTitle>
                  <CardDescription>Update your profile details</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Avatar */}
                  <div className="flex items-center gap-4">
                    <Avatar className="h-20 w-20">
                      <AvatarImage src="/avatars/analyst.png" />
                      <AvatarFallback className="text-lg">YS</AvatarFallback>
                    </Avatar>
                    <div>
                      <Button variant="outline" size="sm">Change Avatar</Button>
                      <p className="text-xs text-muted-foreground mt-1">
                        JPG, PNG or GIF. Max 2MB.
                      </p>
                    </div>
                  </div>

                  <Separator />

                  {/* Name */}
                  <div className="grid gap-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      value={settings.name}
                      onChange={(e) => updateSetting('name', e.target.value)}
                    />
                  </div>

                  {/* Email */}
                  <div className="grid gap-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      value={settings.email}
                      onChange={(e) => updateSetting('email', e.target.value)}
                    />
                  </div>

                  {/* Role */}
                  <div className="grid gap-2">
                    <Label>Role</Label>
                    <Select value={settings.role} onValueChange={(v) => updateSetting('role', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="admin">Administrator</SelectItem>
                        <SelectItem value="analyst">Security Analyst</SelectItem>
                        <SelectItem value="viewer">Viewer</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Timezone */}
                  <div className="grid gap-2">
                    <Label>Timezone</Label>
                    <Select value={settings.timezone} onValueChange={(v) => updateSetting('timezone', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="America/New_York">Eastern Time (ET)</SelectItem>
                        <SelectItem value="America/Chicago">Central Time (CT)</SelectItem>
                        <SelectItem value="America/Denver">Mountain Time (MT)</SelectItem>
                        <SelectItem value="America/Los_Angeles">Pacific Time (PT)</SelectItem>
                        <SelectItem value="Europe/London">GMT</SelectItem>
                        <SelectItem value="Europe/Paris">Central European Time</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Notifications Tab */}
          {activeTab === 'notifications' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div>
                <h1 className="text-2xl font-bold">Notifications</h1>
                <p className="text-muted-foreground">
                  Configure how you receive alerts and updates
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Notification Channels</CardTitle>
                  <CardDescription>Choose how you want to be notified</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Mail className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="font-medium">Email Notifications</p>
                        <p className="text-sm text-muted-foreground">
                          Receive alerts via email
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.emailNotifications}
                      onCheckedChange={(v) => updateSetting('emailNotifications', v)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Smartphone className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="font-medium">Push Notifications</p>
                        <p className="text-sm text-muted-foreground">
                          Receive push notifications on your devices
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.pushNotifications}
                      onCheckedChange={(v) => updateSetting('pushNotifications', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Alert Types</CardTitle>
                  <CardDescription>Choose which alerts you want to receive</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">High-Risk Alerts</p>
                      <p className="text-sm text-muted-foreground">
                        Immediate alerts for critical risk transactions
                      </p>
                    </div>
                    <Switch
                      checked={settings.highRiskAlerts}
                      onCheckedChange={(v) => updateSetting('highRiskAlerts', v)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Daily Digest</p>
                      <p className="text-sm text-muted-foreground">
                        Summary of daily activity
                      </p>
                    </div>
                    <Switch
                      checked={settings.dailyDigest}
                      onCheckedChange={(v) => updateSetting('dailyDigest', v)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Weekly Report</p>
                      <p className="text-sm text-muted-foreground">
                        Comprehensive weekly analytics report
                      </p>
                    </div>
                    <Switch
                      checked={settings.weeklyReport}
                      onCheckedChange={(v) => updateSetting('weeklyReport', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div>
                <h1 className="text-2xl font-bold">Security</h1>
                <p className="text-muted-foreground">
                  Manage your account security settings
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Two-Factor Authentication</CardTitle>
                  <CardDescription>Add an extra layer of security to your account</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        'h-10 w-10 rounded-full flex items-center justify-center',
                        settings.twoFactorEnabled ? 'bg-emerald-500/10' : 'bg-muted'
                      )}>
                        <Shield className={cn(
                          'h-5 w-5',
                          settings.twoFactorEnabled ? 'text-emerald-500' : 'text-muted-foreground'
                        )} />
                      </div>
                      <div>
                        <p className="font-medium">
                          {settings.twoFactorEnabled ? 'Enabled' : 'Disabled'}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {settings.twoFactorEnabled 
                            ? 'Your account is protected with 2FA'
                            : 'Enable 2FA for enhanced security'}
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={settings.twoFactorEnabled}
                      onCheckedChange={(v) => updateSetting('twoFactorEnabled', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Session Settings</CardTitle>
                  <CardDescription>Configure session timeout and restrictions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-2">
                    <Label>Session Timeout</Label>
                    <Select value={settings.sessionTimeout} onValueChange={(v) => updateSetting('sessionTimeout', v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="15">15 minutes</SelectItem>
                        <SelectItem value="30">30 minutes</SelectItem>
                        <SelectItem value="60">1 hour</SelectItem>
                        <SelectItem value="120">2 hours</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">IP Whitelisting</p>
                      <p className="text-sm text-muted-foreground">
                        Restrict access to specific IP addresses
                      </p>
                    </div>
                    <Switch
                      checked={settings.ipWhitelisting}
                      onCheckedChange={(v) => updateSetting('ipWhitelisting', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* API Tab */}
          {activeTab === 'api' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div>
                <h1 className="text-2xl font-bold">API & Integrations</h1>
                <p className="text-muted-foreground">
                  Manage API keys and external integrations
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>API Key</CardTitle>
                  <CardDescription>Your API key for programmatic access</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      value={settings.apiKey}
                      readOnly
                      className="font-mono text-sm"
                    />
                    <Button variant="outline">Copy</Button>
                    <Button variant="outline">
                      <RefreshCcw className="h-4 w-4" />
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Keep this key secret. Never share it publicly.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Webhooks</CardTitle>
                  <CardDescription>Configure webhook endpoints for real-time events</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-2">
                    <Label>Webhook URL</Label>
                    <Input
                      value={settings.webhookUrl}
                      onChange={(e) => updateSetting('webhookUrl', e.target.value)}
                      placeholder="https://your-server.com/webhook"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-emerald-500" />
                    <span className="text-sm text-emerald-500">Webhook active and verified</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Detection Settings Tab */}
          {activeTab === 'detection' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div>
                <h1 className="text-2xl font-bold">Detection Settings</h1>
                <p className="text-muted-foreground">
                  Configure fraud detection parameters
                </p>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Risk Thresholds</CardTitle>
                  <CardDescription>Set risk score thresholds for actions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-2">
                    <Label>Flagging Threshold</Label>
                    <div className="flex items-center gap-4">
                      <Input
                        type="number"
                        value={settings.riskThreshold}
                        onChange={(e) => updateSetting('riskThreshold', e.target.value)}
                        className="w-24"
                        min="0"
                        max="100"
                      />
                      <span className="text-sm text-muted-foreground">
                        Transactions above this score will be flagged
                      </span>
                    </div>
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Auto-Block High Risk</p>
                      <p className="text-sm text-muted-foreground">
                        Automatically block extremely high-risk transactions
                      </p>
                    </div>
                    <Switch
                      checked={settings.autoBlock}
                      onCheckedChange={(v) => updateSetting('autoBlock', v)}
                    />
                  </div>
                  {settings.autoBlock && (
                    <div className="grid gap-2 ml-6 pl-4 border-l-2 border-border">
                      <Label>Auto-Block Threshold</Label>
                      <div className="flex items-center gap-4">
                        <Input
                          type="number"
                          value={settings.autoBlockThreshold}
                          onChange={(e) => updateSetting('autoBlockThreshold', e.target.value)}
                          className="w-24"
                          min="0"
                          max="100"
                        />
                        <span className="text-sm text-muted-foreground">
                          Transactions above this score will be blocked
                        </span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Detection Features</CardTitle>
                  <CardDescription>Enable or disable detection methods</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Velocity Checks</p>
                      <p className="text-sm text-muted-foreground">
                        Detect unusual transaction frequency
                      </p>
                    </div>
                    <Switch
                      checked={settings.velocityChecks}
                      onCheckedChange={(v) => updateSetting('velocityChecks', v)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Geo-Anomaly Detection</p>
                      <p className="text-sm text-muted-foreground">
                        Flag transactions from unusual locations
                      </p>
                    </div>
                    <Switch
                      checked={settings.geoAnomalyDetection}
                      onCheckedChange={(v) => updateSetting('geoAnomalyDetection', v)}
                    />
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Device Fingerprinting</p>
                      <p className="text-sm text-muted-foreground">
                        Track and verify device identities
                      </p>
                    </div>
                    <Switch
                      checked={settings.deviceFingerprinting}
                      onCheckedChange={(v) => updateSetting('deviceFingerprinting', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Save Button */}
          <div className="flex justify-end pb-6">
            <Button onClick={handleSave} disabled={saving}>
              {saving ? (
                <>
                  <RefreshCcw className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save Changes
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
