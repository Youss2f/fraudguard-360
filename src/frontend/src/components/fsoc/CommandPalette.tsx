'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from '@/components/ui/command';
import {
  Activity,
  Search,
  Network,
  Settings,
  FileText,
  AlertTriangle,
  Database,
  Zap,
  Moon,
  Sun,
  User,
  Calculator,
  ArrowRight,
  Shield,
} from 'lucide-react';
import { useTheme } from 'next-themes';

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const navigationItems = [
  { title: 'Pulse Dashboard', href: '/dashboard', icon: Activity, shortcut: '⇧D' },
  { title: 'CDR Explorer', href: '/transactions', icon: Search, shortcut: '⇧T' },
  { title: 'Alerts Center', href: '/alerts', icon: AlertTriangle, shortcut: '⇧A' },
  { title: 'Graph Analysis', href: '/graph', icon: Network, shortcut: '⇧G' },
  { title: 'Reports', href: '/reports', icon: FileText, shortcut: '⇧R' },
  { title: 'Risk Models', href: '/models', icon: Zap },
  { title: 'Data Sources', href: '/sources', icon: Database },
  { title: 'Settings', href: '/settings', icon: Settings, shortcut: '⇧S' },
];

const quickActions = [
  { title: 'Search CDR by ID', action: 'searchCDR', icon: Search },
  { title: 'Run Risk Assessment', action: 'riskAssess', icon: Calculator },
  { title: 'View High-Risk Queue', action: 'highRisk', icon: AlertTriangle },
  { title: 'Export Report', action: 'export', icon: FileText },
];

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const { setTheme } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');

  // Global keyboard shortcut
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, [open, onOpenChange]);

  const runCommand = useCallback((command: () => void) => {
    onOpenChange(false);
    command();
  }, [onOpenChange]);

  // Check if the search looks like a CDR ID
  const isCDRSearch = searchQuery.match(/^(CDR[-_]?\d+|\d{6,})$/i);

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Search commands, pages, or CDR IDs..."
        value={searchQuery}
        onValueChange={setSearchQuery}
      />
      <CommandList>
        <CommandEmpty>
          {isCDRSearch ? (
            <div className="flex flex-col items-center gap-2 py-6">
              <Search className="h-10 w-10 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">
                Press Enter to search for CDR <span className="font-mono text-foreground">{searchQuery}</span>
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-6">
              <Shield className="h-10 w-10 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">No results found.</p>
            </div>
          )}
        </CommandEmpty>

        {/* CDR Search Shortcut */}
        {isCDRSearch && (
          <CommandGroup heading="CDR Search">
            <CommandItem
              onSelect={() => runCommand(() => router.push(`/transactions?search=${searchQuery}`))}
              className="gap-3"
            >
              <Search className="h-4 w-4" />
              <span>Search for CDR</span>
              <span className="ml-1 font-mono text-primary">{searchQuery}</span>
              <ArrowRight className="ml-auto h-4 w-4 text-muted-foreground" />
            </CommandItem>
          </CommandGroup>
        )}

        {/* Navigation */}
        <CommandGroup heading="Navigation">
          {navigationItems.map((item) => (
            <CommandItem
              key={item.href}
              onSelect={() => runCommand(() => router.push(item.href))}
              className="gap-3"
            >
              <item.icon className="h-4 w-4" />
              <span>{item.title}</span>
              {item.shortcut && (
                <CommandShortcut>{item.shortcut}</CommandShortcut>
              )}
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        {/* Quick Actions */}
        <CommandGroup heading="Quick Actions">
          {quickActions.map((action) => (
            <CommandItem
              key={action.action}
              onSelect={() => runCommand(() => {
                // Handle quick actions
                switch (action.action) {
                  case 'searchTxn':
                    router.push('/transactions');
                    break;
                  case 'highRisk':
                    router.push('/transactions?risk=high');
                    break;
                  case 'riskAssess':
                    router.push('/models');
                    break;
                  case 'export':
                    router.push('/reports');
                    break;
                }
              })}
              className="gap-3"
            >
              <action.icon className="h-4 w-4" />
              <span>{action.title}</span>
            </CommandItem>
          ))}
        </CommandGroup>

        <CommandSeparator />

        {/* Theme */}
        <CommandGroup heading="Appearance">
          <CommandItem onSelect={() => runCommand(() => setTheme('light'))} className="gap-3">
            <Sun className="h-4 w-4" />
            <span>Light Mode</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setTheme('dark'))} className="gap-3">
            <Moon className="h-4 w-4" />
            <span>Dark Mode</span>
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        {/* Account */}
        <CommandGroup heading="Account">
          <CommandItem onSelect={() => runCommand(() => router.push('/settings'))} className="gap-3">
            <User className="h-4 w-4" />
            <span>Profile Settings</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
