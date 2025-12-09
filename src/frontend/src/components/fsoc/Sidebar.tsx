'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Activity,
  Search,
  Network,
  Settings,
  Shield,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  FileText,
  AlertTriangle,
  Bell,
  Database,
  Zap,
} from 'lucide-react';

interface NavItem {
  title: string;
  href: string;
  icon: React.ElementType;
  badge?: number;
  section: 'main' | 'analysis' | 'system';
}

const navItems: NavItem[] = [
  // Main Operations
  { title: 'Pulse', href: '/dashboard', icon: Activity, section: 'main' },
  { title: 'Explorer', href: '/transactions', icon: Search, section: 'main' },
  { title: 'Alerts', href: '/alerts', icon: AlertTriangle, badge: 7, section: 'main' },
  
  // Analysis
  { title: 'Graph View', href: '/graph', icon: Network, section: 'analysis' },
  { title: 'Reports', href: '/reports', icon: FileText, section: 'analysis' },
  { title: 'Risk Models', href: '/models', icon: Zap, section: 'analysis' },
  
  // System
  { title: 'Data Sources', href: '/sources', icon: Database, section: 'system' },
  { title: 'Settings', href: '/settings', icon: Settings, section: 'system' },
];

const sectionLabels = {
  main: 'Operations',
  analysis: 'Analysis',
  system: 'System',
};

export function FSOCSidebar() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Keyboard shortcut to toggle sidebar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '[' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsCollapsed((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const renderNavItems = (section: 'main' | 'analysis' | 'system') => {
    const items = navItems.filter((item) => item.section === section);
    
    return items.map((item) => {
      const isActive = pathname === item.href || pathname.startsWith(`${item.href}/`);
      const Icon = item.icon;

      const navLink = (
        <Link href={item.href} key={item.href}>
          <motion.div
            whileHover={{ x: isCollapsed ? 0 : 4 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
              'group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
              'hover:bg-accent/50',
              isActive && 'bg-primary/10 text-primary',
              !isActive && 'text-muted-foreground hover:text-foreground',
              isCollapsed && 'justify-center px-2'
            )}
          >
            {/* Active indicator */}
            {isActive && (
              <motion.div
                layoutId="activeNav"
                className="absolute left-0 top-1/2 h-6 w-1 -translate-y-1/2 rounded-r-full bg-primary"
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              />
            )}
            
            <Icon className={cn(
              'h-5 w-5 shrink-0 transition-colors',
              isActive ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
            )} />
            
            <AnimatePresence>
              {!isCollapsed && (
                <motion.span
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  transition={{ duration: 0.15 }}
                  className="overflow-hidden whitespace-nowrap"
                >
                  {item.title}
                </motion.span>
              )}
            </AnimatePresence>

            {/* Badge */}
            {item.badge && !isCollapsed && (
              <span className="ml-auto flex h-5 min-w-5 items-center justify-center rounded-full bg-destructive px-1.5 text-[10px] font-semibold text-destructive-foreground">
                {item.badge}
              </span>
            )}
            {item.badge && isCollapsed && (
              <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[9px] font-semibold text-destructive-foreground">
                {item.badge > 9 ? '9+' : item.badge}
              </span>
            )}
          </motion.div>
        </Link>
      );

      if (isCollapsed) {
        return (
          <TooltipProvider key={item.href} delayDuration={0}>
            <Tooltip>
              <TooltipTrigger asChild>{navLink}</TooltipTrigger>
              <TooltipContent side="right" className="flex items-center gap-2">
                {item.title}
                {item.badge && (
                  <span className="rounded-full bg-destructive px-1.5 py-0.5 text-[10px] text-destructive-foreground">
                    {item.badge}
                  </span>
                )}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      }

      return navLink;
    });
  };

  return (
    <>
      <motion.aside
        animate={{ width: isCollapsed ? 72 : 256 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className={cn(
          'fixed left-0 top-0 z-40 flex h-screen flex-col border-r border-border/50 bg-card/50 backdrop-blur-xl',
        )}
      >
        {/* Logo */}
        <div className={cn(
          'flex h-16 items-center border-b border-border/50 px-4',
          isCollapsed ? 'justify-center' : 'gap-3'
        )}>
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-lg shadow-primary/25">
            <Shield className="h-5 w-5 text-primary-foreground" />
          </div>
          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
              >
                <span className="font-bold text-lg tracking-tight">FSOC</span>
                <span className="block text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                  Security Center
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Navigation */}
        <ScrollArea className="flex-1 px-3 py-4">
          <div className="space-y-6">
            {(['main', 'analysis', 'system'] as const).map((section) => (
              <div key={section}>
                {!isCollapsed && (
                  <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70">
                    {sectionLabels[section]}
                  </p>
                )}
                {isCollapsed && section !== 'main' && (
                  <Separator className="my-2" />
                )}
                <div className="space-y-1">
                  {renderNavItems(section)}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Status & Collapse */}
        <div className="border-t border-border/50 p-3">
          {!isCollapsed && (
            <div className="mb-3 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
              <span className="text-xs font-medium text-emerald-500">All Systems Operational</span>
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={cn('w-full', isCollapsed && 'px-2')}
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <>
                <ChevronLeft className="mr-2 h-4 w-4" />
                <span className="text-xs">Collapse</span>
                <kbd className="ml-auto rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono">⌘[</kbd>
              </>
            )}
          </Button>
        </div>
      </motion.aside>

      {/* Spacer */}
      <motion.div
        animate={{ width: isCollapsed ? 72 : 256 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        className="shrink-0"
      />
    </>
  );
}
