'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  LayoutDashboard,
  CreditCard,
  AlertTriangle,
  BarChart3,
  Settings,
  Users,
  Shield,
  Bell,
  ChevronLeft,
  ChevronRight,
  LogOut,
  User,
  Moon,
  Sun,
  Menu,
  Search,
  HelpCircle,
  Zap,
} from 'lucide-react';

interface NavItem {
  title: string;
  href: string;
  icon: React.ElementType;
  badge?: number;
  badgeVariant?: 'default' | 'destructive' | 'warning';
}

const mainNavItems: NavItem[] = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
  },
  {
    title: 'CDR Explorer',
    href: '/transactions',
    icon: CreditCard,
  },
  {
    title: 'Alerts',
    href: '/alerts',
    icon: AlertTriangle,
    badge: 12,
    badgeVariant: 'destructive',
  },
  {
    title: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
  },
  {
    title: 'Risk Models',
    href: '/risk-models',
    icon: Zap,
  },
];

const secondaryNavItems: NavItem[] = [
  {
    title: 'Users',
    href: '/users',
    icon: Users,
  },
  {
    title: 'Security',
    href: '/security',
    icon: Shield,
  },
  {
    title: 'Settings',
    href: '/settings',
    icon: Settings,
  },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  // Close mobile sidebar on route change
  useEffect(() => {
    setIsMobileOpen(false);
  }, [pathname]);

  // Handle escape key to close mobile sidebar
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsMobileOpen(false);
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const sidebarContent = (
    <>
      {/* Logo Section */}
      <div className={cn(
        'flex items-center h-16 px-4 border-b border-border/40',
        isCollapsed ? 'justify-center' : 'gap-3'
      )}>
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary">
          <Shield className="w-5 h-5 text-primary-foreground" />
        </div>
        <AnimatePresence>
          {!isCollapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <span className="font-semibold text-lg whitespace-nowrap">FraudGuard</span>
              <span className="text-xs text-muted-foreground block -mt-1">Enterprise</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Search Section */}
      <div className={cn('px-3 py-4', isCollapsed && 'px-2')}>
        <Button
          variant="outline"
          className={cn(
            'w-full justify-start text-muted-foreground hover:text-foreground',
            isCollapsed ? 'px-2' : 'px-3'
          )}
        >
          <Search className="h-4 w-4 shrink-0" />
          {!isCollapsed && <span className="ml-2 text-sm">Search...</span>}
          {!isCollapsed && (
            <kbd className="ml-auto pointer-events-none hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
              <span className="text-xs">⌘</span>K
            </kbd>
          )}
        </Button>
      </div>

      <Separator className="mx-3 w-auto" />

      {/* Main Navigation */}
      <ScrollArea className="flex-1 px-3 py-4">
        <div className="space-y-1">
          <p className={cn(
            'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2',
            isCollapsed && 'sr-only'
          )}>
            Main
          </p>
          {mainNavItems.map((item) => (
            <NavLink
              key={item.href}
              item={item}
              isActive={pathname === item.href || pathname.startsWith(`${item.href}/`)}
              isCollapsed={isCollapsed}
            />
          ))}
        </div>

        <Separator className="my-4" />

        <div className="space-y-1">
          <p className={cn(
            'text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2',
            isCollapsed && 'sr-only'
          )}>
            Administration
          </p>
          {secondaryNavItems.map((item) => (
            <NavLink
              key={item.href}
              item={item}
              isActive={pathname === item.href || pathname.startsWith(`${item.href}/`)}
              isCollapsed={isCollapsed}
            />
          ))}
        </div>
      </ScrollArea>

      <Separator />

      {/* User Section */}
      <div className={cn('p-3', isCollapsed && 'p-2')}>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                'w-full h-auto py-2',
                isCollapsed ? 'px-2 justify-center' : 'px-2 justify-start'
              )}
            >
              <Avatar className="h-8 w-8 shrink-0">
                <AvatarImage src="/avatars/user.png" alt="User" />
                <AvatarFallback className="bg-primary/10 text-primary text-sm font-medium">
                  JD
                </AvatarFallback>
              </Avatar>
              {!isCollapsed && (
                <div className="ml-2 text-left overflow-hidden">
                  <p className="text-sm font-medium truncate">John Doe</p>
                  <p className="text-xs text-muted-foreground truncate">Admin</p>
                </div>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align={isCollapsed ? 'center' : 'start'} className="w-56">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              <User className="mr-2 h-4 w-4" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Bell className="mr-2 h-4 w-4" />
              Notifications
            </DropdownMenuItem>
            <DropdownMenuItem>
              <HelpCircle className="mr-2 h-4 w-4" />
              Help & Support
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive focus:text-destructive">
              <LogOut className="mr-2 h-4 w-4" />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Collapse Toggle */}
      <div className="hidden lg:block absolute -right-3 top-20">
        <Button
          variant="outline"
          size="icon"
          className="h-6 w-6 rounded-full shadow-md bg-background"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          {isCollapsed ? (
            <ChevronRight className="h-3 w-3" />
          ) : (
            <ChevronLeft className="h-3 w-3" />
          )}
        </Button>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile Menu Button */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-3 left-3 z-50 lg:hidden"
        onClick={() => setIsMobileOpen(true)}
      >
        <Menu className="h-5 w-5" />
      </Button>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {isMobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm lg:hidden"
            onClick={() => setIsMobileOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isMobileOpen && (
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={cn(
              'fixed left-0 top-0 z-50 h-full w-[280px] bg-card border-r border-border lg:hidden',
              'flex flex-col'
            )}
          >
            {sidebarContent}
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <motion.aside
        animate={{ width: isCollapsed ? 72 : 280 }}
        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
        className={cn(
          'hidden lg:flex flex-col fixed left-0 top-0 h-full z-30',
          'bg-card border-r border-border relative',
          className
        )}
      >
        {sidebarContent}
      </motion.aside>

      {/* Spacer for main content */}
      <motion.div
        animate={{ width: isCollapsed ? 72 : 280 }}
        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
        className="hidden lg:block shrink-0"
      />
    </>
  );
}

interface NavLinkProps {
  item: NavItem;
  isActive: boolean;
  isCollapsed: boolean;
}

function NavLink({ item, isActive, isCollapsed }: NavLinkProps) {
  const Icon = item.icon;

  return (
    <Link href={item.href}>
      <motion.div
        whileHover={{ x: 2 }}
        whileTap={{ scale: 0.98 }}
        className={cn(
          'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all',
          'hover:bg-accent hover:text-accent-foreground',
          isActive && 'bg-primary/10 text-primary',
          !isActive && 'text-muted-foreground',
          isCollapsed && 'justify-center px-2'
        )}
      >
        <Icon className={cn('h-5 w-5 shrink-0', isActive && 'text-primary')} />
        <AnimatePresence>
          {!isCollapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden whitespace-nowrap"
            >
              {item.title}
            </motion.span>
          )}
        </AnimatePresence>
        {!isCollapsed && item.badge !== undefined && (
          <span
            className={cn(
              'ml-auto flex h-5 min-w-5 items-center justify-center rounded-full px-1.5 text-xs font-medium',
              item.badgeVariant === 'destructive' && 'bg-destructive text-destructive-foreground',
              item.badgeVariant === 'warning' && 'bg-yellow-500/20 text-yellow-600',
              !item.badgeVariant && 'bg-muted text-muted-foreground'
            )}
          >
            {item.badge}
          </span>
        )}
        {isCollapsed && item.badge !== undefined && (
          <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[10px] text-destructive-foreground">
            {item.badge > 9 ? '9+' : item.badge}
          </span>
        )}
      </motion.div>
    </Link>
  );
}

export default Sidebar;
