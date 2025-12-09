'use client';

import { useState } from 'react';
import { FSOCSidebar, TopBar, CommandPalette } from '@/components/fsoc';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background">
      <FSOCSidebar />
      <main className="flex-1 flex flex-col min-h-screen">
        <TopBar onCommandPaletteOpen={() => setCommandPaletteOpen(true)} />
        <div className="flex-1 overflow-auto">
          {children}
        </div>
      </main>
      <CommandPalette 
        open={commandPaletteOpen} 
        onOpenChange={setCommandPaletteOpen} 
      />
    </div>
  );
}
