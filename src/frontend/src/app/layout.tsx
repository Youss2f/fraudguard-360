import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import { Providers } from '@/components/providers';
import './globals.css';

const inter = Inter({
  variable: '--font-sans',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: {
    default: 'FraudGuard-360 | Enterprise Fraud Detection',
    template: '%s | FraudGuard-360',
  },
  description: 'Real-time fraud detection and prevention platform powered by advanced ML models. Monitor transactions, detect anomalies, and protect your business.',
  keywords: ['fraud detection', 'machine learning', 'risk scoring', 'transaction monitoring', 'fintech'],
  authors: [{ name: 'FraudGuard Team' }],
  creator: 'FraudGuard-360',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://fraudguard360.com',
    siteName: 'FraudGuard-360',
    title: 'FraudGuard-360 | Enterprise Fraud Detection',
    description: 'Real-time fraud detection and prevention platform powered by advanced ML models.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'FraudGuard-360 | Enterprise Fraud Detection',
    description: 'Real-time fraud detection and prevention platform powered by advanced ML models.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} font-sans antialiased min-h-screen bg-background`}
      >
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
