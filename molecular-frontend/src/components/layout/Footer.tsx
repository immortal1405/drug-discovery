import React from 'react';
import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="py-6 md:px-8 md:py-0 border-t border-border/40">
      <div className="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
        <p className="text-balance text-center text-sm leading-loose text-muted-foreground md:text-left">
          © {new Date().getFullYear()} Molecular AI Platform. Built with Next.js & Love.
          {/* Optional: Add links or other info */}
        </p>
        {/* Optional: Add social links or other footer nav */}
        {/* <div className="flex space-x-4">
          <Link href="/about">About</Link>
          <Link href="/contact">Contact</Link>
        </div> */}
      </div>
    </footer>
  );
} 