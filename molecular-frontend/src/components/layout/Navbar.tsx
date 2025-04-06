"use client";

import * as React from "react";
import Link from "next/link";

import { cn } from "@/lib/utils";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Button } from "@/components/ui/button";
// import { ModeToggle } from "@/components/layout/mode-toggle"; // Add later if needed

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center">
        <Link href="/" className="mr-6 flex items-center space-x-2">
          {/* <Icons.logo className="h-6 w-6" /> // Replace with actual logo */} 
          <span className="font-bold sm:inline-block">
            Molecular AI
          </span>
        </Link>
        <nav className="flex flex-1 items-center justify-end space-x-2">
          {/* Simple Nav for now, replace/enhance with NavigationMenu if needed */}
          <Link href="/generate" legacyBehavior passHref>
            <Button variant="ghost">Generate</Button>
          </Link>
          <Link href="/results" legacyBehavior passHref>
            <Button variant="ghost">Results</Button>
          </Link>
          {/* Add Auth buttons later */}
          {/* <ModeToggle /> */}
          <Link href="/sign-in" legacyBehavior passHref>
             <Button variant="outline">Sign In</Button>
          </Link>
        </nav>
      </div>
    </header>
  );
} 