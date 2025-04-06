'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, CheckCircle, FlaskRound, Dna, Brain, ShieldCheck, Atom, TestTube, Beaker } from 'lucide-react';
import { cn } from "@/lib/utils";

// Loading Screen Component
const LoadingScreen = () => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-background">
    <FlaskRound className="h-16 w-16 animate-pulse text-primary" />
  </div>
);

// Floating Element Component
const FloatingElement = ({ icon: Icon, className, animation }: {
  icon: React.ElementType;
  className?: string;
  animation?: string;
}) => (
  // Ensure low opacity and pointer-events-none for background effect
  <div className={cn("absolute z-0 text-primary/5 pointer-events-none", className)}> 
    <Icon className={cn("h-full w-full", animation)} />
  </div>
);

export default function LandingPage() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      {isLoading && <LoadingScreen />}
      
      <div className={cn(
        "flex flex-col min-h-screen relative overflow-hidden", // Keep overflow-hidden
        isLoading ? 'opacity-0' : 'opacity-100 transition-opacity duration-500'
      )}>
        
        {/* --- Enhanced Floating Background Elements --- */} 
        {!isLoading && (
          <>
            {/* Larger, slower elements */}
            <FloatingElement icon={Dna} className="h-72 w-72 top-[15%] right-[-10%] opacity-5" animation="animate-float-slow rotate-12" />
            <FloatingElement icon={Brain} className="h-64 w-64 top-[55%] left-[-12%] opacity-5" animation="animate-float-slow -rotate-6" />
            <FloatingElement icon={FlaskRound} className="h-56 w-56 bottom-[5%] right-[5%] opacity-5" animation="animate-float-slow rotate-3" />
            
            {/* Medium size/speed elements */}
            <FloatingElement icon={Atom} className="h-48 w-48 top-[5%] left-[10%] opacity-10" animation="animate-float-medium -rotate-12" />
            <FloatingElement icon={TestTube} className="h-40 w-40 bottom-[20%] right-[15%] opacity-5" animation="animate-float-medium rotate-6" />
            <FloatingElement icon={Beaker} className="h-36 w-36 top-[70%] left-[5%] opacity-10" animation="animate-float-medium rotate-4" />

            {/* Smaller, faster elements */}
            <FloatingElement icon={FlaskRound} className="h-24 w-24 top-[30%] left-[30%] opacity-10" animation="animate-float-fast rotate-15" />
            <FloatingElement icon={Atom} className="h-28 w-28 bottom-[40%] right-[30%]" animation="animate-float-fast -rotate-10" />
            <FloatingElement icon={Dna} className="h-32 w-32 top-[85%] left-[45%] opacity-5" animation="animate-float-fast rotate-8" />
             <FloatingElement icon={TestTube} className="h-20 w-20 top-[50%] left-[60%] opacity-10" animation="animate-float-fast" />
          </>
        )}
        
        {/* --- Original Page Content Sections (ensure z-10) --- */}
        <section className="relative py-24 md:py-36 overflow-hidden w-full z-10">
          <div className="absolute inset-0 bg-gradient-to-b from-background via-background/95 to-background/90"></div>
          <div className="container mx-auto px-4 md:px-6 relative z-10 max-w-6xl">
            <div className="flex flex-col items-center text-center space-y-10">
              <div className="space-y-6 max-w-3xl">
                <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/80">
                  AI-Powered Molecular Generation
                </h1>
                <p className="text-xl md:text-2xl text-muted-foreground leading-relaxed">
                  Revolutionize drug discovery with our multi-objective molecular
                  generation platform, enhanced by explainable AI for unprecedented trust
                  and insights.
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <Button asChild size="lg" className="gap-2 px-8 h-12">
                  <Link href="/generate">
                    Get Started <ArrowRight className="h-4 w-4 ml-1" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="h-12">
                  <Link href="/auth/signin">
                    Sign In
                  </Link>
                </Button>
              </div>
            </div>
          </div>
          <div className="absolute -bottom-8 left-0 right-0 h-24 bg-gradient-to-b from-transparent to-background pointer-events-none z-10"></div>
        </section>

        <section className="py-24 bg-background w-full z-10">
          <div className="container mx-auto px-4 md:px-6 max-w-6xl">
            <div className="text-center mb-16 max-w-3xl mx-auto">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Key Features</h2>
              <p className="text-xl text-muted-foreground">
                Our platform combines cutting-edge AI models with explainable insights for drug discovery
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="bg-card rounded-xl p-8 shadow-sm border border-border hover:shadow-md transition-all">
                <div className="rounded-full bg-primary/10 w-14 h-14 flex items-center justify-center mb-6">
                  <FlaskRound className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Multi-Objective Optimization</h3>
                <p className="text-muted-foreground">
                  Simultaneously optimize for binding affinity, solubility, toxicity, and bioavailability.
                </p>
              </div>
              <div className="bg-card rounded-xl p-8 shadow-sm border border-border hover:shadow-md transition-all">
                <div className="rounded-full bg-primary/10 w-14 h-14 flex items-center justify-center mb-6">
                  <Brain className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Explainable AI</h3>
                <p className="text-muted-foreground">
                  Visualize and understand why specific molecular structures are recommended through transparency.
                </p>
              </div>
              <div className="bg-card rounded-xl p-8 shadow-sm border border-border hover:shadow-md transition-all">
                <div className="rounded-full bg-primary/10 w-14 h-14 flex items-center justify-center mb-6">
                  <Dna className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">3D Visualization</h3>
                <p className="text-muted-foreground">
                  Interactive molecular structure visualization with detailed property analysis.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-24 bg-muted/30 w-full z-10">
          <div className="container mx-auto px-4 md:px-6 max-w-6xl">
            <div className="text-center mb-16 max-w-3xl mx-auto">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">How It Works</h2>
              <p className="text-xl text-muted-foreground">
                Our streamlined workflow makes molecular generation and optimization simple
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-12 max-w-5xl mx-auto">
              <div className="flex flex-col items-center text-center">
                <div className="w-14 h-14 rounded-full bg-primary text-white flex items-center justify-center font-bold text-xl mb-6">1</div>
                <h3 className="text-xl font-semibold mb-3">Define Targets</h3>
                <p className="text-muted-foreground">
                  Upload your target protein structure and define your optimization objectives.
                </p>
              </div>
              <div className="flex flex-col items-center text-center">
                <div className="w-14 h-14 rounded-full bg-primary text-white flex items-center justify-center font-bold text-xl mb-6">2</div>
                <h3 className="text-xl font-semibold mb-3">Generate & Optimize</h3>
                <p className="text-muted-foreground">
                  Our AI models generate candidate molecules optimized for your specific requirements.
                </p>
              </div>
              <div className="flex flex-col items-center text-center">
                <div className="w-14 h-14 rounded-full bg-primary text-white flex items-center justify-center font-bold text-xl mb-6">3</div>
                <h3 className="text-xl font-semibold mb-3">Analyze Results</h3>
                <p className="text-muted-foreground">
                  Review detailed reports with explainable insights into molecular properties.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-24 bg-background w-full z-10">
          <div className="container mx-auto px-4 md:px-6 max-w-6xl">
            <div className="flex flex-col md:flex-row items-center gap-12">
              <div className="flex-1 md:pr-6">
                <h2 className="text-3xl md:text-4xl font-bold mb-6">3D Molecule Visualization</h2>
                <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
                  Interactive molecular structure visualization with detailed property analysis. Explore binding sites, 
                  chemical properties, and structural features with our advanced visualization tools.
                </p>
                <ul className="space-y-4 mb-8">
                  {['Rotate and zoom in 3D space', 'Analyze binding pocket interactions', 'Explore atomic-level contributions'].map((item, i) => (
                    <li key={i} className="flex items-center gap-3">
                      <CheckCircle className="h-5 w-5 text-primary flex-shrink-0" />
                      <span>{item}</span>
          </li>
                  ))}
                </ul>
                <Button asChild size="lg" className="h-12">
                  <Link href="/generate">Try It Now</Link>
                </Button>
              </div>
              <div className="flex-1 w-full">
                <div className="bg-muted/20 rounded-2xl overflow-hidden shadow-lg border border-border aspect-square max-w-md mx-auto md:mx-0 md:ml-auto">
                  <div className="flex items-center justify-center h-full p-8 relative">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Dna className="h-32 w-32 text-primary/30" strokeWidth={1} />
                    </div>
                    <div className="relative text-center">
                      <h3 className="text-xl font-medium mb-2">3D Molecule Visualization</h3>
                      <p className="text-sm text-muted-foreground">Interactive molecular structure visualization</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="py-24 bg-primary/5 w-full z-10">
          <div className="container mx-auto px-4 md:px-6 max-w-6xl">
            <div className="rounded-2xl bg-card shadow-lg border border-border p-10 md:p-14 max-w-4xl mx-auto text-center">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Revolutionize Drug Discovery?</h2>
              <p className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto">
                Join researchers worldwide using our platform to develop the next generation of therapeutic molecules.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button asChild size="lg" className="gap-2 h-12 px-8">
                  <Link href="/generate">
                    Get Started <ArrowRight className="h-4 w-4 ml-1" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="h-12">
                  <Link href="/contact">
                    Contact Us
                  </Link>
                </Button>
              </div>
            </div>
        </div>
        </section>
        
        {/* Keep existing float animations */}
        <style jsx>{`
          @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
            100% { transform: translateY(0px); }
          }
          .animate-float-slow {
            animation: float 12s ease-in-out infinite alternate;
          }
          .animate-float-medium {
            animation: float 8s ease-in-out infinite alternate;
          }
          .animate-float-fast {
            animation: float 5s ease-in-out infinite alternate;
          }
        `}</style>
    </div>
    </>
  );
}
