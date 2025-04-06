import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { ArrowRight, BotMessageSquare, Cpu, Rocket, TestTubeDiagonal } from "lucide-react"; // Example icons

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-dvh">
      {/* Hero Section */}
      <section className="relative flex flex-col items-center justify-center flex-1 pt-16 md:pt-24 lg:pt-32 border-b">
        <div className="container px-4 md:px-6 text-center">
          <div className="space-y-4">
            <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl/none bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              AI-Powered Molecular Discovery
            </h1>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
              Leverage state-of-the-art generative models and explainable AI to accelerate your drug discovery pipeline.
              Generate novel candidates optimized for multiple properties.
            </p>
            <div className="space-x-4">
              <Link href="/generate">
                <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                  Start Generating <Rocket className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="#features">
                <Button size="lg" variant="outline">
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
        </div>
        {/* Optional: Add subtle background animation/graphic */}
        {/* <div className="absolute inset-0 -z-10 h-full w-full bg-white bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px]"></div> */}
        {/* <div className="absolute bottom-0 left-0 right-0 top-0 -z-10 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div> */}
      </section>

      {/* Features Section */}
      <section id="features" className="w-full py-12 md:py-24 lg:py-32 bg-muted/40">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <div className="inline-block rounded-lg bg-secondary text-secondary-foreground px-3 py-1 text-sm">
                Core Features
              </div>
              <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">Accelerate Your Research</h2>
              <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Our platform integrates advanced AI techniques to streamline the molecular generation and evaluation process.
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl items-start gap-8 sm:grid-cols-2 md:gap-12 lg:grid-cols-3 lg:max-w-none mt-12">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><TestTubeDiagonal className="h-6 w-6 text-primary" /> Multi-Objective Generation</CardTitle>
                <CardDescription>Generate novel molecules optimized simultaneously for binding affinity, solubility, toxicity, and more using VAEs/GNNs.</CardDescription>
              </CardHeader>
              {/* <CardContent> <p>Details...</p> </CardContent> */}
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Cpu className="h-6 w-6 text-primary" /> Advanced Property Prediction</CardTitle>
                <CardDescription>Utilize built-in models (DeepChem, RDKit) and integrate docking tools (AutoDock Vina) for accurate property assessment.</CardDescription>
              </CardHeader>
              {/* <CardContent> <p>Details...</p> </CardContent> */}
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><BotMessageSquare className="h-6 w-6 text-primary" /> Explainable AI (XAI)</CardTitle>
                <CardDescription>Understand model decisions with SHAP, attention visualization, and saliency maps to build trust and guide optimization.</CardDescription>
              </CardHeader>
              {/* <CardContent> <p>Details...</p> </CardContent> */}
            </Card>
          </div>
        </div>
      </section>

      {/* Workflow Section (Simplified) */}
      <section className="w-full py-12 md:py-24 lg:py-32">
         <div className="container px-4 md:px-6 text-center">
           <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Simplified Workflow</h2>
           <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl/relaxed mt-4 mb-8">
             From target definition to explainable results in just a few steps.
           </p>
           {/* Add simple visual diagram or step-by-step cards here */}
           <div className="mx-auto grid max-w-3xl items-start gap-8 sm:grid-cols-3 md:gap-12">
                <div className="flex flex-col items-center space-y-2">
                     <div className="p-3 rounded-full bg-primary/10 text-primary">1</div>
                     <h3 className="text-lg font-semibold">Define Target & Objectives</h3>
                     <p className="text-sm text-muted-foreground">Upload PDB, specify goals.</p>
                </div>
                 <div className="flex flex-col items-center space-y-2">
                     <div className="p-3 rounded-full bg-primary/10 text-primary">2</div>
                     <h3 className="text-lg font-semibold">Generate & Optimize</h3>
                     <p className="text-sm text-muted-foreground">AI generates candidates.</p>
                </div>
                 <div className="flex flex-col items-center space-y-2">
                     <div className="p-3 rounded-full bg-primary/10 text-primary">3</div>
                     <h3 className="text-lg font-semibold">Analyze & Explain</h3>
                     <p className="text-sm text-muted-foreground">Review results with XAI.</p>
                </div>
           </div>
         </div>
      </section>

      {/* Call to Action Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 border-t bg-muted/40">
        <div className="container grid items-center justify-center gap-4 px-4 text-center md:px-6">
          <div className="space-y-3">
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">Ready to Accelerate Your Discovery?</h2>
            <p className="mx-auto max-w-[600px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Sign up and start leveraging the power of AI for your molecular design challenges.
            </p>
          </div>
          <div className="mx-auto w-full max-w-sm space-y-2">
            <Link href="/sign-up">
                <Button type="submit" size="lg" className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">
                     Sign Up for Free <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
            </Link>
            {/* <p className="text-xs text-muted-foreground">
              Get started now. <Link href="#" className="underline underline-offset-2">Terms & Conditions</Link>
            </p> */}
          </div>
        </div>
      </section>

    </div>
  );
} 