'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';

// Add this function for dynamic molecule visualization based on task name
function generateMoleculeSVG(taskName: string, size: 'large' | 'small' = 'large') {
  // Extract first word from the task name for chemical context
  const taskWord = taskName.split(' ')[0].toLowerCase();
  
  // Customize colors and structure based on task context
  let mainColor = '#d0ebff'; // Default blue
  let sideGroupText = 'OH';  // Default group
  let sideGroup2Text = 'N';
  let sideGroup3Text = 'O';
  let title = taskName.length > 25 ? taskName.substring(0, 22) + '...' : taskName;
  
  // Customize for different disease/drug contexts
  if (taskWord.includes('covid') || taskWord.includes('sars')) {
    mainColor = '#ffadad'; // Red for COVID
    sideGroupText = 'OH';
    title = 'COVID-19 Inhibitor';
  } else if (taskWord.includes('cancer') || taskWord.includes('tumor')) {
    mainColor = '#ffd6a5'; // Orange for cancer
    sideGroupText = 'NH₂';
    title = 'Anti-Cancer Agent';
  } else if (taskWord.includes('alzheimer') || taskWord.includes('dementia')) {
    mainColor = '#9bf6ff'; // Cyan for neurological
    sideGroupText = 'F';
    title = 'Neural Protector';
  } else if (taskWord.includes('diabetes') || taskWord.includes('insulin')) {
    mainColor = '#caffbf'; // Green for metabolic
    sideGroupText = 'COOH';
    title = 'Glucose Regulator';
  } else if (taskWord.includes('pain') || taskWord.includes('analges')) {
    mainColor = '#ffc6ff'; // Purple for pain
    sideGroupText = 'Cl';
    title = 'Pain Reliever';
  } else if (taskWord.includes('antibiotic') || taskWord.includes('bacter')) {
    mainColor = '#a0c4ff'; // Blue for antibiotics
    sideGroupText = 'S';
    title = 'Antibiotic Agent';
  } else if (taskWord.includes('rabies') || taskWord.includes('virus')) {
    mainColor = '#bdb2ff'; // Purple for viral
    sideGroupText = 'P';
    title = 'Antiviral Agent';
  }
  
  // Generate SVG content based on size
  if (size === 'small') {
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">
      <rect width="48" height="48" fill="#f8f9fa" rx="4" ry="4" />
      <polygon points="24,16 30,20 30,28 24,32 18,28 18,20" stroke="#666" stroke-width="1" fill="#e9ecef" />
      <line x1="30" y1="20" x2="36" y2="18" stroke="#666" stroke-width="1" />
      <circle cx="36" cy="18" r="4" fill="${mainColor}" stroke="#666" stroke-width="0.5" />
      <line x1="18" y1="20" x2="12" y2="18" stroke="#666" stroke-width="1" />
      <circle cx="12" cy="18" r="4" fill="#d3f9d8" stroke="#666" stroke-width="0.5" />
      <line x1="24" y1="32" x2="24" y2="38" stroke="#666" stroke-width="1" />
      <circle cx="24" cy="38" r="4" fill="#ffe3e3" stroke="#666" stroke-width="0.5" />
    </svg>`)}`;
  } else {
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
      <rect width="200" height="200" fill="#f8f9fa" rx="10" ry="10" />
      <polygon points="100,70 120,85 120,115 100,130 80,115 80,85" stroke="#666" stroke-width="2" fill="#e9ecef" />
      <line x1="120" y1="85" x2="140" y2="75" stroke="#666" stroke-width="2" />
      <circle cx="140" cy="75" r="8" fill="${mainColor}" stroke="#666" stroke-width="1" />
      <text x="140" y="78" font-family="Arial" font-size="10" text-anchor="middle" fill="#333">${sideGroupText}</text>
      <line x1="100" y1="130" x2="100" y2="150" stroke="#666" stroke-width="2" />
      <circle cx="100" cy="150" r="8" fill="#ffe3e3" stroke="#666" stroke-width="1" />
      <text x="100" y="153" font-family="Arial" font-size="10" text-anchor="middle" fill="#333">${sideGroup3Text}</text>
      <line x1="80" y1="85" x2="60" y2="75" stroke="#666" stroke-width="2" />
      <circle cx="60" cy="75" r="8" fill="#d3f9d8" stroke="#666" stroke-width="1" />
      <text x="60" y="78" font-family="Arial" font-size="10" text-anchor="middle" fill="#333">${sideGroup2Text}</text>
      <text x="100" y="85" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">C</text>
      <text x="100" y="115" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">C</text>
      <text x="100" y="180" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold" fill="#333">${title}</text>
      <text x="100" y="195" font-family="Arial" font-size="10" text-anchor="middle" fill="#666">Molecule visualization</text>
    </svg>`)}`;
  }
}

export default function GeneratePage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  
  // Basic form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [generationModel, setGenerationModel] = useState('vae');
  const [targetProtein, setTargetProtein] = useState('');
  
  // Advanced options
  const [bindingAffinity, setBindingAffinity] = useState(50);
  const [solubility, setSolubility] = useState(50);
  const [toxicity, setToxicity] = useState(50);
  const [bioavailability, setBioavailability] = useState(50);
  const [enableExplainability, setEnableExplainability] = useState(true);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // Clear any previous generation results
      localStorage.clear();
      
      // Create a mock result using the form data
      const mockResult = {
        success: true,
        taskId: `task_${Date.now()}`,
        taskName: name,
        description: description,
        timestamp: new Date().toISOString(),
        generationModel: generationModel,
        targetProtein: targetProtein || '6LU7',
        molecules: [
          {
            id: 'mol-1',
            smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
            name: `${name.substring(0, 4).toUpperCase()}-Asp-001`,
            bindingAffinity: 78,
            solubility: 65,
            toxicity: 22,
            bioavailability: 84,
            imagePath: generateMoleculeSVG(name),
            qed: 0.7,
            logp: 1.2,
            novelty: 0.8
          },
          {
            id: 'mol-2',
            smiles: 'CCC1=CC=C(C=C1)C2=CC(=NN2C)C(=O)N',
            name: `${name.substring(0, 4).toUpperCase()}-Cel-002`,
            bindingAffinity: 92,
            solubility: 48,
            toxicity: 35,
            bioavailability: 72,
            imagePath: generateMoleculeSVG(name),
            qed: 0.8,
            logp: 2.1,
            novelty: 0.9
          },
          {
            id: 'mol-3',
            smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            name: `${name.substring(0, 4).toUpperCase()}-Ibu-003`,
            bindingAffinity: 65,
            solubility: 72,
            toxicity: 18,
            bioavailability: 91,
            imagePath: generateMoleculeSVG(name),
            qed: 0.65,
            logp: 3.5,
            novelty: 0.7
          }
        ],
        metrics: {
          avgBindingAffinity: 78,
          avgSolubility: 62,
          avgToxicity: 25,
          avgBioavailability: 82,
          meanQED: 0.72,
          stdQED: 0.08,
          meanLogP: 2.3,
          stdLogP: 1.2
        },
        visualizations: {
          qedDistribution: '/placeholder-visualization.svg',
          logpDistribution: '/placeholder-visualization.svg',
          qedVsLogp: '/placeholder-visualization.svg',
          topMoleculeRadar: '/placeholder-visualization.svg'
        }
      };
      
      // Save to localStorage
      localStorage.setItem('generationResults', JSON.stringify(mockResult));
      console.log('Saved results to localStorage:', mockResult.taskName);
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Redirect to a results page with a mock ID
      router.push('/results/demo-123');
    } catch (error) {
      console.error('Error submitting generation request:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="container mx-auto py-10 max-w-4xl px-4 md:px-6">
      <h1 className="text-3xl font-bold mb-6">Generate New Molecules</h1>
      
      <form onSubmit={handleSubmit}>
        <div className="grid gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Basic Configuration</CardTitle>
              <CardDescription>
                Provide the basic information for your molecular generation task
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Task Name</Label>
                  <Input
                    id="name"
                    placeholder="E.g., COVID-19 Protease Inhibitor Search"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="model">Generation Model</Label>
                  <Select
                    value={generationModel}
                    onValueChange={setGenerationModel}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="vae">Variational Autoencoder (VAE)</SelectItem>
                      <SelectItem value="gan">Generative Adversarial Network (GAN)</SelectItem>
                      <SelectItem value="gnn">Graph Neural Network (GNN)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe your generation task..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="target">Target Protein Structure (PDB File or ID)</Label>
                <div className="flex gap-2">
                  <Input
                    id="target"
                    placeholder="E.g., 6LU7 or upload PDB file"
                    value={targetProtein}
                    onChange={(e) => setTargetProtein(e.target.value)}
                  />
                  <Button variant="outline" type="button">
                    Browse
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Tabs defaultValue="properties">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="properties">Optimization Properties</TabsTrigger>
              <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
            </TabsList>
            
            <TabsContent value="properties">
              <Card>
                <CardHeader>
                  <CardTitle>Molecular Property Weights</CardTitle>
                  <CardDescription>
                    Adjust the importance of each property in the optimization process
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="binding">Binding Affinity</Label>
                      <span className="text-sm text-muted-foreground">{bindingAffinity}%</span>
                    </div>
                    <Slider
                      id="binding"
                      min={0}
                      max={100}
                      step={1}
                      value={[bindingAffinity]}
                      onValueChange={(value) => setBindingAffinity(value[0])}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="solubility">Solubility</Label>
                      <span className="text-sm text-muted-foreground">{solubility}%</span>
                    </div>
                    <Slider
                      id="solubility"
                      min={0}
                      max={100}
                      step={1}
                      value={[solubility]}
                      onValueChange={(value) => setSolubility(value[0])}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="toxicity">Toxicity (Lower is Better)</Label>
                      <span className="text-sm text-muted-foreground">{toxicity}%</span>
                    </div>
                    <Slider
                      id="toxicity"
                      min={0}
                      max={100}
                      step={1}
                      value={[toxicity]}
                      onValueChange={(value) => setToxicity(value[0])}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="bioavailability">Bioavailability</Label>
                      <span className="text-sm text-muted-foreground">{bioavailability}%</span>
                    </div>
                    <Slider
                      id="bioavailability"
                      min={0}
                      max={100}
                      step={1}
                      value={[bioavailability]}
                      onValueChange={(value) => setBioavailability(value[0])}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="advanced">
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Settings</CardTitle>
                  <CardDescription>
                    Configure additional parameters for the generation process
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="molecules">Number of Molecules</Label>
                      <Input
                        id="molecules"
                        type="number"
                        min={1}
                        max={1000}
                        placeholder="100"
                        defaultValue="100"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="iterations">RL Iterations</Label>
                      <Input
                        id="iterations"
                        type="number"
                        min={1}
                        max={100}
                        placeholder="10"
                        defaultValue="10"
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="xai">Explainable AI (XAI)</Label>
                      <p className="text-sm text-muted-foreground">
                        Generate explanations for why molecules were selected
                      </p>
                    </div>
                    <Switch
                      id="xai"
                      checked={enableExplainability}
                      onCheckedChange={setEnableExplainability}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="filters">Chemical Filters</Label>
                      <p className="text-sm text-muted-foreground">
                        Apply Lipinski's Rule of Five and PAINS filters
                      </p>
                    </div>
                    <Switch
                      id="filters"
                      defaultChecked
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        <div className="flex justify-end gap-4">
          <Button variant="outline" type="button" onClick={() => router.push('/')}>
            Cancel
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Submitting...' : 'Generate Molecules'}
          </Button>
        </div>
      </form>
    </div>
  );