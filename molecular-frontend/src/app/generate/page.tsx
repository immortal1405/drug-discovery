'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../../drugdiscovery/src/components/ui/card';
import { Button } from '../../../drugdiscovery/src/components/ui/button';
import { Input } from '../../../drugdiscovery/src/components/ui/input';
import { Label } from '../../../drugdiscovery/src/components/ui/label';
import { Slider } from '../../../drugdiscovery/src/components/ui/slider';
import { Textarea } from '../../../drugdiscovery/src/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../drugdiscovery/src/components/ui/select';
import { FaSpinner } from 'react-icons/fa';

export default function GeneratePage() {
  const { status } = useSession();
  const router = useRouter();
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  
  // Form state
  const [taskName, setTaskName] = useState('');
  const [description, setDescription] = useState('');
  const [targetProtein, setTargetProtein] = useState('');
  const [generationModel, setGenerationModel] = useState('vae');
  const [numMolecules, setNumMolecules] = useState(10);
  
  // Property weights (0-100)
  const [bindingAffinityWeight, setBindingAffinityWeight] = useState(50);
  const [solubilityWeight, setSolubilityWeight] = useState(50);
  const [toxicityWeight, setToxicityWeight] = useState(50);
  const [bioavailabilityWeight, setBioavailabilityWeight] = useState(50);

  // Redirect to sign-in if not authenticated
  if (status === 'unauthenticated') {
    router.push('/auth/signin');
    return null;
  }

  // Show loading state while checking authentication
  if (status === 'loading') {
    return (
      <div className="flex items-center justify-center min-h-[80vh]">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-t-primary border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  const handleGenerate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setGenerating(true);
    setGenerationError(null);
    
    try {
      // Clear ALL previous results from localStorage
      console.log('Clearing localStorage...');
      localStorage.clear(); // Clear everything to be safe
      console.log('LocalStorage cleared, submitting form with values:', {
        taskName,
        description,
        targetProtein,
        generationModel,
        numMolecules,
        bindingAffinityWeight,
        solubilityWeight,
        toxicityWeight,
        bioavailabilityWeight
      });
      
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          taskName,
          description,
          targetProtein,
          generationModel,
          numMolecules,
          bindingAffinityWeight,
          solubilityWeight,
          toxicityWeight,
          bioavailabilityWeight,
        }),
      });
      
      console.log('API response received, status:', response.status);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate molecules');
      }
      
      console.log('API response successfully processed:', data.taskName);
      
      // Store results in localStorage to pass to results page
      localStorage.setItem('generationResults', JSON.stringify(data));
      console.log('Results saved to localStorage, redirecting to results page');
      
      // Navigate to results page
      router.push('/results');
    } catch (error) {
      console.error('Generation error:', error);
      setGenerationError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="container mx-auto py-6 px-4">
      <h1 className="text-3xl font-bold mb-6">Generate New Molecules</h1>
      
      <form onSubmit={handleGenerate}>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Basic Configuration</CardTitle>
              <CardDescription>
                Provide the basic information for your molecular generation task
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="taskName">Task Name</Label>
                <Input 
                  id="taskName" 
                  placeholder="e.g., COVID-19 Protease Inhibitor Search" 
                  value={taskName}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTaskName(e.target.value)}
                  required
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea 
                  id="description" 
                  placeholder="Describe the purpose of this molecule generation task"
                  value={description}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setDescription(e.target.value)}
                  rows={3}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="targetProtein">Target Protein Structure (PDB File or ID)</Label>
                <div className="flex space-x-2">
                  <Input 
                    id="targetProtein"
                    placeholder="E.g., 6LU7 or upload PDB file"
                    value={targetProtein}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTargetProtein(e.target.value)}
                  />
                  <Button type="button" variant="outline">Browse</Button>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="generationModel">Generation Model</Label>
                <Select 
                  value={generationModel}
                  onValueChange={setGenerationModel}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="vae">Variational Autoencoder (VAE)</SelectItem>
                    <SelectItem value="gan">Generative Adversarial Network</SelectItem>
                    <SelectItem value="diffusion">Diffusion Model</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Button 
              type="button" 
              variant="outline" 
              className="w-full" 
              onClick={() => {
                // This would typically open advanced settings in a real app
                // For now, it's just a visual element to match the screenshot
              }}
            >
              Optimization Properties
            </Button>
            
            <Button 
              type="button" 
              variant="outline" 
              className="w-full" 
              onClick={() => {
                // This would typically open advanced settings in a real app
                // For now, it's just a visual element to match the screenshot
              }}
            >
              Advanced Settings
            </Button>
          </div>
          
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
                  <Label htmlFor="bindingAffinity">Binding Affinity</Label>
                  <span>{bindingAffinityWeight}%</span>
                </div>
                <Slider 
                  id="bindingAffinity"
                  min={0} 
                  max={100} 
                  step={1} 
                  value={[bindingAffinityWeight]}
                  onValueChange={(values: number[]) => setBindingAffinityWeight(values[0])}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="solubility">Solubility</Label>
                  <span>{solubilityWeight}%</span>
                </div>
                <Slider 
                  id="solubility"
                  min={0} 
                  max={100} 
                  step={1} 
                  value={[solubilityWeight]}
                  onValueChange={(values: number[]) => setSolubilityWeight(values[0])}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="toxicity">Toxicity (Lower is Better)</Label>
                  <span>{toxicityWeight}%</span>
                </div>
                <Slider 
                  id="toxicity"
                  min={0} 
                  max={100} 
                  step={1} 
                  value={[toxicityWeight]}
                  onValueChange={(values: number[]) => setToxicityWeight(values[0])}
                />
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="bioavailability">Bioavailability</Label>
                  <span>{bioavailabilityWeight}%</span>
                </div>
                <Slider 
                  id="bioavailability"
                  min={0} 
                  max={100} 
                  step={1} 
                  value={[bioavailabilityWeight]}
                  onValueChange={(values: number[]) => setBioavailabilityWeight(values[0])}
                />
              </div>
            </CardContent>
          </Card>
          
          <div className="flex justify-end space-x-4">
            <Button type="button" variant="outline" onClick={() => router.push('/')}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={generating}
              className="min-w-[180px]"
            >
              {generating ? (
                <>
                  <FaSpinner className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                'Generate Molecules'
              )}
            </Button>
          </div>
          
          {generationError && (
            <div className="p-4 bg-red-50 text-red-700 rounded-md">
              Error: {generationError}
            </div>
          )}
        </div>
      </form>
    </div>
  );
}