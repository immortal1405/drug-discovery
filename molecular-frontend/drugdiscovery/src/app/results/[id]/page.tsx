'use client';

import { useState, useEffect, useMemo } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

// Dynamic molecule visualization function
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

// Default mock molecules if nothing is in localStorage
const defaultMockMolecules = [
  {
    id: 'mol-1',
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    name: 'Aspirin-like',
    bindingAffinity: 78,
    solubility: 65,
    toxicity: 22,
    bioavailability: 84,
    image: '/placeholder-visualization.svg',
  },
  {
    id: 'mol-2',
    smiles: 'CCC1=CC=C(C=C1)C2=CC(=NN2C)C(=O)N',
    name: 'Celecoxib-like',
    bindingAffinity: 92,
    solubility: 48,
    toxicity: 35,
    bioavailability: 72,
    image: '/placeholder-visualization.svg',
  },
  {
    id: 'mol-3',
    smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    name: 'Ibuprofen-like',
    bindingAffinity: 65,
    solubility: 72,
    toxicity: 18,
    bioavailability: 91,
    image: '/placeholder-visualization.svg',
  }
];

export default function ResultsPage() {
  const { id } = useParams();
  const router = useRouter();
  const [selectedMolecule, setSelectedMolecule] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [resultData, setResultData] = useState({
    id: id as string,
    name: 'Molecule Generation Results',
    description: '',
    createdAt: new Date().toISOString(),
    status: 'completed',
    model: 'Variational Autoencoder (VAE)',
    molecules: defaultMockMolecules,
    targetProtein: '6LU7',
  });

  // Generate molecule SVGs based on the task name
  const moleculeSvgLarge = useMemo(() => 
    generateMoleculeSVG(resultData.name, 'large'), 
  [resultData.name]);

  const moleculeSvgSmall = useMemo(() => 
    generateMoleculeSVG(resultData.name, 'small'), 
  [resultData.name]);

  useEffect(() => {
    // Try to load data from localStorage first
    const storedResults = localStorage.getItem('generationResults');
    
    if (storedResults) {
      try {
        console.log('Found stored generation results');
        const parsed = JSON.parse(storedResults);
        console.log('Task name:', parsed.taskName);
        
        // Update the resultData with stored values
        setResultData(prev => ({
          ...prev,
          name: parsed.taskName || prev.name,
          description: parsed.description || '',
          model: parsed.generationModel || prev.model,
          targetProtein: parsed.targetProtein || prev.targetProtein,
          // Use the stored molecule data if available
          molecules: parsed.molecules && parsed.molecules.length > 0 
            ? parsed.molecules.map((mol) => ({
                id: mol.id,
                smiles: mol.smiles,
                name: mol.name,
                bindingAffinity: mol.bindingAffinity,
                solubility: mol.solubility,
                toxicity: mol.toxicity,
                bioavailability: mol.bioavailability,
                image: mol.imagePath || mol.svg || generateMoleculeSVG(parsed.taskName, 'large')
              }))
            : prev.molecules
        }));
        
        // Update selected molecule
        if (parsed.molecules && parsed.molecules.length > 0) {
          const firstMol = parsed.molecules[0];
          setSelectedMolecule({
            ...firstMol,
            image: firstMol.imagePath || firstMol.svg || generateMoleculeSVG(parsed.taskName, 'large')
          });
        }
      } catch (err) {
        console.error('Error parsing stored results:', err);
      }
    } else {
      console.log('No stored results found, using demo data');
    }
    
    // Simulate API call to get results
    const timer = setTimeout(() => {
      setIsLoading(false);
      // Set first molecule as selected if not already done
      if (!selectedMolecule && resultData.molecules.length > 0) {
        setSelectedMolecule(resultData.molecules[0]);
      }
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [id]);

  if (isLoading) {
    return (
      <div className="container mx-auto py-10 flex items-center justify-center min-h-[50vh] px-4 md:px-6">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold">Loading results...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-10 max-w-7xl px-4 md:px-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">{resultData.name}</h1>
          <div className="flex gap-2 mt-2">
            <Badge variant="outline">{resultData.model}</Badge>
            <Badge variant="outline">Target: {resultData.targetProtein}</Badge>
            <Badge className="bg-green-500">{resultData.status}</Badge>
          </div>
          {resultData.description && (
            <p className="mt-2 text-muted-foreground">{resultData.description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="outline">Export Data</Button>
          <Button variant="outline">Share</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Generated Molecules</CardTitle>
              <CardDescription>
                Showing top {resultData.molecules.length} molecules
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {resultData.molecules.map((molecule) => (
                  <div
                    key={molecule.id}
                    className={`flex items-center gap-3 p-3 rounded-md cursor-pointer hover:bg-secondary/50 transition-colors ${
                      selectedMolecule?.id === molecule.id ? 'bg-secondary' : ''
                    }`}
                    onClick={() => setSelectedMolecule(molecule)}
                  >
                    <div className="w-12 h-12 bg-muted rounded flex items-center justify-center overflow-hidden">
                      <img 
                        src={moleculeSvgSmall}
                        alt={molecule.name}
                        className="max-w-full h-auto"
                        width={48}
                        height={48}
                      />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium">{molecule.name}</h3>
                      <p className="text-xs text-muted-foreground truncate">{molecule.smiles}</p>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">{molecule.bindingAffinity}%</div>
                      <div className="text-xs text-muted-foreground">Binding</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {selectedMolecule && (
            <Tabs defaultValue="details">
              <TabsList className="w-full">
                <TabsTrigger value="details">Molecule Details</TabsTrigger>
                <TabsTrigger value="visualization">3D Visualization</TabsTrigger>
                <TabsTrigger value="explainability">Explainability</TabsTrigger>
              </TabsList>

              <TabsContent value="details">
                <Card>
                  <CardHeader>
                    <CardTitle>{selectedMolecule.name}</CardTitle>
                    <CardDescription>
                      SMILES: {selectedMolecule.smiles}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row gap-6">
                      <div className="flex-1">
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Binding Affinity</span>
                              <span className="text-sm">{selectedMolecule.bindingAffinity}%</span>
                            </div>
                            <Progress value={selectedMolecule.bindingAffinity} className="h-2" />
                          </div>
                          
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Solubility</span>
                              <span className="text-sm">{selectedMolecule.solubility}%</span>
                            </div>
                            <Progress value={selectedMolecule.solubility} className="h-2" />
                          </div>
                          
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Toxicity (Lower is Better)</span>
                              <span className="text-sm">{selectedMolecule.toxicity}%</span>
                            </div>
                            <Progress value={selectedMolecule.toxicity} className="h-2" />
                          </div>
                          
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">Bioavailability</span>
                              <span className="text-sm">{selectedMolecule.bioavailability}%</span>
                            </div>
                            <Progress value={selectedMolecule.bioavailability} className="h-2" />
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex-1 flex items-center justify-center">
                        <div className="border border-border rounded-md p-4 flex items-center justify-center">
                          <img
                            src={moleculeSvgLarge}
                            alt={selectedMolecule.name}
                            className="max-w-full h-auto"
                            width={200}
                            height={200}
                          />
                        </div>
                      </div>
                    </div>
                    
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Property</TableHead>
                          <TableHead>Value</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <TableRow>
                          <TableCell>Molecular Weight</TableCell>
                          <TableCell>324.4 g/mol</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>LogP</TableCell>
                          <TableCell>2.34</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>H-Bond Donors</TableCell>
                          <TableCell>2</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>H-Bond Acceptors</TableCell>
                          <TableCell>5</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Rotatable Bonds</TableCell>
                          <TableCell>4</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="visualization">
                <Card>
                  <CardHeader>
                    <CardTitle>3D Visualization</CardTitle>
                    <CardDescription>
                      Interactive 3D model of the molecule
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                      <div className="text-center p-6">
                        <div className="w-16 h-16 mx-auto mb-4">
                          <img 
                            src={moleculeSvgSmall} 
                            alt="3D Visualization placeholder"
                            className="w-full h-full"
                          />
                        </div>
                        <h3 className="text-lg font-medium mb-2">3D Visualization</h3>
                        <p className="text-sm text-muted-foreground">
                          Interactive 3D model coming soon...
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="explainability">
                <Card>
                  <CardHeader>
                    <CardTitle>Explainability</CardTitle>
                    <CardDescription>
                      Understanding why this molecule was selected
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="p-4 bg-muted rounded-md">
                      <h3 className="text-lg font-medium mb-2">Key Factors</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        This molecule was selected based on the following factors:
                      </p>
                      <ul className="space-y-2 list-disc pl-5">
                        <li>Strong binding affinity with the target protein</li>
                        <li>Good balance of solubility and bioavailability</li>
                        <li>Low predicted toxicity profile</li>
                        <li>Structural similarity to known {resultData.name.split(' ')[0]} inhibitors</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-2">Important Substructures</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 border rounded-md">
                          <div className="flex gap-2 mb-2">
                            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-700">1</div>
                            <div className="font-medium">Binding Site Interaction</div>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            The {resultData.name.split(' ')[0]}-specific functional group enhances binding to the target protein.
                          </p>
                        </div>
                        <div className="p-4 border rounded-md">
                          <div className="flex gap-2 mb-2">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-700">2</div>
                            <div className="font-medium">Solubility Enhancer</div>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Polar functional groups increase water solubility while maintaining membrane permeability.
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}
        </div>
      </div>
    </div>
  );
} 