'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import Image from 'next/image';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../../drugdiscovery/src/components/ui/card';
import { Button } from '../../../drugdiscovery/src/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../drugdiscovery/src/components/ui/tabs';
import { Progress } from '../../../drugdiscovery/src/components/ui/progress';
import { Badge } from '../../../drugdiscovery/src/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../drugdiscovery/src/components/ui/select';
import { Input } from '../../../drugdiscovery/src/components/ui/input';
import { 
  FaSort, FaSortUp, FaSortDown, FaChevronLeft, FaChevronRight, 
  FaFlask, FaDna, FaSearch, FaDownload, FaShare, FaInfoCircle,
  FaThLarge, FaList, FaStar, FaStarHalfAlt, FaRegStar
} from 'react-icons/fa';

// Define molecule type
interface Molecule {
  id: string;
  smiles: string;
  name: string;
  bindingAffinity: number;
  solubility: number;
  toxicity: number;
  bioavailability: number;
  qed: number;
  logp: number;
  novelty: number;
  imagePath: string;
  svg?: string;
}

// Define metrics type
interface Metrics {
  avgBindingAffinity: number;
  avgSolubility: number;
  avgToxicity: number;
  avgBioavailability: number;
  meanQED: number;
  stdQED: number;
  meanLogP: number;
  stdLogP: number;
}

// Define generation result type
interface GenerationResult {
  success: boolean;
  taskId: string;
  taskName: string;
  description?: string;
  timestamp: string;
  generationModel: string;
  targetProtein: string;
  molecules: Molecule[];
  metrics: Metrics;
  visualizations: {
    qedDistribution: string;
    logpDistribution: string;
    qedVsLogp: string;
    topMoleculeRadar: string;
  };
}

export default function ResultsPage() {
  const { status } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<GenerationResult | null>(null);
  const [selectedMolecule, setSelectedMolecule] = useState<Molecule | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');
  const [sortField, setSortField] = useState<string>('bindingAffinity');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Load generation results from localStorage
  useEffect(() => {
    console.log('Results page mounted, checking localStorage...');
    const storedResults = localStorage.getItem('generationResults');
    
    if (storedResults) {
      try {
        console.log('Found generation results in localStorage');
        const parsedResults = JSON.parse(storedResults);
        console.log('Parsed results:', {
          taskName: parsedResults.taskName, 
          description: parsedResults.description,
          timestamp: parsedResults.timestamp,
          modelType: parsedResults.generationModel,
          moleculesCount: parsedResults.molecules?.length || 0
        });
        
        // Ensure we have the correct field mappings
        if (parsedResults.molecules && parsedResults.molecules.length > 0) {
          // Make sure imagePath is set
          parsedResults.molecules = parsedResults.molecules.map((mol: any) => ({
            ...mol,
            imagePath: mol.imagePath || mol.svg || '' // Fallback in case imagePath is missing
          }));
          
          console.log('Sample molecule:', parsedResults.molecules[0]);
          
          setResults(parsedResults);
          
          // Select the top performing molecule by default
          const sortedMolecules = [...parsedResults.molecules].sort((a, b) => {
            const scoreA = (a.bindingAffinity + a.solubility + (100 - a.toxicity) + a.bioavailability) / 4;
            const scoreB = (b.bindingAffinity + b.solubility + (100 - b.toxicity) + b.bioavailability) / 4;
            return scoreB - scoreA;
          });
          setSelectedMolecule(sortedMolecules[0]);
        } else {
          console.error('No molecules found in results');
        }
      } catch (error) {
        console.error('Error parsing results:', error);
      }
    } else {
      console.log('No results found in localStorage');
    }
    setLoading(false);
  }, []);
  
  // Redirect to sign-in if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/signin');
    }
  }, [status, router]);
  
  // Filtered molecules based on search query
  const filteredMolecules = results?.molecules
    ? results.molecules.filter(mol => 
        mol.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        mol.smiles.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : [];
    
  // Sorted molecules based on sort field and direction
  const sortedMolecules = [...filteredMolecules].sort((a, b) => {
    let valA: number, valB: number;
    
    switch (sortField) {
      case 'toxicity':
        // For toxicity, lower is better
        valA = a[sortField];
        valB = b[sortField];
        return sortDirection === 'asc' ? valA - valB : valB - valA;
      case 'overall':
        // For overall score, calculate average of all properties (with toxicity inverted)
        valA = (a.bindingAffinity + a.solubility + (100 - a.toxicity) + a.bioavailability) / 4;
        valB = (b.bindingAffinity + b.solubility + (100 - b.toxicity) + b.bioavailability) / 4;
        return sortDirection === 'asc' ? valA - valB : valB - valA;
      default:
        if (sortField in a && sortField in b) {
          valA = a[sortField as keyof Molecule] as number;
          valB = b[sortField as keyof Molecule] as number;
          if (typeof valA === 'number' && typeof valB === 'number') {
            return sortDirection === 'asc' ? valA - valB : valB - valA;
          }
        }
        return 0;
    }
  });
  
  // Calculate overall score for a molecule (0-100)
  const calculateOverallScore = (molecule: Molecule): number => {
    return Math.round(
      (molecule.bindingAffinity + molecule.solubility + (100 - molecule.toxicity) + molecule.bioavailability) / 4
    );
  };
  
  // Handle column header click for sorting
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc'); // Default to descending for new field
    }
  };
  
  // Get sort icon based on current sort state
  const getSortIcon = (field: string) => {
    if (sortField !== field) return <FaSort className="ml-1 text-gray-400" />;
    return sortDirection === 'asc' ? <FaSortUp className="ml-1 text-primary" /> : <FaSortDown className="ml-1 text-primary" />;
  };
  
  // Get star rating based on value (0-100)
  const getStarRating = (value: number) => {
    const normalizedValue = value / 20; // Convert to 0-5 scale
    const fullStars = Math.floor(normalizedValue);
    const hasHalfStar = normalizedValue - fullStars >= 0.5;
    
    return (
      <div className="flex">
        {[...Array(5)].map((_, i) => {
          if (i < fullStars) return <FaStar key={i} className="text-yellow-500" />;
          if (i === fullStars && hasHalfStar) return <FaStarHalfAlt key={i} className="text-yellow-500" />;
          return <FaRegStar key={i} className="text-gray-300" />;
        })}
      </div>
    );
  };
  
  // Show loading state while checking authentication
  if (status === 'loading' || loading) {
    return (
      <div className="flex items-center justify-center min-h-[80vh]">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-t-primary border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4">Loading results...</p>
        </div>
      </div>
    );
  }
  
  // Show message if no results are available
  if (!results) {
    return (
      <div className="container mx-auto py-6 px-4">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>No Generation Results Found</CardTitle>
            <CardDescription>
              No molecule generation results were found in your session.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button onClick={() => router.push('/generate')}>
              Generate New Molecules
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }
  
  return (
    <div className="container mx-auto py-6 px-4">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold">{results.taskName}</h1>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-sm text-muted-foreground">
              {new Date(results.timestamp).toLocaleString()}
            </span>
            <Badge variant="outline" className="ml-2">
              {results.generationModel}
            </Badge>
            <Badge variant="outline">
              Target: {results.targetProtein}
            </Badge>
            <Badge variant="default" className="bg-green-100 text-green-800">
              Completed
            </Badge>
          </div>
          {results.description && (
            <p className="mt-2 text-muted-foreground max-w-2xl">
              {results.description}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => router.push('/generate')}>
            New Generation
          </Button>
          <Button variant="outline">
            <FaDownload className="mr-2" /> Export
          </Button>
          <Button variant="outline">
            <FaShare className="mr-2" /> Share
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left panel - Molecule list */}
        <div className="xl:col-span-1">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle>Generated Molecules</CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'outline'}
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setViewMode('grid')}
                  >
                    <FaThLarge className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'outline'}
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setViewMode('list')}
                  >
                    <FaList className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="flex gap-2 mt-2">
                <Input 
                  placeholder="Search molecules" 
                  value={searchQuery}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                  className="flex-1"
                />
                <Select value={sortField} onValueChange={setSortField}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="overall">Overall Score</SelectItem>
                    <SelectItem value="bindingAffinity">Binding Affinity</SelectItem>
                    <SelectItem value="solubility">Solubility</SelectItem>
                    <SelectItem value="toxicity">Toxicity (Low)</SelectItem>
                    <SelectItem value="bioavailability">Bioavailability</SelectItem>
                    <SelectItem value="qed">QED</SelectItem>
                    <SelectItem value="logp">LogP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent className="h-[600px] overflow-y-auto pt-0">
              {viewMode === 'list' ? (
                <div className="divide-y">
                  {sortedMolecules.map((molecule) => {
                    const isSelected = selectedMolecule?.id === molecule.id;
                    const overallScore = calculateOverallScore(molecule);
                    
                    return (
                      <div 
                        key={molecule.id}
                        className={`py-3 px-2 cursor-pointer transition-colors hover:bg-accent rounded-md ${
                          isSelected ? 'bg-accent' : ''
                        }`}
                        onClick={() => setSelectedMolecule(molecule)}
                      >
                        <div className="flex justify-between mb-1">
                          <div className="font-medium">{molecule.name}</div>
                          <Badge 
                            variant={overallScore >= 70 ? "default" : overallScore >= 50 ? "secondary" : "destructive"}
                            className="ml-2"
                          >
                            {overallScore}%
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-2">
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Binding Affinity</span>
                              <span>{molecule.bindingAffinity}%</span>
                            </div>
                            <Progress value={molecule.bindingAffinity} className="h-2" />
                          </div>
                          
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Solubility</span>
                              <span>{molecule.solubility}%</span>
                            </div>
                            <Progress value={molecule.solubility} className="h-2" />
                          </div>
                          
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Toxicity (Lower Better)</span>
                              <span>{molecule.toxicity}%</span>
                            </div>
                            <Progress value={100 - molecule.toxicity} className="h-2" />
                          </div>
                          
                          <div className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span>Bioavailability</span>
                              <span>{molecule.bioavailability}%</span>
                            </div>
                            <Progress value={molecule.bioavailability} className="h-2" />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {sortedMolecules.map((molecule) => {
                    const isSelected = selectedMolecule?.id === molecule.id;
                    const overallScore = calculateOverallScore(molecule);
                    
                    return (
                      <Card 
                        key={molecule.id} 
                        className={`cursor-pointer hover:border-primary transition-colors ${
                          isSelected ? 'border-primary' : ''
                        }`}
                        onClick={() => setSelectedMolecule(molecule)}
                      >
                        <CardContent className="p-3">
                          <div className="relative h-24 mb-2">
                            <div className="absolute inset-0 flex items-center justify-center">
                              {molecule.imagePath || molecule.svg ? (
                                <img 
                                  src={molecule.imagePath || molecule.svg} 
                                  alt={molecule.name}
                                  className="max-h-full max-w-full object-contain"
                                  onError={(e) => {
                                    console.warn('Error loading molecule image:', molecule.imagePath || molecule.svg);
                                    e.currentTarget.src = '/placeholder-visualization.svg';
                                  }}
                                />
                              ) : (
                                <FaDna className="h-12 w-12 text-muted-foreground" />
                              )}
                            </div>
                          </div>
                          <div className="flex justify-between items-center mb-1">
                            <div className="font-medium truncate">{molecule.name}</div>
                            <Badge variant={overallScore >= 70 ? "default" : overallScore >= 50 ? "secondary" : "destructive"}>
                              {overallScore}%
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground truncate">
                            {molecule.smiles.substring(0, 20)}...
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
              
              {sortedMolecules.length === 0 && (
                <div className="text-center py-12">
                  <FaSearch className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                  <p>No molecules found matching your search.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* Right panels - Molecule details and visualizations */}
        <div className="xl:col-span-2 space-y-6">
          {/* Molecule details */}
          {selectedMolecule && (
            <Card>
              <CardHeader className="pb-2">
                <div className="flex justify-between">
                  <CardTitle>Molecule Details</CardTitle>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <FaDownload className="mr-2 h-4 w-4" /> Export
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Molecule visualization */}
                  <div className="flex flex-col items-center">
                    <div className="bg-accent rounded-lg p-4 w-full flex items-center justify-center">
                      {selectedMolecule.imagePath || selectedMolecule.svg ? (
                        <img 
                          src={selectedMolecule.imagePath || selectedMolecule.svg} 
                          alt={selectedMolecule.name}
                          className="max-h-[200px] max-w-full object-contain"
                          onError={(e) => {
                            console.warn('Error loading selected molecule image:', selectedMolecule.imagePath || selectedMolecule.svg);
                            e.currentTarget.src = '/placeholder-visualization.svg';
                          }}
                        />
                      ) : (
                        <FaDna className="h-24 w-24 text-muted-foreground" />
                      )}
                    </div>
                    <div className="mt-4 text-center w-full">
                      <p className="font-medium text-lg">{selectedMolecule.name}</p>
                      <p className="text-sm font-mono text-muted-foreground break-all mt-1">
                        {selectedMolecule.smiles}
                      </p>
                      <div className="flex justify-center mt-2">
                        {getStarRating(calculateOverallScore(selectedMolecule))}
                      </div>
                    </div>
                  </div>
                  
                  {/* Properties */}
                  <div className="md:col-span-2 space-y-4">
                    <h3 className="font-semibold text-lg">Properties</h3>
                    
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="font-medium">Binding Affinity</span>
                          <span>{selectedMolecule.bindingAffinity}%</span>
                        </div>
                        <Progress value={selectedMolecule.bindingAffinity} className="h-3" />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="font-medium">Solubility</span>
                          <span>{selectedMolecule.solubility}%</span>
                        </div>
                        <Progress value={selectedMolecule.solubility} className="h-3" />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="font-medium">Toxicity (Lower is Better)</span>
                          <span>{selectedMolecule.toxicity}%</span>
                        </div>
                        <Progress value={100 - selectedMolecule.toxicity} className="h-3" />
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="font-medium">Bioavailability</span>
                          <span>{selectedMolecule.bioavailability}%</span>
                        </div>
                        <Progress value={selectedMolecule.bioavailability} className="h-3" />
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 pt-4">
                      <div className="bg-accent rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Novelty</div>
                        <div className="font-medium text-lg">{(selectedMolecule.novelty * 100).toFixed(0)}%</div>
                      </div>
                      
                      <div className="bg-accent rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">QED (Drug-likeness)</div>
                        <div className="font-medium text-lg">{selectedMolecule.qed.toFixed(2)}</div>
                      </div>
                      
                      <div className="bg-accent rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">LogP</div>
                        <div className="font-medium text-lg">{selectedMolecule.logp.toFixed(2)}</div>
                      </div>
                      
                      <div className="bg-accent rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Overall Score</div>
                        <div className="font-medium text-lg">{calculateOverallScore(selectedMolecule)}%</div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Visualizations */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis & Visualizations</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="distributions">
                <TabsList className="grid grid-cols-3 mb-4">
                  <TabsTrigger value="distributions">Property Distributions</TabsTrigger>
                  <TabsTrigger value="scatter">QED vs LogP</TabsTrigger>
                  <TabsTrigger value="radar">Top Molecule Radar</TabsTrigger>
                </TabsList>
                
                <TabsContent value="distributions" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-accent rounded-lg p-4 flex items-center justify-center">
                      {results.visualizations && results.visualizations.qedDistribution ? (
                        <img 
                          src={results.visualizations.qedDistribution} 
                          alt="QED Distribution"
                          className="max-w-full h-auto"
                          onError={(e) => {
                            console.warn('Error loading QED distribution image');
                            e.currentTarget.src = '/placeholder-visualization.svg';
                          }}
                        />
                      ) : (
                        <div className="text-center p-12">
                          <FaInfoCircle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                          <p>QED distribution visualization not available</p>
                        </div>
                      )}
                    </div>
                    
                    <div className="bg-accent rounded-lg p-4 flex items-center justify-center">
                      {results.visualizations && results.visualizations.logpDistribution ? (
                        <img 
                          src={results.visualizations.logpDistribution} 
                          alt="LogP Distribution"
                          className="max-w-full h-auto"
                          onError={(e) => {
                            console.warn('Error loading LogP distribution image');
                            e.currentTarget.src = '/placeholder-visualization.svg';
                          }}
                        />
                      ) : (
                        <div className="text-center p-12">
                          <FaInfoCircle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                          <p>LogP distribution visualization not available</p>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="bg-accent rounded-lg p-4">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-background rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Avg. Binding Affinity</div>
                        <div className="font-medium text-lg">{results.metrics.avgBindingAffinity.toFixed(1)}%</div>
                      </div>
                      
                      <div className="bg-background rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Avg. Solubility</div>
                        <div className="font-medium text-lg">{results.metrics.avgSolubility.toFixed(1)}%</div>
                      </div>
                      
                      <div className="bg-background rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Avg. Toxicity</div>
                        <div className="font-medium text-lg">{results.metrics.avgToxicity.toFixed(1)}%</div>
                      </div>
                      
                      <div className="bg-background rounded-lg p-3">
                        <div className="text-sm text-muted-foreground">Avg. Bioavailability</div>
                        <div className="font-medium text-lg">{results.metrics.avgBioavailability.toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
                
                <TabsContent value="scatter">
                  <div className="bg-accent rounded-lg p-4 flex items-center justify-center min-h-[400px]">
                    {results.visualizations && results.visualizations.qedVsLogp ? (
                      <img 
                        src={results.visualizations.qedVsLogp} 
                        alt="QED vs LogP"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          console.warn('Error loading QED vs LogP scatter plot');
                          e.currentTarget.src = '/placeholder-visualization.svg';
                        }}
                      />
                    ) : (
                      <div className="text-center p-12">
                        <FaInfoCircle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                        <p>QED vs LogP scatter plot not available</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
                
                <TabsContent value="radar">
                  <div className="bg-accent rounded-lg p-4 flex items-center justify-center min-h-[400px]">
                    {results.visualizations && results.visualizations.topMoleculeRadar ? (
                      <img 
                        src={results.visualizations.topMoleculeRadar} 
                        alt="Top Molecule Properties"
                        className="max-w-full h-auto"
                        onError={(e) => {
                          console.warn('Error loading top molecule radar chart');
                          e.currentTarget.src = '/placeholder-visualization.svg';
                        }}
                      />
                    ) : (
                      <div className="text-center p-12">
                        <FaInfoCircle className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                        <p>Top molecule radar chart not available</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 