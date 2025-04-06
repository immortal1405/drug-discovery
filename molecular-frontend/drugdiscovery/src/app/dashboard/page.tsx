'use client';

import { useState } from 'react';
import Link from 'next/link';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { 
  Activity, 
  Calendar, 
  Clock, 
  Database, 
  FlaskRound,
  Plus, 
  Search, 
  Settings 
} from 'lucide-react';

// Mock data for recent tasks
const recentTasks = [
  {
    id: 'task-001',
    name: 'COVID-19 Protease Inhibitor Search',
    createdAt: '2023-04-01T10:30:00Z',
    status: 'completed',
    model: 'VAE',
    molecules: 124,
    targetProtein: '6LU7',
  },
  {
    id: 'task-002',
    name: 'Neuraminidase Inhibitor Design',
    createdAt: '2023-03-25T14:15:00Z',
    status: 'completed',
    model: 'GNN',
    molecules: 87,
    targetProtein: '3NSS',
  },
  {
    id: 'task-003',
    name: 'Kinase Inhibitor Library',
    createdAt: '2023-03-20T09:45:00Z',
    status: 'completed',
    model: 'GAN',
    molecules: 153,
    targetProtein: '4R5S',
  },
  {
    id: 'task-004',
    name: 'GPCR Agonist Exploration',
    createdAt: '2023-03-15T16:00:00Z',
    status: 'failed',
    model: 'VAE',
    molecules: 0,
    targetProtein: '5TGZ',
  },
];

// Mock data for quick stats
const stats = [
  { name: 'Total Generations', value: '12', icon: FlaskRound },
  { name: 'Generated Molecules', value: '1,245', icon: Database },
  { name: 'Processing Time', value: '86.4 hrs', icon: Clock },
  { name: 'Success Rate', value: '92%', icon: Activity },
];

export default function DashboardPage() {
  const router = useRouter();
  
  return (
    <div className="container mx-auto py-10 max-w-7xl px-4 md:px-6">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <Button onClick={() => router.push('/generate')}>
          <Plus className="mr-2 h-4 w-4" /> New Generation
        </Button>
      </div>
      
      <div className="grid gap-6">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.map((stat, index) => (
            <Card key={index}>
              <CardContent className="p-6 flex items-center space-x-4">
                <div className="bg-primary/10 p-3 rounded-full">
                  <stat.icon className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{stat.name}</p>
                  <h3 className="text-2xl font-bold">{stat.value}</h3>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        
        {/* Recent Generations */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Recent Generation Tasks</CardTitle>
              <CardDescription>Your recently created molecular generation tasks</CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Search className="h-4 w-4 mr-2" /> Filter
              </Button>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" /> Manage
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Task Name</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Target Protein</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Molecules</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentTasks.map((task) => (
                  <TableRow key={task.id}>
                    <TableCell className="font-medium">
                      <Link href={`/results/${task.id}`} className="hover:underline">
                        {task.name}
                      </Link>
                    </TableCell>
                    <TableCell>{task.model}</TableCell>
                    <TableCell>{task.targetProtein}</TableCell>
                    <TableCell>
                      <Badge
                        className={
                          task.status === 'completed' 
                            ? 'bg-green-500' 
                            : task.status === 'running' 
                            ? 'bg-blue-500' 
                            : 'bg-red-500'
                        }
                      >
                        {task.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center">
                        <Calendar className="h-4 w-4 mr-2 text-muted-foreground" />
                        {new Date(task.createdAt).toLocaleDateString()}
                      </div>
                    </TableCell>
                    <TableCell>{task.molecules}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={task.status === 'failed'}
                          onClick={() => router.push(`/results/${task.id}`)}
                        >
                          View
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => router.push(`/generate?clone=${task.id}`)}
                        >
                          Clone
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
          <CardFooter className="flex justify-between">
            <p className="text-sm text-muted-foreground">Showing {recentTasks.length} of {recentTasks.length} tasks</p>
            <Button variant="outline" size="sm">View All Tasks</Button>
          </CardFooter>
        </Card>
        
        {/* Molecule Library Preview */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Top Molecules</CardTitle>
              <CardDescription>Your highest-performing molecules</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-muted rounded-md flex items-center justify-center">
                      <span className="text-xl font-bold">{i}</span>
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Molecule-{i}</p>
                      <p className="text-sm text-muted-foreground truncate">
                        CC(=O)OC1=CC=CC=C1C(=O)O
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{85 + i}%</p>
                      <p className="text-xs text-muted-foreground">Score</p>
                    </div>
                  </div>
                ))}
                <Button variant="outline" className="w-full mt-2">
                  View All Molecules
                </Button>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Activity Summary</CardTitle>
              <CardDescription>
                Your molecular generation activity over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[200px] flex items-center justify-center bg-muted rounded-md">
                <p className="text-center text-muted-foreground">
                  Activity chart will be displayed here<br />
                  (Using Recharts or Chart.js)
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 