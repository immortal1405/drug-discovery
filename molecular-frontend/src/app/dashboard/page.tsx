'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FaPlus, FaChartBar, FaDna, FaClock, FaFlask } from 'react-icons/fa';

export default function Dashboard() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [recentActivities] = useState([
    {
      id: 1,
      type: 'generation',
      name: 'Kinase Inhibitor Project',
      date: '2 hours ago',
      status: 'Completed',
    },
    {
      id: 2,
      type: 'optimization',
      name: 'Solubility Enhancement',
      date: '1 day ago',
      status: 'Completed',
    },
    {
      id: 3,
      type: 'evaluation',
      name: 'ADME Properties Analysis',
      date: '3 days ago',
      status: 'Completed',
    },
  ]);

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
          <p className="mt-4">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Welcome, {session?.user?.name || 'Researcher'}</h1>
        <p className="text-muted-foreground">
          Access your molecular generation projects and analysis results
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="text-xl flex items-center">
              <FaPlus className="mr-2 text-primary" />
              Generate Molecules
            </CardTitle>
            <CardDescription>
              Create new molecules with AI-driven generation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              className="w-full" 
              onClick={() => router.push('/generate')}
            >
              Start Generation
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="text-xl flex items-center">
              <FaChartBar className="mr-2 text-primary" />
              View Results
            </CardTitle>
            <CardDescription>
              Analyze and explore your generated molecules
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              className="w-full" 
              onClick={() => router.push('/results')}
              variant="outline"
            >
              See Results
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader className="pb-3">
            <CardTitle className="text-xl flex items-center">
              <FaDna className="mr-2 text-primary" />
              Optimization
            </CardTitle>
            <CardDescription>
              Optimize existing molecules for specific properties
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              className="w-full" 
              onClick={() => router.push('/optimize')}
              variant="outline"
            >
              Start Optimization
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Recent Activities</h2>
          <Button variant="ghost" size="sm">
            View All
          </Button>
        </div>
        
        <div className="space-y-4">
          {recentActivities.map((activity) => (
            <Card key={activity.id} className="hover:bg-accent/50 cursor-pointer transition-colors">
              <CardContent className="p-4 flex items-center justify-between">
                <div className="flex items-center">
                  <div className="mr-4 p-2 rounded-full bg-primary/10">
                    {activity.type === 'generation' && <FaDna className="text-primary h-5 w-5" />}
                    {activity.type === 'optimization' && <FaFlask className="text-primary h-5 w-5" />}
                    {activity.type === 'evaluation' && <FaChartBar className="text-primary h-5 w-5" />}
                  </div>
                  <div>
                    <p className="font-medium">{activity.name}</p>
                    <div className="flex items-center text-sm text-muted-foreground">
                      <FaClock className="mr-1 h-3 w-3" />
                      <span>{activity.date}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <Button variant="secondary" size="sm">
                    View
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
} 