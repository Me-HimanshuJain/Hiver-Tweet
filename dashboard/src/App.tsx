import { useState } from 'react';
import LiveQueue from './components/LiveQueue';
import MessageDetail from './components/MessageDetail';
import EvalDashboard from './components/EvalDashboard';
import PipelineVisualization from './components/PipelineVisualization';
import { Activity, MessageSquare, BarChart2, Network } from 'lucide-react';
import './index.css';

export default function App() {
  const [activeTab, setActiveTab] = useState<'queue' | 'detail' | 'eval' | 'topology'>('queue');
  const [selectedTweet, setSelectedTweet] = useState<any | null>(null);

  const renderContent = () => {
    switch (activeTab) {
      case 'queue':
        return <LiveQueue onSelectTweet={(tweet) => { setSelectedTweet(tweet); setActiveTab('detail'); }} />;
      case 'detail':
        return <MessageDetail tweet={selectedTweet} onBack={() => setActiveTab('queue')} />;
      case 'eval':
        return <EvalDashboard />;
      case 'topology':
        return <div className="p-8 text-on-background">Topology View (Coming Soon)</div>;
      default:
        return <LiveQueue onSelectTweet={(tweet) => { setSelectedTweet(tweet); setActiveTab('detail'); }} />;
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-background font-sans">
      {/* Global Header */}
      <header className="sticky top-0 z-50 border-b border-white/5 bg-background/80 backdrop-blur-md">
        <div className="flex h-14 items-center justify-between px-6">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-neon-cyan"></div>
              <span className="font-semibold tracking-wider text-sm">HIVER AI OPS CONSOLE</span>
            </div>
            
            <nav className="hidden md:flex gap-1">
              {[
                { id: 'queue', label: 'LIVE QUEUE', icon: MessageSquare },
                { id: 'detail', label: 'MESSAGE DETAIL', icon: Activity },
                { id: 'eval', label: 'EVAL DASHBOARD', icon: BarChart2 },
                { id: 'topology', label: 'TOPOLOGY', icon: Network },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-neon-cyan/10 text-neon-cyan'
                      : 'text-on-background/60 hover:bg-white/5 hover:text-on-background'
                  }`}
                >
                  <tab.icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-6 text-xs text-on-background/60">
            <div className="flex items-center gap-2">
              <span className="font-mono">PIPELINE:</span>
              <span className="text-neon-cyan">ONLINE</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono">AVG LATENCY:</span>
              <span className="text-white">1.2s</span>
            </div>
          </div>
        </div>

        {/* 3D Pipeline Visualization Viewport */}
        <div className="h-20 w-full border-t border-white/5 bg-abyss-black relative overflow-hidden">
          <PipelineVisualization />
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-[1920px]">
        {renderContent()}
      </main>
    </div>
  );
}
