import { useState } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { PipelineCanvas } from './components/PipelineCanvas';
import { LiveQueue } from './components/LiveQueue';
import { MessageDetail } from './components/MessageDetail';
import { EvalDashboard } from './components/EvalDashboard';
import { messages } from './data/mockData';

function App() {
  const [activeMessageId, setActiveMessageId] = useState<string>(messages[0].id);

  const activeMessage = messages.find((m) => m.id === activeMessageId);

  return (
    <div className="min-h-screen flex flex-col pt-20 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative">
      <div className="absolute inset-0 bg-background/90 z-[-1]"></div>
      <Header />
      
      <main className="flex-1 w-full max-w-max-container mx-auto px-6 py-8 flex flex-col space-y-6">
        <PipelineCanvas />
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 relative">
          <LiveQueue activeId={activeMessageId} onSelect={setActiveMessageId} />
          <MessageDetail message={activeMessage} />
          <EvalDashboard />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
