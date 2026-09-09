import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { PipelineCanvas } from './components/PipelineCanvas';
import { LiveQueue } from './components/LiveQueue';
import { MessageDetail } from './components/MessageDetail';
import { EvalDashboard } from './components/EvalDashboard';

function App() {
  const [messages, setMessages] = useState<any[]>([]);
  const [activeMessageId, setActiveMessageId] = useState<string>('');

  useEffect(() => {
    fetch('/data/judge_scores.json')
      .then(res => res.json())
      .then(data => {
        if (data.judge_scores) {
          const formattedMessages = data.judge_scores.map((t: any, i: number) => ({
            index: i,
            id: `TKT-${89400 + i}`,
            user: `@user_${100 + i}`,
            avatar: "U",
            tweet: t.tweet,
            intent: t.intent,
            confidence: Math.floor((t.confidence || 0.95) * 100),
            escalation: {
              decision: t.needs_escalation ? 'escalate' : 'auto_handle',
              reason: t.escalation_reason || "Determined by arbitration rules."
            },
            retrieval: t.retrieved_examples ? t.retrieved_examples.map((ex: any, j: number) => ({
              id: `KB-${j}`,
              title: "Retrieved Example",
              similarity: 0.85,
              snippet: ex
            })) : [],
            draft: t.drafted_reply || t.generated_reply,
            meta: { tier: "Standard", seats: "N/A", mrr: "N/A" }
          }));
          setMessages(formattedMessages.slice(0, 50)); // Load top 50 for performance
          if (formattedMessages.length > 0) setActiveMessageId(formattedMessages[0].id);
        }
      });
  }, []);

  const activeMessage = messages.find((m) => m.id === activeMessageId);

  return (
    <div className="min-h-screen flex flex-col pt-20 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative">
      <div className="absolute inset-0 bg-background/90 z-[-1]"></div>
      <Header />
      
      <main className="flex-1 w-full max-w-max-container mx-auto px-6 py-8 flex flex-col space-y-6">
        <PipelineCanvas />
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 relative">
          <LiveQueue messages={messages} activeId={activeMessageId} onSelect={setActiveMessageId} />
          <MessageDetail message={activeMessage} />
          <EvalDashboard />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
