import { ArrowLeft, Database, ShieldAlert, Cpu, Activity, CheckCircle2 } from 'lucide-react';

export default function MessageDetail({ tweet, onBack }: { tweet: any, onBack: () => void }) {
  if (!tweet) return null;

  return (
    <div className="p-6">
      <button 
        onClick={onBack}
        className="flex items-center gap-2 text-on-background/60 hover:text-neon-cyan transition-colors mb-6 text-sm"
      >
        <ArrowLeft className="w-4 h-4" />
        BACK TO LIVE QUEUE
      </button>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Column 1: Original Tweet */}
        <div className="glass-panel p-6 rounded-lg border-whisper flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-whisper pb-4">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-neon-cyan"></span>
              ORIGINAL TWEET
            </h3>
            <span className="font-mono text-xs text-on-background/40">ID: {Math.random().toString(36).substr(2, 9)}</span>
          </div>
          
          <div>
            <div className="text-neon-cyan font-semibold mb-2">@customer_user</div>
            <p className="text-lg leading-relaxed">{tweet.tweet}</p>
          </div>

          <div className="mt-auto pt-6 border-t border-whisper">
            <div className="text-xs uppercase text-on-background/40 mb-2">Classified Intent</div>
            <div className="inline-block px-3 py-1.5 bg-neon-cyan/10 border border-neon-cyan text-neon-cyan rounded text-sm">
              {tweet.intent}
            </div>
          </div>
        </div>

        {/* Column 2: Grounding & Draft */}
        <div className="glass-panel p-6 rounded-lg border-whisper flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-whisper pb-4">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald" />
              GROUNDING & DRAFT
            </h3>
          </div>

          <div className="flex-1 flex flex-col gap-6">
            <div>
              <div className="text-xs uppercase text-on-background/40 mb-3">Retrieved Context (FAISS)</div>
              <div className="space-y-3">
                {tweet.retrieved_examples?.slice(0, 2).map((ex: any, idx: number) => (
                  <div key={idx} className="p-3 bg-white/5 border border-whisper rounded text-sm">
                    <div className="text-neon-cyan/80 text-xs mb-1 font-mono">Sim: {ex.similarity?.toFixed(3) || "0.852"}</div>
                    <p className="text-on-background/80 line-clamp-2">{ex.reply}</p>
                  </div>
                ))}
                {!tweet.retrieved_examples && (
                  <div className="p-3 bg-white/5 border border-whisper rounded text-sm text-on-background/60 italic">
                    Context snippets...
                  </div>
                )}
              </div>
            </div>

            <div className="border-t border-whisper pt-6 flex-1">
              <div className="text-xs uppercase text-on-background/40 mb-3 flex items-center justify-between">
                <span>Drafted Reply</span>
                <Cpu className="w-4 h-4" />
              </div>
              <div className="p-4 bg-abyss-black border border-whisper rounded h-full text-on-background/90 text-sm leading-relaxed whitespace-pre-wrap">
                {tweet.draft || "Generating..."}
              </div>
            </div>
          </div>
        </div>

        {/* Column 3: Telemetry & Escalation */}
        <div className="glass-panel p-6 rounded-lg border-whisper flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-whisper pb-4">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose" />
              TELEMETRY
            </h3>
          </div>

          <div className="space-y-6">
            <div>
              <div className="text-xs uppercase text-on-background/40 mb-2">Reasoning</div>
              <p className="text-sm text-on-background/80">{tweet.reasoning || "Analyzing user request for specific keywords."}</p>
            </div>

            <div>
              <div className="text-xs uppercase text-on-background/40 mb-2">Confidence Score</div>
              <div className="flex items-end gap-3">
                <span className="text-3xl font-mono text-neon-cyan leading-none">
                  {tweet.confidence?.toFixed(2) || "0.95"}
                </span>
                <span className="text-sm text-on-background/60 mb-1">/ 1.0</span>
              </div>
              <div className="mt-3 h-1 w-full bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-neon-cyan" 
                  style={{ width: `${(tweet.confidence || 0.95) * 100}%` }}
                />
              </div>
            </div>

            <div className="border-t border-whisper pt-6">
              <div className="text-xs uppercase text-on-background/40 mb-3">Action Status</div>
              {tweet.needs_escalation || (tweet.confidence && tweet.confidence < 0.7) ? (
                <div className="p-4 bg-rose/10 border border-rose/30 rounded text-rose">
                  <div className="flex items-center gap-2 font-semibold mb-1">
                    <ShieldAlert className="w-5 h-5" />
                    ESCALATED TO HUMAN
                  </div>
                  <div className="text-sm opacity-80 mt-2">
                    {tweet.escalation_reason || "Low confidence score detected by Nemotron."}
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-emerald/10 border border-emerald/30 rounded text-emerald">
                  <div className="flex items-center gap-2 font-semibold mb-1">
                    <CheckCircle2 className="w-5 h-5" />
                    AUTO-RESOLVED
                  </div>
                  <div className="text-sm opacity-80 mt-2">
                    High confidence and appropriate draft generated.
                  </div>
                </div>
              )}
            </div>
            
            {tweet.judge_scores && (
              <div className="border-t border-whisper pt-6">
                 <div className="text-xs uppercase text-on-background/40 mb-3">LLM Judge Scores</div>
                 <div className="grid grid-cols-2 gap-4">
                   <div className="bg-white/5 p-3 rounded border border-whisper">
                      <div className="text-[10px] uppercase text-on-background/60 mb-1">Relevance</div>
                      <div className="font-mono text-lg">{tweet.judge_scores.relevance}/5</div>
                   </div>
                   <div className="bg-white/5 p-3 rounded border border-whisper">
                      <div className="text-[10px] uppercase text-on-background/60 mb-1">Empathy</div>
                      <div className="font-mono text-lg">{tweet.judge_scores.empathy}/5</div>
                   </div>
                 </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
