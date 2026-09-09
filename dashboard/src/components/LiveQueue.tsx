import { useEffect, useState } from 'react';
import { ShieldAlert, CheckCircle2, Clock } from 'lucide-react';

export default function LiveQueue({ onSelectTweet }: { onSelectTweet: (t: any) => void }) {
  const [tweets, setTweets] = useState<any[]>([]);

  useEffect(() => {
    // Load from results/golden_labeled.csv or judge_scores.json
    // For demo, we'll fetch the JSON if it exists
    fetch('/src/data/judge_scores.json')
      .then(r => r.json())
      .then(data => setTweets(data.judge_scores ? data.judge_scores.slice(0, 20) : []))
      .catch(err => console.error("Failed to load data", err));
  }, []);

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold tracking-wide">LIVE QUEUE</h2>
        <div className="flex gap-2">
          <span className="px-3 py-1 bg-neon-cyan/10 text-neon-cyan text-xs font-mono border border-neon-cyan rounded">
            FILTER: ALL
          </span>
        </div>
      </div>

      <div className="glass-panel rounded-lg overflow-hidden border-whisper">
        <table className="w-full text-sm text-left">
          <thead className="bg-white/5 border-b border-whisper text-xs uppercase text-on-background/60">
            <tr>
              <th className="px-6 py-4 font-mono">Timestamp</th>
              <th className="px-6 py-4">User / Tweet</th>
              <th className="px-6 py-4">Intent</th>
              <th className="px-6 py-4">Confidence</th>
              <th className="px-6 py-4">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-whisper">
            {tweets.map((t, i) => (
              <tr 
                key={i} 
                onClick={() => onSelectTweet(t)}
                className="hover:bg-white/5 cursor-pointer transition-colors"
              >
                <td className="px-6 py-4 font-mono text-on-background/60">
                  {new Date(Date.now() - i * 60000).toISOString().split('T')[1].slice(0, 12)}
                </td>
                <td className="px-6 py-4 max-w-md">
                  <div className="font-semibold text-neon-cyan mb-1">@user_{i + 100}</div>
                  <div className="truncate text-on-background/80">{t.tweet}</div>
                </td>
                <td className="px-6 py-4">
                  <span className="px-2 py-1 bg-neon-cyan/10 text-neon-cyan border border-neon-cyan rounded text-xs">
                    {t.intent}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{t.confidence?.toFixed(2) || "0.95"}</span>
                    <div className="w-16 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-neon-cyan" 
                        style={{ width: `${(t.confidence || 0.95) * 100}%` }}
                      />
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  {t.needs_escalation ? (
                    <div className="flex items-center gap-1.5 text-rose">
                      <ShieldAlert className="w-4 h-4" />
                      <span className="text-xs uppercase">Escalated</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-emerald">
                      <CheckCircle2 className="w-4 h-4" />
                      <span className="text-xs uppercase">Resolved</span>
                    </div>
                  )}
                </td>
              </tr>
            ))}
            {tweets.length === 0 && (
              <tr>
                <td colSpan={5} className="px-6 py-12 text-center text-on-background/40">
                  <Clock className="w-6 h-6 mx-auto mb-2 animate-pulse" />
                  Waiting for pipeline telemetry...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
