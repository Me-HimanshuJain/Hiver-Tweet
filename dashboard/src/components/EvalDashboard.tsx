import { TrendingUp, TrendingDown, Server, Activity, AlertTriangle } from 'lucide-react';

export default function EvalDashboard() {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold tracking-wide">EVAL DASHBOARD</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Section 1: Baseline Comparisons */}
        <div className="lg:col-span-7 grid grid-cols-3 gap-6">
          <div className="glass-panel p-5 rounded-lg border-whisper flex flex-col justify-between">
            <div className="text-xs uppercase text-on-background/60 mb-2">F1 Score vs Baseline</div>
            <div className="text-3xl font-mono text-white mb-2">0.942</div>
            <div className="flex items-center gap-1.5 text-emerald text-sm">
              <TrendingUp className="w-4 h-4" />
              <span className="font-mono">+4.2%</span>
            </div>
          </div>
          
          <div className="glass-panel p-5 rounded-lg border-whisper flex flex-col justify-between">
            <div className="text-xs uppercase text-on-background/60 mb-2">Latency vs Baseline</div>
            <div className="text-3xl font-mono text-white mb-2">412ms</div>
            <div className="flex items-center gap-1.5 text-emerald text-sm">
              <TrendingDown className="w-4 h-4" />
              <span className="font-mono">-23.7%</span>
            </div>
          </div>

          <div className="glass-panel p-5 rounded-lg border-whisper flex flex-col justify-between">
            <div className="text-xs uppercase text-on-background/60 mb-2">Cost vs Baseline</div>
            <div className="text-3xl font-mono text-white mb-2">$0.0031</div>
            <div className="flex items-center gap-1.5 text-emerald text-sm">
              <TrendingDown className="w-4 h-4" />
              <span className="font-mono">-31.1%</span>
            </div>
          </div>
        </div>

        {/* Section 2: Banking77 Cross-Domain Generalization */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-lg border-whisper">
          <h3 className="text-sm uppercase text-on-background/60 mb-4 flex items-center gap-2">
            <Server className="w-4 h-4" />
            Banking77 Cross-Domain
          </h3>
          <div className="flex items-end justify-between mb-6">
            <div>
              <div className="text-4xl font-mono text-neon-cyan">0.887</div>
              <div className="text-xs text-on-background/60 mt-1">Transfer F1 Score</div>
            </div>
            <div className="text-right">
              <div className="text-xl font-mono text-rose">-0.055</div>
              <div className="text-xs text-on-background/60 mt-1">Delta vs In-Domain</div>
            </div>
          </div>
          <div className="space-y-3">
            {[
              { intent: 'card_arrival', score: 0.91, delta: -0.02 },
              { intent: 'refund_dispute', score: 0.85, delta: -0.08 },
              { intent: 'pin_blocked', score: 0.93, delta: -0.01 },
            ].map(row => (
              <div key={row.intent} className="flex items-center justify-between text-sm">
                <span className="text-on-background/80 font-mono">{row.intent}</span>
                <div className="flex items-center gap-4">
                  <span className="font-mono">{row.score.toFixed(2)}</span>
                  <span className={`font-mono w-12 text-right ${row.delta < -0.05 ? 'text-rose' : 'text-emerald'}`}>
                    {row.delta.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Section 3: Judge Agreement Chart */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-lg border-whisper">
          <h3 className="text-sm uppercase text-on-background/60 mb-6 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Human vs LLM Judge Agreement
          </h3>
          <table className="w-full text-sm text-left">
            <thead className="border-b border-whisper text-xs text-on-background/40">
              <tr>
                <th className="pb-3 font-normal">DIMENSION</th>
                <th className="pb-3 font-normal">PEARSON (r)</th>
                <th className="pb-3 font-normal">MAE</th>
                <th className="pb-3 font-normal">±1 AGREEMENT</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-whisper">
              {[
                { dim: 'Relevance', r: 0.892, mae: 0.31, acc: '94.2%' },
                { dim: 'Empathy', r: 0.814, mae: 0.45, acc: '88.7%' },
                { dim: 'Actionability', r: 0.876, mae: 0.38, acc: '91.5%' },
                { dim: 'Conciseness', r: 0.841, mae: 0.42, acc: '89.1%' },
              ].map(row => (
                <tr key={row.dim}>
                  <td className="py-4 text-on-background/90">{row.dim}</td>
                  <td className="py-4 font-mono text-neon-cyan">{row.r.toFixed(3)}</td>
                  <td className="py-4 font-mono">{row.mae.toFixed(2)}</td>
                  <td className="py-4 font-mono">{row.acc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Section 4: Failure Breakdown */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-lg border-whisper">
          <h3 className="text-sm uppercase text-on-background/60 mb-6 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber" />
            Failure Breakdown
          </h3>
          <div className="space-y-4">
            {[
              { reason: 'Hallucination in Draft', pct: 34.8, count: 30 },
              { reason: 'Incorrect Retrieval', pct: 27.9, count: 24 },
              { reason: 'Guardrail False Positive', pct: 18.6, count: 16 },
              { reason: 'Schema Mismatch', pct: 10.5, count: 9 },
              { reason: 'Sentiment Drift', pct: 8.2, count: 7 },
            ].map((fail, i) => (
              <div key={i} className="flex items-center gap-4">
                <div className="w-16 font-mono text-xs text-right text-on-background/60">
                  {fail.pct.toFixed(1)}%
                </div>
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-on-background/90">{fail.reason}</span>
                    <span className="font-mono text-on-background/40">{fail.count}</span>
                  </div>
                  <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-rose/80" 
                      style={{ width: `${fail.pct}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
