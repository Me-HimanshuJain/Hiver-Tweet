

export const MessageDetail = ({ message }: { message: any }) => {
  if (!message) return null;

  const isEscalate = message.escalation.decision === 'escalate';

  return (
    <section className="lg:col-span-5 xl:col-span-5 rounded-2xl bg-glass-fill/60 backdrop-blur-xl border border-cyan/20 p-5 flex flex-col h-[820px] overflow-y-auto space-y-4">
      {/* Inspection Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-cyan/20">
        <div className="flex items-center space-x-3">
          <div className="p-1.5 bg-secondary-container/20 border border-secondary-container/40 rounded-lg text-secondary-fixed">
            <span className="material-symbols-outlined text-[20px]">confirmation_number</span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="font-meta-lg text-meta-lg font-bold text-starlight-white">Ticket #{message.id}</h3>
              {isEscalate && (
                <span className="px-1.5 py-0.5 bg-error-container/40 border border-error/40 text-error font-meta-sm text-[10px] font-bold rounded">
                  P1 CRITICAL
                </span>
              )}
            </div>
            <p className="font-meta-sm text-meta-sm text-on-surface-variant">Received 2m ago via Twitter/X Webhook</p>
          </div>
        </div>
        
        {/* Quick Actions */}
        <div className="flex items-center space-x-2">
          <button className="p-1.5 rounded-lg bg-surface-container-high border border-cyan/20 hover:text-primary active:-translate-y-px transition-all" title="View Payload JSON">
            <span className="material-symbols-outlined text-[18px]">code</span>
          </button>
          <button className="p-1.5 rounded-lg bg-surface-container-high border border-cyan/20 hover:text-primary active:-translate-y-px transition-all" title="Flag Thread">
            <span className="material-symbols-outlined text-[18px]">bookmark</span>
          </button>
        </div>
      </div>

      {/* Section A: Original Message */}
      <div className="p-4 rounded-xl bg-surface-container-low/90 border border-cyan/20 space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-meta-sm text-xs font-semibold text-primary uppercase tracking-wider">A. Customer Inbound Payload</span>
          <div className="flex items-center space-x-2 font-meta-sm text-[11px] text-on-surface-variant">
            <span className="px-2 py-0.5 rounded bg-surface-container-highest text-secondary-fixed">Tier: {message.meta.tier}</span>
            <span>{message.meta.seats} seats</span>
            <span>•</span>
            <span className="text-primary-fixed">MRR: {message.meta.mrr}</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-surface-container-lowest/90 border border-cyan/10 font-body-md text-sm text-on-surface leading-relaxed">
          {message.tweet}
        </div>
      </div>

      {/* Section B: Retrieved Grounding Examples */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-meta-sm text-xs font-semibold text-primary uppercase tracking-wider">B. Grounding References (RAG Vector Search)</span>
          <span className="font-meta-sm text-[11px] text-on-surface-variant">{message.retrieval.length} chunks matched</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {message.retrieval.map((item: any) => (
            <div key={item.id} className="p-2.5 rounded-lg bg-surface-container-lowest/80 border border-cyan/20 hover:border-cyan/40 transition-colors">
              <div className="flex items-center justify-between mb-1">
                <span className="font-meta-sm text-xs font-bold text-starlight-white truncate">{item.id}: {item.title}</span>
                <span className="font-meta-sm text-[10px] text-primary-container font-semibold px-1 rounded bg-primary-container/10">{item.similarity} Sim</span>
              </div>
              <p className="font-meta-sm text-[11px] text-on-surface-variant line-clamp-2">
                {item.snippet}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Section C: Drafted AI Reply */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-meta-sm text-xs font-semibold text-primary uppercase tracking-wider">C. Synthesized Response Candidate</span>
          <button 
            className="font-meta-sm text-[11px] text-primary hover:underline flex items-center space-x-1" 
            onClick={() => navigator.clipboard.writeText(message.draft)}
          >
            <span className="material-symbols-outlined text-[13px]">content_copy</span>
            <span>Copy Response</span>
          </button>
        </div>
        <div className="relative rounded-xl bg-surface-container-lowest/90 border border-cyan/30 p-3.5 focus-within:border-secondary-container transition-colors">
          <textarea 
            className="w-full bg-transparent border-0 p-0 text-sm text-starlight-white font-body-md focus:ring-0 resize-none leading-relaxed" 
            rows={4}
            value={message.draft}
            readOnly
          />
          <div className="flex items-center justify-between pt-2 border-t border-cyan/10 font-meta-sm text-[11px] text-on-surface-variant">
            <span>Token Count: {Math.floor((message.draft?.length || 0) / 4)} tokens</span>
            <span className="text-primary-fixed-dim">Tone: Empathetic</span>
          </div>
        </div>
      </div>

      {/* Section D: Escalation Reason */}
      <div className={`p-3.5 rounded-xl bg-surface-container-high/90 border ${isEscalate ? 'border-secondary-container/50' : 'border-primary-fixed/50'} space-y-3 mt-auto`}>
        <div className="flex items-start space-x-3">
          <span className={`material-symbols-outlined text-[22px] flex-shrink-0 ${isEscalate ? 'text-secondary-container' : 'text-primary-fixed'}`}>
            policy
          </span>
          <div className="space-y-1">
            <div className={`font-meta-sm text-xs font-bold ${isEscalate ? 'text-secondary-fixed' : 'text-primary-fixed'}`}>
              Arbitration Flag: {isEscalate ? 'Human Sign-off Required' : 'Auto-Handle Approved'}
            </div>
            <p className="font-body-md text-xs text-on-surface leading-normal">
              {message.escalation.reason}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2 border-t border-cyan/10">
          <button className={`w-full py-2 px-3 rounded-lg ${isEscalate ? 'bg-secondary-container hover:bg-secondary-fixed text-on-secondary' : 'bg-primary-container hover:bg-primary-fixed text-on-primary'} font-headline-md text-xs font-bold active:-translate-y-px transition-all flex items-center justify-center space-x-1.5 shadow-lg`}>
            <span className="material-symbols-outlined text-[16px]">send</span>
            <span>Approve &amp; Send</span>
          </button>
          <button className="w-full py-2 px-3 rounded-lg bg-surface-container-highest hover:bg-surface-bright border border-cyan/20 text-starlight-white font-headline-md text-xs font-semibold active:-translate-y-px transition-all flex items-center justify-center space-x-1">
            <span className="material-symbols-outlined text-[16px]">edit</span>
            <span>Edit Draft</span>
          </button>
          <button className="w-full py-2 px-3 rounded-lg bg-surface-container-highest hover:bg-surface-bright border border-secondary-container/30 text-secondary-fixed font-headline-md text-xs font-semibold active:-translate-y-px transition-all flex items-center justify-center space-x-1">
            <span className="material-symbols-outlined text-[16px]">forward_to_inbox</span>
            <span>Transfer Senior</span>
          </button>
        </div>
      </div>
    </section>
  );
};
