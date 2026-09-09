import { messages } from '../data/mockData';

export const LiveQueue = ({ activeId, onSelect }: { activeId: string, onSelect: (id: string) => void }) => {
  return (
    <section className="lg:col-span-3 xl:col-span-3 rounded-2xl bg-glass-fill/60 backdrop-blur-xl border border-cyan/20 p-5 flex flex-col h-[820px]">
      {/* Queue Header */}
      <div className="flex items-center justify-between pb-3 border-b border-cyan/20">
        <div>
          <h2 className="font-headline-md text-headline-md font-bold text-starlight-white tracking-tight">Live Queue</h2>
          <p className="font-meta-sm text-meta-sm text-on-surface-variant">Dynamic stream arbitration</p>
        </div>
        <span className="px-2 py-0.5 rounded-full bg-surface-container-high border border-cyan/20 font-meta-sm text-xs font-semibold text-primary">
          {messages.length} Active
        </span>
      </div>

      {/* Filter Pills */}
      <div className="flex items-center space-x-1.5 my-3.5">
        <button className="px-2.5 py-1 rounded-md bg-primary-container text-on-primary-container font-meta-sm text-xs font-bold active:-translate-y-px transition-transform">
          All {messages.length}
        </button>
        <button className="px-2.5 py-1 rounded-md bg-surface-container-high hover:bg-surface-bright text-on-surface-variant font-meta-sm text-xs font-medium border border-cyan/20 active:-translate-y-px transition-transform">
          Escalated {messages.filter(m => m.escalation.decision === 'escalate').length}
        </button>
        <button className="px-2.5 py-1 rounded-md bg-surface-container-high hover:bg-surface-bright text-on-surface-variant font-meta-sm text-xs font-medium border border-cyan/20 active:-translate-y-px transition-transform">
          Auto {messages.filter(m => m.escalation.decision === 'auto_handle').length}
        </button>
      </div>

      {/* Search Input */}
      <div className="relative mb-3">
        <label className="sr-only" htmlFor="queue-search">Filter tickets</label>
        <div className="absolute inset-y-0 left-0 pl-2.5 flex items-center pointer-events-none">
          <span className="material-symbols-outlined text-[16px] text-outline">search</span>
        </div>
        <input 
          className="w-full pl-8 pr-3 py-1.5 bg-surface-container-lowest/80 border border-cyan/20 rounded-lg font-meta-sm text-xs text-starlight-white placeholder-outline focus:border-secondary-container focus:ring-0 outline-none" 
          id="queue-search" 
          placeholder="Search payload, intent, user..." 
          type="text" 
        />
      </div>

      {/* Scrollable Feed List */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {messages.map((msg) => {
          const isActive = msg.id === activeId;
          const isEscalate = msg.escalation.decision === 'escalate';
          
          return (
            <article 
              key={msg.id}
              onClick={() => onSelect(msg.id)}
              className={`p-3.5 rounded-xl cursor-pointer transition-all relative overflow-hidden ${
                isActive 
                  ? 'bg-surface-container-high/90 border-2 border-secondary-container' 
                  : 'bg-surface-container-low/80 hover:bg-surface-container-high/50 border border-cyan/20'
              }`}
            >
              {isActive && (
                <div className="absolute top-0 right-0 w-12 h-12 bg-secondary-container/10 -mr-6 -mt-6 rounded-full blur-sm pointer-events-none"></div>
              )}
              
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <div className={`w-6 h-6 rounded-full bg-surface-container-highest border border-cyan/30 flex items-center justify-center font-meta-sm text-[10px] ${isEscalate ? 'text-error' : 'text-primary'}`}>
                    {msg.avatar}
                  </div>
                  <span className="font-meta-sm text-xs font-bold text-starlight-white">{msg.user}</span>
                </div>
                
                {isEscalate ? (
                  <span className="font-meta-sm text-[10px] text-error font-semibold px-1.5 py-0.5 rounded bg-error-container/30 border border-error/40">
                    ESCALATE TO L2
                  </span>
                ) : (
                  <span className="font-meta-sm text-[10px] text-primary-fixed font-semibold px-1.5 py-0.5 rounded bg-primary-container/10 border border-primary-container/30">
                    AUTO-RESOLVED
                  </span>
                )}
              </div>
              
              <p className="font-body-md text-xs text-on-surface-variant line-clamp-2 mb-2 leading-relaxed">
                {msg.tweet}
              </p>
              
              <div className="flex items-center justify-between pt-2 border-t border-cyan/10 font-meta-sm text-[11px]">
                <span className="text-on-surface-variant bg-surface-container-lowest px-1.5 py-0.5 rounded border border-cyan/10">
                  {msg.intent.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </span>
                <div className={`flex items-center space-x-1 ${isEscalate ? 'text-error' : 'text-primary-fixed-dim'}`}>
                  <span className="material-symbols-outlined text-[13px]">{isEscalate ? 'warning' : 'check_circle'}</span>
                  <span>Conf: {msg.confidence}%</span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
};
