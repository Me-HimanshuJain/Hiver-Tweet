export const Header = () => {
  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-glass-fill/60 backdrop-blur-xl border-b border-cyan/20 h-20">
      <div className="max-w-max-container mx-auto h-full px-6 flex items-center justify-between">
        {/* Left: Breadcrumb & Cluster Topology */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-primary-container animate-pulse shadow-[0_0_8px_#00e5ff]"></span>
            <span className="font-display-lg text-headline-md font-bold text-primary tracking-tight">Hiver Core</span>
          </div>
          <div className="h-4 w-px bg-outline-variant/60 hidden md:block"></div>
          <div className="hidden md:flex items-center space-x-2 font-meta-sm text-meta-sm text-on-surface-variant">
            <span className="material-symbols-outlined text-[16px] text-primary-fixed-dim">hub</span>
            <span>Cluster us-east-4</span>
            <span className="text-outline-variant">//</span>
            <span className="text-primary-fixed">Ingest Gateway v3.12</span>
          </div>
        </div>
        {/* Center: Pipeline Operational Status Flag */}
        <div className="hidden lg:flex items-center space-x-2 bg-surface-container-lowest/80 border border-cyan/20 px-3 py-1 rounded-full">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary-container opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary-container"></span>
          </span>
          <span className="font-meta-sm text-meta-sm text-secondary-fixed tracking-wider font-semibold">PIPELINE RUNTIME: ACTIVE</span>
        </div>
        {/* Right: Global Metric HUD & Operator Profile */}
        <div className="flex items-center space-x-5">
          <div className="hidden xl:flex items-center space-x-5 border-r border-cyan/20 pr-5">
            {/* Pipeline Health */}
            <div className="flex flex-col text-right">
              <span className="font-meta-sm text-[10px] text-on-surface-variant uppercase tracking-wider">Health</span>
              <div className="flex items-center space-x-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-primary-container"></span>
                <span className="font-meta-lg text-meta-lg font-bold text-primary-fixed">99.94%</span>
              </div>
            </div>
            {/* Throughput */}
            <div className="flex flex-col text-right">
              <span className="font-meta-sm text-[10px] text-on-surface-variant uppercase tracking-wider">Throughput</span>
              <span className="font-meta-lg text-meta-lg font-bold text-starlight-white">1,480 req/s</span>
            </div>
            {/* p95 Latency */}
            <div className="flex flex-col text-right">
              <span className="font-meta-sm text-[10px] text-on-surface-variant uppercase tracking-wider">Inf p95</span>
              <span className="font-meta-lg text-meta-lg font-bold text-secondary-fixed">142ms</span>
            </div>
            {/* Auto-resolve */}
            <div className="flex flex-col text-right">
              <span className="font-meta-sm text-[10px] text-on-surface-variant uppercase tracking-wider">Auto-resolve</span>
              <span className="font-meta-lg text-meta-lg font-bold text-primary-fixed-dim">78.4%</span>
            </div>
          </div>
          {/* Controls / Admin Profile */}
          <div className="flex items-center space-x-3">
            <button aria-label="Notifications" className="w-9 h-9 flex items-center justify-center rounded-lg bg-surface-container-low border border-cyan/20 text-on-surface-variant hover:text-primary active:-translate-y-px transition-all">
              <span className="material-symbols-outlined text-[20px]">notifications</span>
            </button>
            <div className="flex items-center space-x-2.5 pl-1 py-1 pr-3 rounded-lg bg-surface-container-low/70 border border-cyan/20">
              <div className="w-7 h-7 rounded-md bg-secondary-container text-on-secondary-container flex items-center justify-center font-meta-sm font-bold text-xs">
                OP
              </div>
              <div className="flex flex-col">
                <span className="font-meta-sm text-xs font-semibold text-starlight-white leading-tight">Admin Ops</span>
                <span className="font-meta-sm text-[10px] text-primary-fixed-dim leading-none">L4 Architect</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
