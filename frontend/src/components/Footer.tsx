export const Footer = () => {
  return (
    <footer className="w-full py-6 px-6 border-t border-cyan/20 bg-glass-fill/80 backdrop-blur-xl mt-auto">
      <div className="max-w-max-container mx-auto flex flex-col md:flex-row items-center justify-between gap-4 font-meta-sm text-meta-sm text-on-surface-variant">
        {/* System Ident & Version */}
        <div className="flex items-center space-x-3">
          <span className="text-primary font-bold">Hiver Core Operational System</span>
          <span className="text-outline-variant">|</span>
          <span className="text-primary-fixed-dim">v4.2.9-prod</span>
          <span className="text-outline-variant">|</span>
          <span>Pipeline Latency: <strong className="text-starlight-white font-bold">138ms</strong> (Target &lt;250ms)</span>
        </div>
        {/* Models & Security Badges */}
        <div className="flex flex-wrap items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1.5">
            <span className="material-symbols-outlined text-secondary-container text-[14px]">memory</span>
            <span>Nemotron-Super-120B // Nemotron-Super-49B</span>
          </div>
          <span className="text-outline-variant hidden sm:inline">•</span>
          <div className="flex items-center space-x-1.5">
            <span className="material-symbols-outlined text-primary-fixed text-[14px]">lock</span>
            <span>TLS 1.3 // SOC2 Type II Certified</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
