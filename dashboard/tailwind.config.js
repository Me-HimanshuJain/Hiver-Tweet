/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'abyss-black': 'var(--bg-abyss-black)',
        'surface': 'var(--bg-surface)',
        'obsidian-glass': 'var(--bg-obsidian-glass)',
        'neon-cyan': 'var(--accent-neon-cyan)',
        'emerald': 'var(--accent-emerald)',
        'amber': 'var(--accent-amber)',
        'rose': 'var(--accent-rose)',
        'on-background': 'var(--text-primary)',
      },
      fontFamily: {
        sans: ['Geist', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono', 'monospace'],
      },
      borderWidth: {
        'whisper': '1px',
      },
      borderColor: {
        'whisper': 'var(--border-whisper)',
      }
    },
  },
  plugins: [],
}
