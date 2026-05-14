/** @type {import('tailwindcss').Config} */
//
// === Citi-themed light palette ===
//
// Hierarchy rule: ONLY two text colors — black and Citi deep blue.
// No grays anywhere.
//
//   text-ink        #0b1320   primary body text (near-black, slight blue warmth)
//   text-citi-blue  #003B70   secondary text / labels / chip text
//   text-accent     #056DAE   links + emphasis only
//
// Contrast against the white card surface (#ffffff):
//   ink       on white → 18.7 : 1  (AAA)
//   citi-blue on white → 12.4 : 1  (AAA)
//   accent    on white →  6.3 : 1  (AA)
//
// Contrast against the page surface (#f5f8fd):
//   ink       → 17.5 : 1  (AAA)
//   citi-blue → 11.6 : 1  (AAA)
//
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:     { DEFAULT: '#f5f8fd', soft: '#eaf0f7', card: '#ffffff' },
        // Citi blue palette
        accent: { DEFAULT: '#056DAE', soft: '#1B8FD9', dark: '#003B70' },
        citi:   { red:    '#DD1A22', blue: '#003B70', mid:  '#056DAE' },
        // Two-tone text — black + navy. No gray.
        ink:    { DEFAULT: '#0b1320', soft: '#1a2436' },   // primary body
        muted:  '#003B70',   // = citi.blue — "muted" still means secondary, just navy not gray
        line:   '#b8c5d6',   // stronger blue-tinted border so it's visible
        cite:   '#0a7e3a',
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      borderRadius: {
        xl: '0.875rem',
        '2xl': '1.125rem',
      },
      boxShadow: {
        glow:    '0 0 0 1px rgba(5,109,174,0.40), 0 8px 24px -8px rgba(5,109,174,0.35)',
        card:    '0 1px 2px rgba(11,19,32,0.08), 0 4px 12px -4px rgba(11,19,32,0.10)',
        cardLg:  '0 2px 4px rgba(11,19,32,0.10), 0 14px 32px -10px rgba(11,19,32,0.18)',
      },
      animation: {
        'fade-in': 'fade-in 200ms ease-out',
        'pulse-slow': 'pulse 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(2px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}
