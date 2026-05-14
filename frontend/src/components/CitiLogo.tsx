/**
 * Citi brand mark — heavy lowercase "citi" wordmark in Citi navy with the
 * iconic red arc swooping over the "ti" so the arc terminus lands as the
 * dot of the second "i".
 *
 * The wordmark is rendered as text in a heavy geometric sans (Arial Black
 * stack) so it stays sharp at any size without bundling a webfont. The
 * arc is drawn AFTER the text so the red right cap visually replaces
 * the blue i-dot the font would otherwise paint.
 */
type Props = { size?: number; className?: string }

const CITI_BLUE = '#003B70'
const CITI_RED = '#DA291C'

export default function CitiLogo({ size = 28, className = '' }: Props) {
  // Aspect ratio matches the canonical Citi mark: ~1.55 : 1 (w : h).
  const h = size
  const w = size * 1.55
  return (
    <svg
      viewBox="0 -20 320 130"
      width={w}
      height={h}
      role="img"
      aria-label="Citi"
      className={className}
      style={{ overflow: 'visible' }}
    >
      {/* ── "citi" wordmark — drawn first so the arc paints over the
             dot of the right "i". Heavy weight + tight negative letter
             spacing approximates the Interstate Bold geometry of the
             corporate type. */}
      <text
        x="160"
        y="115"
        textAnchor="middle"
        fontFamily="'Arial Black', 'Helvetica Neue', Arial, sans-serif"
        fontSize="118"
        fontWeight="900"
        fill={CITI_BLUE}
        letterSpacing="-4"
      >
        citi
      </text>

      {/* ── Red arc — domes over "ti" and lands as the right i-dot.
             Quadratic Bezier with the control point well above the
             baseline gives a clean dome; round caps so both ends read
             as solid markers, not stroke ends. */}
      <path
        d="M 142 18  Q 198 -52, 252 18"
        stroke={CITI_RED}
        strokeWidth="22"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  )
}
