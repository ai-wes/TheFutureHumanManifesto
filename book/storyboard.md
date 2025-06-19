One-page “Gentle Singularity” Microsite — storyboard

(Desktop first, 1440 px wide; each numbered block = one full-viewport Figma frame)

# Working title & goal Core copy (≈ 20-40 words) Primary visual & style notes Interaction / motion

1 Hero – “Co-author the Gentle Singularity”
Hook visitors + collect emails • Headline: “Humanity × Intelligence: Co-authoring What Comes Next”
• Sub-line: “Join the wait-list for Future Human Journal” Midjourney glass-brain hologram on dark synth-wave grid.
Glassy CTA capsule (“→ Join wait-list”). Parallax: brain tilts subtly on scroll.
Background orbits slow-rotate (CSS keyframes).
2 Altman’s spark
Frame the conversation Quote pull-out: “We can achieve a gentle singularity…” – Sam Altman (2024).
Short paragraph linking manifesto as direct response. Left: low-poly portrait of Altman (Midjourney > Topaz).
Right: translucent quote card. Quote card fades in, 40 px upward slide.
3 Foundational Visions Matrix
Show lineage & credibility 3-sentence intro + mini bios of Kurzweil, Harari, Tegmark. 3 glass cards in a grid.
Each card: hero portrait thumb, 3-bullet thesis, optimism meter bar. Hover expands card 10 % & reveals “Read more” link (external).
4 2025 Snapshot Radar
Position today Copy: “Four domains are converging faster than expected.”
Legend for AI, Bio, Neuro, Climate. Radar chart component styled “frosted glass” over subtle nebula texture.
Domain icons (Recraft.ai SVG line-icons) on axis tips. Radar draws in (stroke-dashoffset) as section enters viewport.
5 Longevity Escape Velocity
Why it matters 2--paragraph primer + three bullet blockers. Horizontal timeline ribbon (Illustrator) from 2020 → 2045 with milestones; senolytic, re-programming, etc.
Right sidebar: “Barrier” glass chips with icons. Timeline scrubs left/right on scroll (“skew on scroll” illusion).
6 GAPS Pipeline
Differentiator section One-line value prop + numbered pipeline steps (data → scenarios → probabilities → visuals). Vertical stacked cards; each step icon in neon-teal circle; faint connecting line. Scroll-triggered reveal: cards pop-in sequentially (0.15 s stagger).
7 Scenario Decision-Tree
Alpha / Beta / Gamma Micro-copy intro + 3-line synopsis each branch. Branching tree SVG (Whimsical base → Illustrator) with natural-frequency bars (icon array of 100 dots). Hover a branch: highlight path, show tooltip with key turning-point.
8 Ethical Compass — 5 Principles 1-sentence setup + list (Equity, Transparency, Foresight, Agency, Resilience) each ≤ 12 words. Five badge icons in a gentle arc; inner glass circle glows on hover. Badges pulse (opacity 0.9 ↔ 1.0) every 8 s; click = modal with 75-word explainer.
9 Act Now – Community CTA • Headline: “Become a co-author today.”
• Bullets: Join list, submit scenarios, host a futures circle.
• Email field + Discord button. Glass capsule form on dark blur.
QR code to same URL for mobile hand-off. Form shake on empty submit; success confetti burst (Lottie).
10 About / Footer 40-word mission blurb + mini-bio + © + social icons. FHJ logo mono-line SVG; subtle grain BG continues. Fade-in at 10 % opacity while user nears end (intersection observer).
Component & asset checklist
Reusable element Built in Notes
Glass card / capsule Figma component 12 px inner stroke #FFFFFF16, background-blur: 24px, fill #FFFFFF0D
Noise overlay Figma Noise & Texture plugin 20 % opacity, “fine” grain, masked to entire frame
Icon set (20+) Recraft.ai → SVG → Figma Nanobot, Data-Eye, AI chip, etc.; line-weight 2 px
Fonts Google: Inter Tight (display), Space Grotesk (body), JetBrains Mono (code) Loaded via CSS / Figma styles
Colour tokens Neon Teal #00F1FF, Magenta #FF39B6, Slate 900 → 800 gradient Store as Figma variables
Animation specs GSAP (site) / Jitter export (Hero loop) Keep duration ≤ 0.6 s, easing power2.out
Build order (1-day sprint)

    Figma master-file: set up grid, colour, text styles, base glass card.

    AI asset blitz: Midjourney hero + portraits; Recraft icon batch; upscale with Topaz.

    Section frames: drop graphics, wire copy placeholders.

    Prototype: add smart-animate in Figma for stakeholder sign-off.

    Export to Framer/Webflow: wire scroll-triggers & GSAP micro-animations.

    Connect integrations: Mailchimp form, Discord link, Google Analytics.

Use this board as your living checklist—every asset dropped into the corresponding Figma frame keeps design, slide-deck, and PDF perfectly in sync.
