---
name: Neural Sentry
colors:
  surface: '#0c1324'
  surface-dim: '#0c1324'
  surface-bright: '#33394c'
  surface-container-lowest: '#070d1f'
  surface-container-low: '#151b2d'
  surface-container: '#191f31'
  surface-container-high: '#23293c'
  surface-container-highest: '#2e3447'
  on-surface: '#dce1fb'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#dce1fb'
  inverse-on-surface: '#2a3043'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#bec6e0'
  on-secondary: '#283044'
  secondary-container: '#3f465c'
  on-secondary-container: '#adb4ce'
  tertiary: '#ffb2b7'
  on-tertiary: '#67001b'
  tertiary-container: '#ff7f8b'
  on-tertiary-container: '#7d0023'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#dae2fd'
  secondary-fixed-dim: '#bec6e0'
  on-secondary-fixed: '#131b2e'
  on-secondary-fixed-variant: '#3f465c'
  tertiary-fixed: '#ffdadb'
  tertiary-fixed-dim: '#ffb2b7'
  on-tertiary-fixed: '#40000d'
  on-tertiary-fixed-variant: '#92002a'
  background: '#0c1324'
  on-background: '#dce1fb'
  surface-variant: '#2e3447'
  success-emerald: '#10B981'
  danger-rose: '#F43F5E'
  info-cyan: '#06B6D4'
  surface-slate: '#0F172A'
  bg-deep: '#020617'
typography:
  headline-lg:
    fontFamily: Metropolis
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Metropolis
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Metropolis
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 12px
  headline-lg-mobile:
    fontFamily: Metropolis
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  grid-columns: '12'
  gutter: 1.5rem
  margin: 2rem
  base-unit: 4px
  container-max-width: 1440px
---

## Brand & Style

This design system is engineered for high-stakes financial surveillance. The visual identity, "Neural Sentry," balances the cold precision of technical data with the high-alert urgency required for anomaly detection. It is designed for fraud analysts and security engineers who require high information density without cognitive overload.

The aesthetic follows a **Modern / Technical** direction, utilizing deep-space dark modes to reduce eye strain during long monitoring sessions. It incorporates subtle **Glassmorphism** for surface layering and sharp, high-contrast accents to ensure critical status indicators (threats vs. safe transactions) are immediately distinguishable. The overall feel is authoritative, systematic, and uncompromisingly professional.

## Colors

The palette is optimized for dark-mode utility and data visualization. 

- **Primary (Cyan):** Used for interactive elements, focus states, and neutral data points. It provides a high-tech "glow" against the dark background.
- **Surface & Background:** The foundation uses a deep navy-black (`#020617`) for the canvas and a slightly lighter slate (`#0F172A`) for containers and cards to create depth without relying on heavy borders.
- **Semantic Indicators:** Emerald Green is reserved strictly for "Normal" or "Verified" transaction states. Rose Red is the critical alert color for "Anomaly" or "Fraud" flags, ensuring high-priority items dominate the visual hierarchy.

## Typography

The typographic system utilizes a triple-font approach to maximize legibility in data-dense environments:

1.  **Metropolis:** Used for headlines and structural navigation to provide a clean, geometric, and modern feel.
2.  **Geist:** The primary body face, chosen for its exceptional clarity and systematic spacing, ideal for reading logs and transaction details.
3.  **JetBrains Mono:** Reserved for data values, transaction IDs, timestamps, and status labels. The monospaced nature ensures that numeric data aligns perfectly in tables and lists, facilitating rapid scanning.

## Layout & Spacing

This system employs a **Fluid Grid** model with a strict 4px baseline rhythm. 

- **Desktop:** A 12-column grid with 24px (1.5rem) gutters. Sidebars are fixed at 280px, while the main dashboard content area expands fluidly.
- **Tablet:** Transitions to an 8-column grid. Margins reduce to 1.5rem.
- **Mobile:** A 4-column grid with 1rem margins. Complex data tables should collapse into "Summary Cards" or use horizontal scrolling for preservation of data integrity.
- **Spacing Logic:** Padding within cards should be generous (1.5rem to 2rem) to offset the density of the data, providing "visual breathing room."

## Elevation & Depth

Hierarchy is achieved through **Tonal Layering** and **Subtle Outlines** rather than heavy shadows, maintaining a crisp "HUD" (Heads-Up Display) aesthetic.

1.  **Level 0 (Base):** `#020617` - The main canvas.
2.  **Level 1 (Cards/Panels):** `#0F172A` - Used for primary widgets. These feature a 1px solid border using a low-opacity Cyan (`#06B6D4` at 10%) to define edges.
3.  **Level 2 (Popovers/Modals):** A slightly lighter slate with a **Backdrop Blur** (12px) effect to create a glassmorphism feel that maintains context of the data underneath.
4.  **Interactive States:** Hovering over a list item or card should trigger a subtle Cyan outer glow (4px blur, 10% opacity) to indicate activity.

## Shapes

The design system uses **Soft** geometry. Corners are slightly rounded (`0.25rem` for standard elements) to feel modern and accessible, but sharp enough to maintain a precise, technical character. 

- **Primary Buttons & Inputs:** 4px radius (`rounded-md`).
- **Data Cards:** 8px radius (`rounded-lg`).
- **Status Chips:** 100px radius (Full pill) to contrast against the rectangular grid of data tables.

## Components

- **Buttons:** Primary buttons use a solid Cyan fill with dark text. Secondary buttons use a Cyan outline with no fill. Danger actions (e.g., "Block Transaction") use the Rose Red palette.
- **Status Chips:** Small, pill-shaped indicators. "Anomaly" chips use a Rose Red background with 20% opacity and a solid Rose Red text/border. "Normal" chips use the Emerald Green equivalent.
- **Data Tables:** High-density rows with `JetBrains Mono` for all numeric values. Row dividers are subtle (`#0F172A` lightened by 5%). Alternate row striping is discouraged; use hover highlights instead.
- **Input Fields:** Dark background (`#020617`), 1px border. On focus, the border glows Cyan.
- **Anomaly Alerts:** Large-format cards with a 2px left-accent border in Rose Red and an accompanying warning icon.
- **Charts:** Use thin, 2px stroke lines for time-series data. Area charts should use semi-transparent gradients (e.g., Cyan to Transparent).