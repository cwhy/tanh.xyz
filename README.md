# tanh.xyz - Neural Network Playground

An interactive collection of neural network demos running entirely in your browser using [jax-js](https://github.com/ekzhang/jax-js) with WebGPU acceleration.

## 🎨 Design System

This project uses a hand-drawn, playful design aesthetic inspired by sketches and notebook paper:

- **Typography**: Kalam (headings) and Patrick Hand (body text) for a handwritten feel
- **Colors**: Warm paper backgrounds (#fdfbf7) with pencil blacks (#2d2d2d)
- **Borders**: Wobbly, irregular borders with hard offset shadows (no blur)
- **Background**: Subtle dot pattern to simulate notebook paper grain

See [docs/design-guideline.md](docs/design-guideline.md) for complete design specifications.

## 🚀 Demos

### Available Now
- **Two-Moon Classification**: Train a neural network to classify two interleaving half-moon shapes. Watch the decision boundary evolve in real-time.

### Coming Soon
- **MNIST Digit Recognition**: Train a CNN to recognize handwritten digits
- **Autoencoder Visualization**: Explore latent space representations
- **Gradient Descent Playground**: Compare optimization algorithms on various loss landscapes

## 🛠️ Tech Stack

- **Frontend**: [SolidJS](https://solidjs.com) - Reactive UI framework
- **Router**: [TanStack Start](https://tanstack.com/start) - Full-stack framework
- **Styling**: [DaisyUI](https://daisyui.com) + [Tailwind CSS](https://tailwindcss.com)
- **ML Library**: [jax-js](https://github.com/ekzhang/jax-js) - JAX implementation for JavaScript with WebGPU

## 📦 Development

```bash
# Install dependencies
bun install

# Start dev server
bun run dev

# Build for production
bun run build

# Deploy to Cloudflare Pages
bun run deploy
```

## 📁 Project Structure

```
src/
├── routes/
│   ├── index.tsx           # Demo gallery landing page
│   ├── demos/
│   │   └── two-moon.tsx    # Two-Moon Classification demo
│   └── __root.tsx          # Root layout
├── components/
│   ├── DemoCard.tsx        # Reusable demo card component
│   └── TwoMoonDemo.tsx     # Two-Moon demo implementation
├── lib/
│   ├── neural-net.ts       # Neural network training logic
│   └── two-moons.ts        # Two-moon dataset generator
└── styles.css              # Global styles with hand-drawn fonts
```

## 🎯 Adding New Demos

1. Create a new route file in `src/routes/demos/your-demo.tsx`
2. Implement your demo component
3. Add a new `DemoCard` entry in `src/routes/index.tsx`
4. Update the card's `status` from `"coming-soon"` to `"ready"` when complete

## 📄 License

MIT
