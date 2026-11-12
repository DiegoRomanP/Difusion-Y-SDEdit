// Dibujo matemático didáctico. No usa dataset, pesos, una API ni muestras del modelo.
const INK = [16, 18, 15];
const PAPER = [233, 239, 213];
const LIME = [212, 244, 107];
const OLIVE = [96, 125, 58];
const TEAL = [146, 211, 206];
const random = (seed) => { const value = Math.sin(seed * 127.1 + 311.7) * 43758.5453; return value - Math.floor(value); };
const gaussian = (seed) => Math.sqrt(-2 * Math.log(Math.max(0.00001, random(seed)))) * Math.cos(2 * Math.PI * random(seed + 1000));
const alphas = [];
let product = 1;
for (let time = 0; time < 1000; time++) { product *= 1 - (0.0001 + time / 999 * 0.0199); alphas.push(product); }

export function createPixels() {
  return Array.from({ length: 1024 }, (_, index) => {
    const col = index % 32;
    const row = Math.floor(index / 32);
    const x = (col - 15.5) / 15;
    const y = (row - 15) / 15;
    let color = INK;
    if ((x / 0.78) ** 2 + ((y + 0.12) / 0.94) ** 2 < 1) color = x < 0.18 ? LIME : OLIVE;
    if ((x / 0.5) ** 2 + ((y - 0.04) / 0.68) ** 2 < 1) color = PAPER;
    if (y < -0.20 + 0.20 * Math.sin(x * 6) && Math.abs(x) < 0.65 && y > -0.82) color = x < 0.1 ? LIME : OLIVE;
    if (Math.abs(x) > 0.49 && Math.abs(x) < 0.66 && y > -0.25 && y < 0.46) color = OLIVE;
    for (const eye of [-0.23, 0.23]) {
      if (((x - eye) / 0.18) ** 2 + ((y - 0.05) / 0.12) ** 2 < 1) color = INK;
      if (Math.abs(x - eye) < 0.065 && Math.abs(y - 0.07) < 0.08) color = TEAL;
      if (Math.abs(x - eye + 0.03) < 0.03 && Math.abs(y - 0.01) < 0.035) color = [255, 255, 255];
    }
    if (Math.abs(x) < 0.10 && Math.abs(y - 0.40) < 0.025) color = [89, 111, 72];
    if (y > 0.64 && Math.abs(x) < 0.18) color = PAPER;
    if (y > 0.78 && Math.abs(x) < 0.84) color = Math.abs(x) < 0.11 ? LIME : [53, 64, 43];
    return { col, row, color, noise: [gaussian(index + 1), gaussian(index + 8000), gaussian(index + 16000)],
      startX: random(index + 77) * 31, startY: random(index + 200) * 31 };
  });
}

export function noisyColor(pixel, noise) {
  if (noise <= 0) return pixel.color;
  const alpha = alphas[Math.round(Math.min(1, noise) * 999)];
  return pixel.color.map((channel, index) => Math.round(Math.max(0, Math.min(255,
    (Math.sqrt(alpha) * (channel / 127.5 - 1) + Math.sqrt(1 - alpha) * pixel.noise[index] + 1) * 127.5))));
}

const pixels = createPixels();
export function drawPortrait(canvas, noise = 0, progress = 1) {
  const context = canvas.getContext('2d');
  const size = canvas.width;
  const unit = size / 36;
  context.clearRect(0, 0, size, size);
  for (const pixel of pixels) {
    const color = noisyColor(pixel, noise);
    const x = (pixel.col * progress + pixel.startX * (1 - progress) + 2) * unit;
    const y = (pixel.row * progress + pixel.startY * (1 - progress) + 2) * unit;
    context.fillStyle = `rgb(${color.join(',')})`;
    const width = unit * (0.42 + progress * 0.48);
    context.fillRect(x, y, width, width);
  }
}
