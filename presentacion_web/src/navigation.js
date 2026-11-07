export const MAIN_COUNT = 7;

export function routeIndex(hash) {
  const value = hash.replace(/^#\/?/, '');
  if (value === 'preguntas') return 7;
  if (value === 'fuentes') return 8;
  return /^[1-7]$/.test(value) ? Number(value) - 1 : 0;
}

export function routeHash(index) {
  return index === 7 ? '#/preguntas' : index === 8 ? '#/fuentes' : `#/${index + 1}`;
}

export function nextIndex(index, direction) {
  if (index >= MAIN_COUNT) return direction < 0 ? MAIN_COUNT - 1 : index;
  return Math.min(MAIN_COUNT - 1, Math.max(0, index + direction));
}

export function formatTime(seconds) {
  return `${Math.floor(seconds / 60)}`.padStart(2, '0') + ':' + `${seconds % 60}`.padStart(2, '0');
}
