# Contrato de la presentación web

- **Trabajo:** exposición académica de difusión anime; 7 escenas, 5:30 previstos,
  con 2 anexos fuera del recorrido. No es una landing comercial.
- **Entrega:** `presentacion.html` autocontenido, sin CDN, servidor, backend ni
  conexión. Fuente React/Tailwind/Anime.js y dependencias Node aisladas aquí;
  el proyecto Python sigue gestionado con uv y no se modifica.
- **Referencias:** contenido y guion de `../presentacion/`, documentación oficial
  Anime.js/React, Tailwind/Vite. No hay marca externa ni referencia visual del usuario;
  se adopta un lenguaje editorial de laboratorio, sin copiar otra marca.
- **Dirección:** tinta casi negra, blanco, lima como señal y teal de apoyo;
  tipografía sans grande, anotaciones monoespaciadas, retrato pixelado conceptual,
  diagramas y líneas de proceso. Sin tarjetas genéricas ni gradientes decorativos.
- **Movimiento:** entradas escalonadas, revelado tipográfico, reconstrucción de
  píxeles y dibujo de conexiones. Animaciones acotadas, canceladas al cambiar escena,
  con pausa global y respeto a `prefers-reduced-motion`.
- **Interacción:** flechas/espacio, índice, hash enlazable, pantalla completa,
  guion en diálogo y cronómetro de ensayo iniciado por el expositor.
- **Estados:** resultados pendientes explícitos; evidencia real solo si pasa la
  validación existente. Diagramas didácticos nunca se presentan como muestras reales.
- **Responsive:** escritorio/proyector sin cortes; en móvil se apila el contenido
  y se permite desplazamiento dentro de la escena, conservando navegación visible.
- **Verificación:** abrir el HTML por `file://` y sin red en Chromium aislado;
  comprobar controles, todas las escenas, móvil, movimiento reducido, accesibilidad
  y consola. Mantener intactos los módulos, notebook y entregables anteriores.
