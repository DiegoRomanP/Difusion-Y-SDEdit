# Del ruido al rostro — presentación HTML

## Abrir: no hace falta levantar nada

Abrir **[`presentacion.html`](presentacion.html)** directamente en un navegador moderno.
Se puede copiar **solo ese archivo** al equipo del aula: contiene React, los estilos
Tailwind, Anime.js, iconos, contenido y licencias. No usa CDN, fuentes externas,
servidor, backend, Node o Python para visualizarlo. Los enlaces de las referencias
solo necesitan Internet si se decide visitarlos.

Son **7 escenas principales y 2 anexos**, con guion de **5:30 planificados** y tope
de ensayo de 6 minutos. No avanza automáticamente: el ritmo lo controla el expositor.
El cronómetro es una ayuda de ensayo, no una medición previa de la duración del guion.

### Qué incluye

- Reconstrucción animada de píxeles y entradas escalonadas con Anime.js.
- Control de ruido interactivo y diagrama animado de U-Net.
- Navegación por teclado, índice, pantalla completa y guion por escena.
- Cronómetro con pausa, reanudación y reinicio; no arranca solo.
- Adaptación a móvil/tablet y respeto a movimiento reducido del sistema.
- Botón para pausar animaciones; todas las escenas siguen legibles sin movimiento.

**Resultados pendientes:** las ilustraciones son dibujos matemáticos didácticos,
no imágenes del dataset ni salidas de una red entrenada. Las muestras y métricas de
la comparación aparecen como «pendientes»/«—», nunca como ceros o resultados ficticios.

## Controles

| Tecla | Acción |
| --- | --- |
| ← / →, espacio | Anterior / siguiente; espacio avanza |
| 1–7 | Ir a una escena principal |
| Inicio / Fin | Portada / cierre |
| I | Índice, incluidos los anexos |
| N | Guion de la escena actual |
| F | Entrar/salir de pantalla completa |
| T / R | Iniciar/pausar el cronómetro / reiniciar |
| H o ? | Ayuda de teclado |
| Esc | Cerrar un diálogo |

Los controles nativos conservan su teclado: las flechas ajustan el deslizador y
espacio abre las preguntas o activa un botón enfocado. En móvil se puede desplazar
el contenido de la escena sin perder los controles inferiores. Los anexos no se
incluyen al avanzar desde el cierre; se abren desde el índice o la discusión.

## Editar o levantar únicamente el frontend

Requiere Node **22.12+** (recomendado 24). Desde la raíz del curso:

```bash
cd pc3/presentacion_web
npm ci
npm run dev
```

Esto inicia **solo Vite en localhost**, sin entrenar modelos ni ejecutar el notebook.
Las dependencias y `package-lock.json` viven aquí; el proyecto Python mantiene su
entorno y dependencias uv intactos.

ara reconstruir el archivo autocontenido:

```bash
npm run build
```

`dist/` es temporal; el entregable es **`presentacion.html`**. No editar ese HTML
minificado a mano: se sobrescribe al construir. Los componentes están en
`src/scenes/`, la interfaz en `src/components/` y el diseño en `src/styles.css`.
Las clases Tailwind se detectan solo en `src/`, no en el HTML generado.

## Actualizar con resultados reales

El contenido inicial comparte notas y protocolo con la presentación anterior.
`tools/export_content.py` reutiliza su validador de evidencias; **no ejecuta modelos**.
`src/content.json` se incluye en el repositorio para poder reconstruir el frontend
sin instalar el entorno Python. Para refrescarlo se necesita el entorno uv del proyecto.

Tras completar `sample`, `evaluate` y `report` siguiendo el README del proyecto:

```bash
npm run content -- --require-results
npm run build
```

Se exigen los tres casos comparables, test reservado, presupuesto completo,
configuración/código compatibles y hashes válidos de los cuatro primeros PNG por caso.
Estos se embeben en el HTML; no se seleccionan solo las mejores caras. Si faltan
evidencias o se mezclan experimentos, la exportación falla en vez de inventar datos.
Para refrescar solo el protocolo/guion sin exigir resultados: `npm run content`.

Antes de exponer: añadir los integrantes reales si se desea, revisar la interpretación
de los resultados y ensayar con cronómetro. Una semilla no prueba superioridad
universal; la std de KID no es un intervalo de confianza entre entrenamientos.

## Verificar

```bash
npm test
npm run build
npm run browser:install
npm run test:browser
```

- **7 pruebas unitarias** de rutas, empaquetado offline y dibujo determinista.
- **17 pruebas en Chromium aislado**, abriendo `file://` sin red: todas las escenas,
  resoluciones 320/768/1024/1280/1440, teclado, guion, foco, fullscreen, cronómetro,
  Anime.js y movimiento reducido. Consola y peticiones remotas comprobadas.
- Auditoría automática axe de criterios **WCAG 2.1 AA** en las 9 escenas y el guion;
  no sustituye una evaluación completa de accesibilidad.
- Revisión visual de capturas reales. Renders y reportes quedan en `.qa/`, ignorado.

El navegador de pruebas se guarda en `../.cache/presentation-web-browsers/`, sin
instalar paquetes del sistema. No hace falta descargarlo para abrir el HTML o editar.
Se probó en Chromium; comprobar también el navegador/proyector que se utilizará en el aula.

No se modificó el código experimental ni se ejecutaron notebook, calibración o
entrenamientos. Los entregables PPTX/PDF anteriores se conservan en `../presentacion/`.
