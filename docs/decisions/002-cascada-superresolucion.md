# ADR 002 — De comparación de inicialización a cascada difusión + superresolución

## Estado

Aceptado el 10 de noviembre de 2026. **Sustituye a [ADR 001](001-experimento-controlado.md)**
en la pregunta experimental, la arquitectura y los controles de comparabilidad.
Las evidencias de la RTX 3060 que produjo ADR 001 se conservan en `reports/` como
**referencia histórica**, no como brazos de la comparación vigente.

La arquitectura liberada y la etapa de superresolución están decididas aquí; la
ejecución en la RTX 4070 Ti queda pendiente. Este documento no afirma que la calidad
mejore: fija qué se va a medir y con qué límites.

## Contexto

ADR 001 comparaba tres brazos (aleatorio, abierto sin adaptar, fine-tuning) para
aislar el efecto de la inicialización. Aislar ese efecto **exige** mantener la
arquitectura fija, y la arquitectura tenía que ser la del checkpoint abierto
`anton-l/ddpm-ema-flowers-64`. Esa U-Net, a 64×64, coloca los bloques en
64 → 32 → 16 → 8 → 4 → 2 y tiene un solo `AttnDownBlock2D` en la quinta posición:
**la autoatención opera a 4×4**, más el bloque intermedio a 2×2. Con
`attention_head_dim = 8` sobre 512 canales resultan 64 cabezas de ancho 8.

La práctica habitual a 64×64 sitúa la atención a 16×16 y 8×8, donde se decide la
estructura facial (separación de ojos, simetría, disposición del pelo), con ancho de
cabeza 64. Seis bloques para una entrada de 64 px además terminan en 2×2 y gastan
parámetros donde no hay señal espacial. El candado de arquitectura, que era correcto
para la pregunta de ADR 001, limita la calidad alcanzable.

La ejecución completa dio, en test (N=4152, DDIM 50):

| Brazo | FID | KID | s/imagen | Horas locales |
|---|---:|---:|---:|---:|
| Desde cero | 24.902 | 0.018887 | 0.1706 | 3.08 |
| Abierto sin adaptar | 302.603 | 0.354365 | 0.1706 | 0 |
| Fine-tuning | 21.306 | 0.015165 | 0.1707 | 3.07 |

El segundo limitante es el presupuesto: 30 000 updates con lote efectivo 16 son
480 000 exposiciones, unas **14 épocas** sobre 41 522 imágenes, para 113.7 M de
parámetros. Parte del FID observado es arquitectura y parte es entrenamiento corto.
Liberar la arquitectura sin ampliar el presupuesto no transformaría el resultado.

Cambia también el equipo: hay una **RTX 4070 Ti de 12 GB** disponible una noche,
accesible solo por AnyDesk y sin credenciales de GitHub, por lo que el código debe
viajar por clon público y las evidencias volver por transferencia de archivos.
La PC3 se expone el **viernes 13 de noviembre de 2026**.

## Decisión

1. **Retirar los brazos `pretrained` y `finetuned`** de la comparación vigente.
   La pregunta deja de ser si el preentrenamiento ayuda.
2. **Liberar la arquitectura.** Perfil de 4 bloques, `block_out_channels`
   `[128, 256, 384, 512]`, autoatención a **16×16 y 8×8**, `attention_head_dim = 64`,
   `norm_num_groups = 32`. La arquitectura pasa a formar parte del perfil, con su
   hash registrado en `run.json`; deja de estar fijada en el código.
3. **Producir imágenes por cascada:** DDPM a 64×64 y reescalado ×4 a 256×256 con
   **Real-ESRGAN anime preentrenado** (`RealESRGAN_x4plus_anime_6B`). No se entrena
   una SRGAN. El cómputo externo del upsampler se declara igual que se declaró el
   del checkpoint de flores.
4. **La comparación controlada pasa a la etapa de superresolución:** del mismo
   conjunto de muestras de 64 px, brazo **A** reescalado con Real-ESRGAN y brazo **B**
   con interpolación bicúbica. Mismo equipo, mismas semillas, sin entrenamiento extra.
5. **FID/KID se calculan solo a 64×64.** El dataset es nativo de 64 px: **no existe
   verdad de terreno a 256**. Reescalar también las reales mediría el parecido con
   reales reescaladas, no realismo. La etapa de superresolución se juzga por el
   contraste A/B y por inspección visual declarada como tal.
6. **Las guardas de comparabilidad pasan de rechazar a registrar.** Contrato,
   hardware, presupuesto y protocolo divergentes se anotan como divergencias en
   `comparison.json` y se imprimen, en vez de abortar el informe; `--strict`
   restaura el rechazo. Excepción deliberada: `manifest_sha256` y
   `real_subset_sha256` siguen siendo **error duro**, porque no describen
   comparabilidad sino contaminación entre train y test.
7. **Perfiles por equipo.** `configs/rtx4070ti_12gb.toml` con lote mayor y, a
   decidir, bf16. La 3060 registró `skipped_fp16_attempts: 8`; la calibración de la
   4070 Ti dio 0, de modo que fp16 es estable allí y bf16 se justifica por rango
   dinámico y por prescindir de GradScaler, no por un fallo observado.

## Alternativas y consecuencias

**Entrenar una SRGAN propia.** Rechazado: el entrenamiento adversarial es un segundo
proyecto, es inestable y no cabe en tres días junto a un reentrenamiento.

**Difusión directa a 256×256.** Rechazado por presupuesto: multiplica el coste por
paso y por muestra muy por encima de una noche de 4070 Ti.

**Conservar la comparación y arreglar la arquitectura.** Imposible por construcción:
la comparación de inicialización es lo que obliga a copiar la arquitectura ajena.
Las dos cosas que se quieren son el mismo candado visto por sus dos extremos.

**Consecuencias aceptadas.** Se pierde la pregunta sobre transferencia, que era el
eje metodológico del proyecto y de la presentación; ambos se reescriben. La
superresolución **no añade información**: puede volver nítido un artefacto y
producir una cara de 256 px seguramente equivocada donde había una mancha de 64 px,
por lo que se muestran pares con las mismas semillas y sin selección favorable.
Los números de coste de la 3060 y de la 4070 Ti **no son comparables**; los de calidad
a 64 px lo son solo en cuanto comparten partición de test, muestreador y N. Una sola
semilla sigue dando conclusiones exploratorias.

## Fuentes

- ADR 001 y las evidencias en `reports/comparison/` de esta misma repo.
- Ho et al., *Denoising Diffusion Probabilistic Models*: https://arxiv.org/abs/2006.11239
- Dhariwal y Nichol, *Diffusion Models Beat GANs* (resoluciones de atención y
  ancho de cabeza): https://arxiv.org/abs/2105.05233
- Ho et al., *Cascaded Diffusion Models for High Fidelity Image Generation*:
  https://arxiv.org/abs/2106.15282
- Wang et al., *Real-ESRGAN*: https://arxiv.org/abs/2107.10833
- https://huggingface.co/docs/diffusers/api/models/unet2d
