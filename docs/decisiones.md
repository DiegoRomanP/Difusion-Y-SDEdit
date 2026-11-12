# Decisiones del proyecto

Estado al 12 de noviembre de 2026. El entrenamiento terminó y lo que queda es explicar
el modelo, sus resultados y la exploración SDEdit.

## 1. Qué se entrenó

Un DDPM no condicionado que genera rostros anime de 64×64 desde ruido gaussiano,
entrenado con [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
(versión 3).

- **Red:** U-Net de Diffusers (`UNet2DModel`). Tiene 4 bloques de
  `[128, 256, 384, 512]` canales, autoatención a 16×16 y 8×8 y `attention_head_dim` 64,
  en total 88.2 M de parámetros. Está definida en `UNET_CONFIG` de `src/pc3/model.py`.
- **Difusión:** 1000 pasos con beta lineal 0.0001…0.02. La red predice el ruido y la
  pérdida es MSE (el notebook del curso usaba Huber).
- **Optimización:** AdamW con LR 2e-4, warmup lineal de 1000 updates y luego coseno,
  lote 32, fp16 con GradScaler y recorte de gradiente a 1.0.
- **EMA 0.999:** los pesos que se guardan y se usan para generar son la media móvil,
  que da muestras más estables que los pesos crudos.
- **Selección:** `best.safetensors` es el EMA con menor MSE de validación. Esa MSE se
  mide con un ruido y un *t* fijos por imagen, así que es comparable entre validaciones.
- **Corrida:** `runs/anime64_4070ti` hizo 87 274 updates (2.79 M imágenes vistas, unas
  84 épocas) en una RTX 4070 Ti de 12 GB en 5.8 h. La MSE de validación final fue
  0.02388. El perfil es `configs/rtx4070ti_12gb.toml`.

## 2. Datos y particiones

`scripts/prepare_dataset.py` recorta cada imagen al centro a 64×64 (Lanczos) y
descarta los duplicados exactos: de 63 565 archivos se eliminan 22 043. Después agrupa
los casi duplicados con un dHash de 64 bits que no cambia al reflejar la imagen. Cada
grupo va entero a una sola partición, para que test no tenga copias de imágenes de
train. El reparto es 33 218 train / 4 152 val / 4 152 test.

El resultado está en `data/dataset_splits.json`, que **está versionado**: define qué
imágenes son de test. Regenerarlo en otra máquina podría pasar a train imágenes que antes
eran de test, y con eso cualquier FID dejaría de ser válido.

## 3. Cómo se mide

`scripts/evaluate_quality.py` produce un informe con tres partes:

- **FID y KID** a 64×64 entre las imágenes generadas (DDIM, 50 pasos) y las reales de
  test, con extractor Inception (TorchMetrics). Se usan N = 4 152 imágenes, todas las de
  test.
- **Diversidad:** SSIM entre pares de imágenes generadas. Siempre se compara generada
  contra generada, nunca contra una real.
- **Copia del dataset:** para cada imagen generada busca la de train más cercana por
  dHash y deja una lámina para revisarlas a ojo.

Límites de estas medidas:

- Inception se entrenó con fotos naturales, así que en anime FID y KID son orientativos.
- La desviación de KID describe la variación entre subconjuntos; no es un intervalo de
  confianza entre entrenamientos.
- Con menos de 5 000 imágenes, el FID queda sesgado hacia arriba.
- Un vecino cercano por dHash no prueba memorización ni la descarta.
- Hay una sola corrida con una sola semilla: las conclusiones son exploratorias.

Val se usa para decidir: elegir el checkpoint y el número de pasos de muestreo. Test se
usa solo para el número final.

## 4. Resultados

| Medida | Valor |
|---|---|
| FID (test, N = 4 152, DDIM 50) | **10.099** |
| KID | **0.003942 ± 0.000463** |
| MSE de validación (EMA) | 0.02388 |
| SSIM medio entre generadas | 0.119 |
| Distancia dHash media al vecino de train | 8.9 bits de 64 |
| Inferencia | 0.118 s/imagen, 1.03 GiB de VRAM (4070 Ti, lote 32) |

Los archivos están en `reports/samples/anime64_4070ti/`. Las muestras, el GIF del
proceso de eliminación de ruido y las métricas se generaron con el código anterior a la
refactorización del 12 de noviembre. El código nuevo genera las mismas imágenes con las
mismas semillas, salvo diferencias mínimas de fp16 entre GPUs.

## 5. Exploración SDEdit: de una foto real a un rostro anime

**Pregunta.** El modelo solo aprendió a convertir ruido en caras anime y nunca vio una
foto real. ¿Se puede partir de una foto en vez de ruido puro?

**Respuesta.** Sí, sin reentrenar nada. El proceso directo es una fórmula cerrada, no una
red: `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1−ᾱ_t)·ε`. SDEdit (Meng et al., 2021) aplica esa
fórmula a la foto hasta un paso intermedio y desde ahí continúa el muestreo normal con
los pesos entrenados. `--strength` decide cuánto ruido se añade:

- con 0.3 se conserva casi toda la foto;
- con 0.9 el resultado se acerca mucho a una cara anime libre.

**Qué no es.** No es traducción de dominio controlada. No hay codificador, ni pares
foto-anime, ni condicionamiento, y no se conserva la identidad. Es una propiedad general
de los modelos de difusión no condicionados aplicada a los pesos que ya existen.

**Cómo se evalúa.** Solo de forma cualitativa: una fila por foto con la original y el
resultado a 0.3, 0.5, 0.7 y 0.9. No se calcula FID ni KID, porque no existe "la"
traducción correcta de una foto con la cual comparar, y un número así daría una falsa
sensación de rigor.

```bash
uv run python scripts/sdedit.py --weights runs/anime64_4070ti/best.safetensors \
    --source data/real_faces_demo
```

## 6. Qué se dejó fuera, y por qué

- **Una red de traducción foto→anime entrenada** (CycleGAN, img2img condicionado).
  Sería un segundo modelo que entrenar, y no cabía antes de la exposición del
  13 de noviembre.

## Fuentes

- Ho et al., *Denoising Diffusion Probabilistic Models*: https://arxiv.org/abs/2006.11239
- Song et al., *Denoising Diffusion Implicit Models*: https://arxiv.org/abs/2010.02502
- Meng et al., *SDEdit*: https://arxiv.org/abs/2108.01073
- https://huggingface.co/docs/diffusers/api/schedulers/ddim
- https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
