"""Crear diapositivas editables y guion, sin ejecutar modelos ni el notebook.

Desde pc3/: uv run --no-sync python tools/build_presentation.py [--require-results]
Añadir --pdf y, si es necesario, --soffice /ruta/a/soffice para exportar y revisar.
"""

import argparse
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pptx import Presentation
from pptx.enum.dml import MSO_THEME_COLOR as Color
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE as Shape
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as Rel
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.shapes.shapetree import SlideShapes
from pptx.util import Inches, Pt

from pc3.config import Config, load_config, training_contract
from pc3.data import resolve_image
from pc3.io import file_sha256, read_json, source_fingerprint, write_json
from pc3.metrics import assert_comparable

ROOT = Path(__file__).resolve().parents[1]
NAME = "pc3_difusion_anime"
CASES = [
    ("random", "from_scratch", "Desde cero", Color.ACCENT_1),
    ("pretrained_reference", "pretrained", "Abierto sin adaptar", Color.ACCENT_3),
    ("pretrained", "finetuned", "Fine-tuning", Color.ACCENT_2),
]
THEME = {
    "dk1": "21162B",
    "lt1": "FFFFFF",
    "dk2": "5A5264",
    "lt2": "F5F2F8",
    "accent1": "9B1B69",
    "accent2": "007D88",
    "accent3": "B64922",
    "accent4": "D7CDE6",
    "accent5": "E6F3F2",
    "accent6": "FBEAF4",
    "hlink": "007D88",
    "folHlink": "9B1B69",
}


def load_evidence(root: Path, config: Config) -> list[dict]:
    """Fallar ante resultados inválidos; ausencia no equivale a una métrica cero.

    Solo se incorporan los cuatro primeros PNG por caso, sin escoger los mejores.
    La comparación y las muestras deben referirse a los mismos pesos/datos/código.
    """
    path = root / "reports/comparison/comparison.json"
    if not path.is_file():
        return []
    try:
        rows = read_json(path)
        assert_comparable(rows)
        if len(rows) != 3 or {r["provenance"]["initialization"] for r in rows} != {
            case[0] for case in CASES
        }:
            raise ValueError("Se requieren los tres casos del experimento.")
        if rows[0]["protocol"]["split"] != "test":
            raise ValueError("La comparación final debe usar test, no validación.")
        result = []
        for initialization, folder, label, _ in CASES:
            row = next(r for r in rows if r["provenance"]["initialization"] == initialization)
            protocol = row["protocol"]
            if (
                row["provenance"]["contract"] != training_contract(config)
                or protocol["pc3_source_sha256"] != source_fingerprint()
            ):
                raise ValueError("Código o perfil distintos de las evidencias.")
            if protocol["sampler"] != "ddim" or protocol["sampling_steps"] != config.sampling_steps:
                raise ValueError("Los pasos DDIM no coinciden con el perfil presentado.")
            if any(
                not math.isfinite(row[key])
                for key in ("fid", "kid_mean", "kid_std", "seconds_per_image")
            ):
                raise ValueError("Métrica no finita.")
            directory = root / "reports/samples" / folder
            generation = read_json(directory / "generation.json")
            if (
                not generation["complete"]
                or generation["count"] != len(generation["images"])
                or generation["count"] != row["generation"]["count"]
                or generation["count"] < protocol["num_images"]
                or protocol["num_images"] < 4
                or generation["source_hash"] != row["generation"]["source_hash"]
                or generation["provenance"] != row["provenance"]
                or generation["manifest_sha256"] != protocol["manifest_sha256"]
                or generation["manifest_sha256"] != row["provenance"]["manifest_sha256"]
                or generation["pc3_source_sha256"] != protocol["pc3_source_sha256"]
            ):
                raise ValueError("Métricas y muestras de ejecuciones diferentes/incompletas.")
            images = []
            for record in generation["images"][:4]:
                image = resolve_image(directory, record["path"])
                if file_sha256(image) != record["sha256"]:
                    raise ValueError("Una muestra cambió después de su generación.")
                with Image.open(image) as pixels:
                    if pixels.size != (64, 64) or pixels.mode != "RGB":
                        raise ValueError("La muestra debe ser RGB64.")
                    pixels.verify()
                images.append(image)
            result.append({"label": label, "metrics": row, "images": images})
        return result
    except (KeyError, TypeError, StopIteration, OSError, ValueError) as error:
        raise ValueError(f"No se pueden presentar estas evidencias: {error}") from error


def talk_plan(evidence: list[dict], config: Config | None = None) -> list[dict]:
    cfg = config or Config()
    budget = f"{cfg.max_steps:,}".replace(",", " ")
    result_speech = (
        "Esta es la comparación preparada para los tres casos. Todavía no hay muestras, "
        "métricas ni historiales de entrenamiento reales, por eso aparecen como pendientes "
        "y no como ceros. No podemos afirmar que el fine-tuning sea mejor ni que el perfil "
        "quepa en ocho gigabytes. Al ejecutar, aquí se mostrarán las cuatro primeras "
        "semillas de cada caso, sin seleccionar solo imágenes favorables, junto con FID, "
        "KID y segundos por imagen. También revisaremos las curvas para detectar "
        "sobreajuste. El código ya produce esas evidencias; las pruebas auxiliares en CPU "
        "verifican herramientas, pero no reemplazan un entrenamiento ni demuestran calidad."
    )
    closing_speech = (
        "La conclusión actual es metodológica: dejamos una comparación reproducible "
        "que separa el efecto de la inicialización del entrenamiento local. El software "
        "y el protocolo están preparados; falta ejecutar ambos modelos, comprobar "
        "memoria y tiempos, e incorporar muestras y métricas reales. El fine-tuning es "
        "una hipótesis, no un resultado garantizado. Solo después de esas mediciones "
        "podremos responder cuál conviene con este presupuesto. Gracias."
    )
    if evidence:
        values = []
        for item in evidence:
            row = item["metrics"]
            values.append(
                f"{item['label']}: FID {row['fid']:.2f}, KID {row['kid_mean']:.3g}, "
                f"y {row['seconds_per_image']:.3f} segundos por imagen"
            )
        result_speech = (
            "Estas son las cuatro primeras semillas de cada caso, no una selección de "
            "las mejores caras. " + "; ".join(values) + ". Comparen también artefactos "
            "y diversidad: el menor FID no sustituye la inspección visual. La desviación "
            "de KID corresponde a subconjuntos, no a entrenamientos independientes. "
            "El tiempo es amortizado por imagen y excluye guardar PNG. Los pasos de los "
            "pesos seleccionados aparecen debajo de cada modelo. Esta ejecución permite "
            "una comparación exploratoria, no una afirmación de superioridad universal."
        )
        closing_speech = (
            "Con estas evidencias podemos describir el efecto observado de adaptar los "
            "pesos y compararlo con entrenar desde cero. No confundimos menor FID con "
            "significancia estadística ni con mejor rendimiento en todo escenario. "
            "El aporte es un experimento reproducible que conserva datos, arquitectura "
            "y presupuesto local, y declara la ventaja del preentrenamiento externo. "
            "Para fortalecer la conclusión, repetiríamos con varias semillas de "
            "entrenamiento y revisaríamos diversidad y posibles copias. Gracias."
        )
    content = [
        (
            25,
            "¿El preentrenamiento ayuda a generar anime?",
            "Pregunta",
            "Nuestro proyecto estudia la generación de rostros anime mediante difusión. "
            "La pregunta es si, con el mismo presupuesto local, conviene aprender desde cero "
            "o adaptar pesos abiertos. Vamos a explicar el modelo, los controles de la "
            "comparación y cómo se evaluará sin asumir de antemano un ganador.",
            "Abrir con la pregunta; no presentar el diagrama como una imagen del modelo.",
        ),
        (
            55,
            "Un DDPM aprende a quitar ruido",
            "Fundamentos",
            "La difusión tiene dos procesos. En el directo, añadimos ruido gaussiano a una "
            "imagen hasta destruir su estructura. La ecuación combina señal y ruido según "
            "el tiempo. Durante el entrenamiento escogemos un tiempo aleatorio y la U-Net "
            "recibe la imagen corrompida y ese tiempo para predecir el ruido añadido. "
            "Minimizamos el error cuadrático entre el ruido real y el estimado. Para generar, "
            "partimos de ruido y aplicamos el proceso inverso de forma iterativa. Usamos "
            f"DDIM con {cfg.sampling_steps} pasos de inferencia. Los mil pasos del calendario "
            "no significan mil pasadas de la red por imagen al entrenar: "
            "solo se muestrea un tiempo.",
            "Señalar señal → ruido y luego U-Net. Explicar la ecuación; "
            "no leer símbolos uno por uno.",
        ),
        (
            50,
            "Mismos datos. Tres puntos de partida",
            "Experimento",
            "Usamos Anime Face Dataset de Kaggle, versión tres, con recorte central a "
            "sesenta y cuatro píxeles RGB. Eliminamos duplicados exactos y agrupamos por "
            "un hash perceptual que también contempla espejos. Esos grupos no cruzan "
            "entrenamiento, validación y test; el reparto previsto es ochenta, diez y diez. "
            "Comparamos la misma U-Net aleatoria, un checkpoint abierto sin adaptar y ese "
            "mismo checkpoint ajustado con todos sus parámetros. Los pesos abiertos "
            "provienen de flores, no anime, para reducir el riesgo conocido de solapamiento "
            "con test. No podemos garantizar ausencia total de similitud entre imágenes.",
            "Distinguir B, referencia sin entrenamiento local, de C, fine-tuning completo.",
        ),
        (
            50,
            "8 GB: reducir memoria, no la comparación",
            "Implementación",
            f"El perfil prevé {budget} actualizaciones por cada modelo entrenado, "
            f"usando un microbatch de {cfg.batch_size} y acumulación de {cfg.accumulation_steps}. "
            f"El lote efectivo es {cfg.batch_size * cfg.accumulation_steps}. Para reducir memoria "
            "usamos precisión mixta, checkpointing de gradientes y la copia EMA en CPU. "
            "Ambos casos corren secuencialmente en la misma GPU y conservan datos, orden "
            "de ejemplos, objetivo y optimizador. Seleccionamos los pesos EMA con la menor "
            "MSE de validación sobre ruido fijo, nunca mirando test. Los checkpoints guardan "
            "optimizador, posición y azar para reanudar. Es un presupuesto previsto: hay "
            "que calibrar memoria y tiempo en la GPU real antes de prometer una noche.",
            "Aclarar que un update no es una época; el preentrenamiento externo no se iguala.",
        ),
        (
            45,
            "Medir calidad también exige controles",
            "Evaluación",
            "La calidad se evalúa con FID y KID; valores menores indican mayor cercanía "
            "distributiva bajo el mismo protocolo. Se usará el mismo test y el mismo "
            "número de imágenes, hasta cinco mil, con iguales semillas y pasos de muestreo. "
            "Medimos además segundos por imagen y memoria. La MSE selecciona pesos, pero "
            "no garantiza buenas caras. Inception fue entrenada en imágenes naturales y "
            "tiene límites en anime. La desviación de KID entre subconjuntos no es un "
            "intervalo de confianza entre entrenamientos. Una sola semilla de entrenamiento "
            "solo permite conclusiones exploratorias.",
            "No leer FID/KID como accuracy. Si preguntan por SSIM/dHash, usar el anexo.",
        ),
        (
            65,
            "Resultados: medir antes de concluir",
            "Resultados",
            result_speech,
            "Con resultados, señalar A → B → C y dar una interpretación visual, "
            "no solo leer cifras. "
            "Sin resultados, decirlo explícitamente. No ejecutar una demo en vivo.",
        ),
        (
            40,
            "La conclusión debe seguir a las evidencias",
            "Cierre",
            closing_speech,
            "Cerrar aquí. Los anexos no forman parte del recorrido de 5:30.",
        ),
    ]
    return [
        {"seconds": seconds, "title": title, "section": section, "speech": speech, "cue": cue}
        for seconds, title, section, speech, cue in content
    ]


def text(
    owner,
    value,
    x,
    y,
    w,
    h,
    size=18,
    color=Color.DARK_1,
    *,
    bold=False,
    align=PP_ALIGN.LEFT,
    name="Texto",
):
    box = owner.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for index, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(3)
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.theme_color = color
        paragraph._p.get_or_add_pPr().get_or_add_defRPr().set("lang", "es-PE")
    return box


def block(slide, x, y, w, h, color=Color.LIGHT_2, kind=Shape.ROUNDED_RECTANGLE, name="Bloque"):
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.theme_color = color
    shape.line.fill.background()
    shape._element.find(qn("p:style")).find(qn("a:effectRef")).set("idx", "0")
    if kind == Shape.ROUNDED_RECTANGLE:
        shape.adjustments[0] = 0.10
    return shape


def arrow(slide, x, y, color=Color.ACCENT_4, w=0.45):
    block(slide, x, y, w, 0.30, color, Shape.CHEVRON, "Dirección del proceso")


def pixels(slide, x, y, size, noise=0.0):
    """Mosaico vectorial conceptual, no una imagen extraída de un modelo o dataset."""
    face = [
        "00111100",
        "01111110",
        "01222210",
        "12222221",
        "12211221",
        "12222221",
        "01222210",
        "00111100",
    ]
    palette = [Color.ACCENT_6, Color.ACCENT_1, Color.ACCENT_4]
    rng = random.Random(25)
    for row in range(8):
        for col in range(8):
            index = rng.randrange(3) if rng.random() < noise else int(face[row][col])
            block(
                slide,
                x + col * size / 8,
                y + row * size / 8,
                size / 8 - 0.006,
                size / 8 - 0.006,
                palette[index],
                Shape.RECTANGLE,
                "Píxel esquemático",
            )


def _theme(presentation):
    part = next(
        r.target_part
        for r in presentation.slide_master.part.rels.values()
        if r.reltype == Rel.THEME
    )
    theme = parse_xml(part.blob)
    theme.set("name", "PC3 · Píxeles y difusión")
    scheme = theme.find(".//" + qn("a:clrScheme"))
    scheme.set("name", "Anime · tinta, frambuesa y teal")
    for name, value in THEME.items():
        element = scheme.find(qn(f"a:{name}"))
        for child in list(element):
            element.remove(child)
        child = parse_xml(
            f'<a:srgbClr xmlns:a="http://schemas.openxmlformats.org/'
            f'drawingml/2006/main" val="{value}"/>'
        )
        element.append(child)
    for name in ("majorFont", "minorFont"):
        theme.find(f".//{qn('a:' + name)}/{qn('a:latin')}").set("typeface", "Arial")
    from lxml import etree

    part._blob = etree.tostring(theme, xml_declaration=True, encoding="UTF-8", standalone=True)


def _layouts(presentation):
    _theme(presentation)
    layouts = []
    for index, name, dark in [(0, "PC3 · Portada y cierre", True), (5, "PC3 · Contenido", False)]:
        layout = presentation.slide_layouts[index]
        layout.name = name
        for placeholder in list(layout.placeholders):
            if placeholder.placeholder_format.idx != 0:
                placeholder._element.getparent().remove(placeholder._element)
        title = layout.placeholders[0]
        title.left, title.top = Inches(0.65), Inches(0.92)
        title.width, title.height = Inches(12.0), Inches(0.75)
        paragraph = title.text_frame.paragraphs[0]
        paragraph.font.size = Pt(36)
        paragraph.font.bold = True
        paragraph.font.color.theme_color = Color.LIGHT_1 if dark else Color.DARK_1
        paragraph.alignment = PP_ALIGN.LEFT
        title.text_frame.margin_left = title.text_frame.margin_right = 0
        title.text_frame.vertical_anchor = MSO_ANCHOR.TOP
        layout.background.fill.solid()
        layout.background.fill.fore_color.theme_color = Color.DARK_1 if dark else Color.LIGHT_1
        canvas = SimpleNamespace(shapes=SlideShapes(layout._element.spTree, layout))
        text(
            canvas,
            "PC3  /  COMPUTACIÓN GRÁFICA",
            0.65,
            7.04,
            6,
            0.2,
            10,
            Color.ACCENT_4 if dark else Color.DARK_2,
            name="Pie heredado del diseño",
        )
        text(
            canvas,
            "13 NOV 2026",
            10.65,
            7.04,
            2,
            0.2,
            10,
            Color.ACCENT_4 if dark else Color.DARK_2,
            align=PP_ALIGN.RIGHT,
            name="Fecha heredada del diseño",
        )
        layouts.append(layout)
    return layouts


def set_title(slide, value, *, dark=False, size=36):
    title = slide.shapes.title
    title.text = value
    for paragraph in title.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.LEFT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = True
        paragraph.font.color.theme_color = Color.LIGHT_1 if dark else Color.DARK_1
        paragraph.font.name = "+mj-lt"
    return title


def new_slide(presentation, layout, item, number, hidden=False):
    slide = presentation.slides.add_slide(layout)
    # Usar el placeholder; aplicar estilo explícito para que LibreOffice y PowerPoint
    # coincidan incluso al sobrescribir el texto de un título heredado del diseño.
    set_title(slide, item["title"], dark=layout == presentation.slide_layouts[0])
    text(
        slide,
        f"{number:02}  /  {item['section'].upper()}",
        0.65,
        0.43,
        11.9,
        0.25,
        12,
        Color.ACCENT_4 if layout == presentation.slide_layouts[0] else Color.ACCENT_1,
        bold=True,
        name="Sección",
    )
    slide.notes_slide.notes_text_frame.text = (
        f"{item.get('seconds', 0)} segundos previstos\n\n{item['speech']}\n\n"
        f"INDICACIÓN: {item['cue']}"
    )
    if hidden:
        slide._element.set("show", "0")
    return slide


def _cover(slide, pending):
    title = set_title(slide, "¿Aprender desde cero\no adaptar lo aprendido?", dark=True, size=43)
    title.left = Inches(0.65)
    title.top, title.width, title.height = Inches(1.35), Inches(8.1), Inches(2.0)
    text(
        slide,
        "Difusión para generar\nrostros de anime",
        0.65,
        3.8,
        7.8,
        1.1,
        28,
        Color.LIGHT_1,
        name="Tema",
    )
    text(slide, "PyTorch  ·  DDPM  ·  RGB 64 × 64", 0.65, 5.25, 7.8, 0.4, 19, Color.ACCENT_4)
    text(
        slide,
        "Ejecución pendiente" if pending else "Comparación con evidencias ejecutadas",
        0.65,
        6.1,
        7.8,
        0.4,
        16,
        Color.ACCENT_4,
    )
    pixels(slide, 9.05, 1.7, 1.95, noise=1)
    arrow(slide, 10.9, 3.85, Color.ACCENT_4)
    pixels(slide, 10.25, 4.3, 1.95)
    text(slide, "Ruido → estructura", 8.95, 6.4, 3.6, 0.25, 14, Color.ACCENT_4)
    text(
        slide, "Esquema conceptual, no salida del modelo", 8.95, 6.74, 3.8, 0.2, 10, Color.ACCENT_4
    )


def _diffusion(slide, cfg):
    for index, (label, amount) in enumerate(
        [("Imagen x₀", 0), ("Imagen ruidosa xₜ", 0.55), ("Ruido xT", 1)]
    ):
        x = 0.85 + index * 2.45
        pixels(slide, x, 2.13, 1.43, noise=amount)
        text(slide, label, x, 3.8, 2.1, 0.4, 16)
        if index < 2:
            arrow(slide, x + 1.78, 2.7)
    block(slide, 8.55, 2.08, 4.1, 2.25)
    text(slide, "U-Net(xₜ, t)", 8.9, 2.35, 3.4, 0.5, 27, bold=True)
    text(slide, "Predice el ruido ε\nObjetivo: MSE", 8.9, 3.12, 3.4, 0.9, 21)
    text(
        slide,
        "xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε",
        0.85,
        4.66,
        11.8,
        0.65,
        30,
        Color.ACCENT_1,
        name="Proceso directo en forma cerrada",
    )
    text(slide, "Entrenar: muestrear t y predecir ε", 0.85, 5.56, 6.2, 0.4, 19)
    text(slide, f"Generar: ruido → DDIM {cfg.sampling_steps} → imagen", 7.2, 5.56, 5.5, 0.65, 19)
    text(
        slide,
        "Diagrama didáctico, no muestras reales · Ho et al. (2020); Song et al. (2020)",
        0.85,
        6.45,
        11.8,
        0.3,
        11,
        Color.DARK_2,
    )


def _experiment(slide, cfg):
    text(
        slide,
        "Anime Face Dataset v3 → RGB64 → deduplicación → particiones por grupos",
        0.65,
        1.83,
        12.0,
        0.4,
        20,
    )
    for x, w, label, color in [
        (0.65, 9.1, "80% train", Color.ACCENT_6),
        (10.05, 1.15, "10%\nval", Color.ACCENT_5),
        (11.5, 1.15, "10%\ntest", Color.LIGHT_2),
    ]:
        block(slide, x, 2.5, w, 0.85, color)
        text(slide, label, x + 0.14, 2.66, w - 0.28, 0.6, 17, bold=True)
    text(
        slide,
        "Reparto aproximado previsto; ningún grupo cruza particiones",
        0.65,
        3.52,
        12.0,
        0.35,
        14,
        Color.DARK_2,
    )
    for index, (_, _, label, color) in enumerate(CASES):
        x = 0.65 + index * 4.1
        block(slide, x, 4.14, 3.8, 1.65)
        text(slide, f"{chr(65 + index)}  {label}", x + 0.23, 4.4, 3.35, 0.55, 22, color, bold=True)
        description = ["Pesos aleatorios", "Pesos de flores", "Flores → anime"][index]
        text(slide, description, x + 0.23, 5.02, 3.35, 0.4, 19)
    text(
        slide,
        "A y C: misma U-Net y mismos datos; se ajustan todos sus parámetros",
        0.65,
        6.06,
        12.0,
        0.4,
        18,
        bold=True,
    )
    text(
        slide,
        f"Pesos abiertos: {cfg.pretrained_id} · Apache-2.0 · SHA fijado en el anexo",
        0.65,
        6.63,
        12.0,
        0.22,
        11,
        Color.DARK_2,
    )


def _training(slide, cfg):
    effective = cfg.batch_size * cfg.accumulation_steps
    text(
        slide,
        f"{cfg.batch_size} × {cfg.accumulation_steps} = {effective}",
        0.65,
        2.03,
        6.0,
        1.1,
        60,
        Color.ACCENT_1,
        bold=True,
    )
    text(slide, "Microbatch × acumulación = lote efectivo", 0.65, 3.32, 6.1, 0.6, 18)
    for i, (label, color) in enumerate(
        [("A · Desde cero", Color.ACCENT_1), ("C · Fine-tuning", Color.ACCENT_2)]
    ):
        block(slide, 7.2, 2.12 + i * 1.15, 5.45, 0.85)
        text(slide, label, 7.45, 2.3 + i * 1.15, 2.65, 0.4, 19, color, bold=True)
        text(
            slide,
            f"{cfg.max_steps:,} updates".replace(",", " "),
            10.05,
            2.3 + i * 1.15,
            2.4,
            0.4,
            18,
            color,
        )
    for i, label in enumerate(["FP16 + GradScaler", "Checkpointing de gradientes", "EMA en CPU"]):
        block(slide, 0.65, 4.32 + i * 0.54, 0.16, 0.16, Color.ACCENT_2, Shape.OVAL)
        text(slide, label, 1.0, 4.23 + i * 0.54, 5.8, 0.4, 20)
    text(slide, "Una GPU · ejecuciones secuenciales", 7.2, 4.45, 5.45, 0.4, 19)
    block(slide, 7.2, 5.13, 5.45, 1.08, Color.ACCENT_5)
    text(
        slide,
        "Elegir pesos EMA con\nMSE de validación fija",
        7.45,
        5.32,
        4.95,
        0.8,
        21,
        Color.ACCENT_2,
        bold=True,
    )
    text(
        slide,
        "Presupuesto previsto; falta calibrar en 8 GB · El preentrenamiento externo no se iguala",
        0.65,
        6.62,
        12.0,
        0.25,
        11,
        Color.DARK_2,
    )


def _evaluation(slide, cfg):
    cards = [
        ("01", "Calidad", "FID + KID ↓", "Distribuciones,\nno accuracy", Color.ACCENT_1),
        (
            "02",
            "Muestras",
            f"Hasta {cfg.sample_count:,}".replace(",", " "),
            "Mismo test, N\ny semillas",
            Color.ACCENT_2,
        ),
        ("03", "Coste", "s/imagen + VRAM", "Misma GPU\ny lote de inferencia", Color.ACCENT_3),
    ]
    for index, (number, title, value, detail, color) in enumerate(cards):
        x = 0.65 + index * 4.1
        block(slide, x, 2.1, 3.8, 3.15)
        block(slide, x + 0.25, 2.38, 0.62, 0.62, color, Shape.OVAL)
        text(
            slide,
            number,
            x + 0.3,
            2.54,
            0.52,
            0.28,
            15,
            Color.LIGHT_1,
            bold=True,
            align=PP_ALIGN.CENTER,
        )
        text(slide, title, x + 1.05, 2.52, 2.45, 0.4, 23, color, bold=True)
        text(slide, value, x + 0.25, 3.36, 3.3, 0.55, 25, color, bold=True)
        text(slide, detail, x + 0.25, 4.21, 3.3, 0.8, 20)
    text(
        slide,
        f"Protocolo principal: DDIM {cfg.sampling_steps} · Test solo para evaluar",
        0.65,
        5.7,
        12,
        0.4,
        21,
        bold=True,
    )
    text(
        slide,
        "Inception tiene límites en anime · Std KID ≠ intervalo de confianza entre entrenamientos",
        0.65,
        6.35,
        12,
        0.55,
        14,
        Color.DARK_2,
    )


def _results(slide, evidence):
    pending = not evidence
    subtitle = (
        "Sin ejecución real: las métricas no se sustituyen por ceros"
        if pending
        else f"Test · N={evidence[0]['metrics']['protocol']['num_images']} · "
        "4 primeras semillas por caso, sin selección manual"
    )
    text(slide, subtitle, 0.65, 1.79, 12, 0.4, 17, Color.DARK_2)
    for index, (_, _, label, color) in enumerate(CASES):
        x = 0.65 + index * 4.1
        block(slide, x, 2.42, 3.8, 3.95)
        text(slide, label, x + 0.2, 2.65, 3.4, 0.4, 22, color, bold=True)
        if pending:
            text(
                slide,
                "Sin muestras",
                x + 0.25,
                3.38,
                3.3,
                0.5,
                24,
                Color.DARK_2,
                align=PP_ALIGN.CENTER,
            )
            text(
                slide,
                "Pendientes de entrenamiento",
                x + 0.25,
                4.0,
                3.3,
                0.3,
                14,
                Color.DARK_2,
                align=PP_ALIGN.CENTER,
            )
        else:
            for image_index, image in enumerate(evidence[index]["images"]):
                picture = slide.shapes.add_picture(
                    str(image),
                    Inches(x + 1.05 + (image_index % 2) * 0.88),
                    Inches(3.17 + (image_index // 2) * 0.88),
                    width=Inches(0.82),
                    height=Inches(0.82),
                )
                picture.name = f"Semilla predefinida {image_index} · {label}"
        values = ["Pendiente"] * 3
        if not pending:
            row = evidence[index]["metrics"]
            values = [
                f"{row['fid']:.2f}",
                f"{row['kid_mean']:.3g} ± {row['kid_std']:.3g}",
                f"{row['seconds_per_image']:.3f}",
            ]
        for metric_index, (metric, value) in enumerate(
            zip(["FID ↓", "KID ↓", "s/imagen ↓"], values, strict=True)
        ):
            y = 5.04 + metric_index * 0.38
            text(slide, metric, x + 0.2, y, 1.42, 0.3, 15)
            text(slide, value, x + 1.65, y, 1.95, 0.3, 14, color, bold=True)
        step = (
            "Sin checkpoint"
            if pending
            else f"EMA/pesos: paso {evidence[index]['metrics']['provenance']['step']}"
        )
        text(slide, step, x + 0.2, 6.15, 3.4, 0.2, 10, Color.DARK_2)
    text(
        slide,
        "Aún no se puede elegir un ganador"
        if pending
        else "KID: media ± std de subconjuntos; comparación exploratoria de una ejecución",
        0.65,
        6.65,
        12,
        0.25,
        14,
        Color.DARK_2,
    )


def _closing(slide, evidence):
    title = set_title(
        slide,
        (
            "Adaptar pesos no garantiza\nmejores rostros"
            if not evidence
            else "La comparación es local,\nno una superioridad universal"
        ),
        dark=True,
        size=39,
    )
    title.left = Inches(0.65)
    title.top, title.height, title.width = Inches(1.35), Inches(2.3), Inches(8.0)
    text(
        slide,
        "Controlar datos + arquitectura + presupuesto\n"
        "permite estudiar el efecto de la inicialización",
        0.65,
        4.2,
        7.4,
        1.0,
        24,
        Color.LIGHT_1,
    )
    text(
        slide,
        "Siguiente paso: "
        + ("ejecutar y medir" if not evidence else "repetir con varias semillas"),
        0.65,
        6.12,
        7.5,
        0.4,
        19,
        Color.ACCENT_4,
    )
    for i, (heading, body) in enumerate(
        [
            ("Listo", "Código y protocolo\nPruebas auxiliares CPU"),
            ("Por medir", "Memoria y tiempos\nCalidad y diversidad"),
        ]
        if not evidence
        else [
            ("Observado", "Muestras y métricas\nPresupuesto local común"),
            ("Por fortalecer", "Variabilidad entre semillas\nInspección y diversidad"),
        ]
    ):
        text(slide, heading, 9.0, 2.03 + 2.08 * i, 3.7, 0.5, 29, Color.LIGHT_1, bold=True)
        text(slide, body, 9.0, 2.74 + 2.08 * i, 3.7, 0.85, 20, Color.ACCENT_4)


def _appendices(presentation, layout, cfg):
    questions = [
        (
            "¿Por qué un checkpoint de flores?",
            "Reduce el riesgo conocido de solapamiento anime.\n"
            "La transferencia puede ayudar o no; hay que medirla.",
        ),
        (
            "¿Fine-tuning significa sobreajuste?",
            "No. Es adaptar pesos. Sobreajuste es mejorar train\n"
            "sin generalizar a datos de validación.",
        ),
        (
            "¿Cómo detectar diversidad o copias?",
            "SSIM entre generadas y vecinos dHash de train\n"
            "son proxies, no pruebas definitivas de calidad o copia.",
        ),
        (
            "¿Qué demuestran las pruebas CPU?",
            "Verifican herramientas y controles auxiliares.\n"
            "No prueban ajuste en GPU, convergencia ni FID real.",
        ),
    ]
    item = {
        "title": "Respuestas para la discusión",
        "section": "Anexo · preguntas",
        "speech": "\n\n".join(f"{q}\n{a}" for q, a in questions),
        "cue": "Diapositiva oculta. Abrir solo si una pregunta lo requiere.",
    }
    slide = new_slide(presentation, layout, item, 8, hidden=True)
    for index, (question, answer) in enumerate(questions):
        x, y = 0.65 + (index % 2) * 6.25, 2.1 + (index // 2) * 2.15
        block(slide, x, y, 5.75, 1.8)
        text(slide, question, x + 0.22, y + 0.22, 5.3, 0.55, 21, Color.ACCENT_1, bold=True)
        text(slide, answer, x + 0.22, y + 0.95, 5.3, 0.7, 15)
    sources = [
        ("DDPM · Ho et al. (2020)", "https://arxiv.org/abs/2006.11239"),
        ("DDIM · Song et al. (2020)", "https://arxiv.org/abs/2010.02502"),
        ("FID / TTUR · Heusel et al. (2017)", "https://arxiv.org/abs/1706.08500"),
        ("KID · Bińkowski et al. (2018)", "https://arxiv.org/abs/1801.01401"),
        (
            "Anime Face Dataset · splcher · versión 3",
            "https://www.kaggle.com/datasets/splcher/animefacedataset",
        ),
        (cfg.pretrained_id + " · Apache-2.0", "https://huggingface.co/" + cfg.pretrained_id),
    ]
    item = {
        "title": "Fuentes y trazabilidad",
        "section": "Anexo · referencias",
        "speech": "\n".join(f"{title}: {url}" for title, url in sources)
        + f"\nRevisión del checkpoint: {cfg.pretrained_revision}\n"
        "Base docente: semanas 8 y 9. Cambio explícito: Huber → MSE y "
        "TensorFlow/CelebA → PyTorch/anime. Arquitectura de Diffusers; "
        "bucles y herramientas en src/pc3/. Las imágenes y métricas, "
        "cuando existan, proceden de reports/, no de ilustraciones externas.",
        "cue": "Diapositiva oculta; los enlaces son seleccionables en PowerPoint.",
    }
    slide = new_slide(presentation, layout, item, 9, hidden=True)
    for index, (title, url) in enumerate(sources):
        box = text(
            slide,
            title,
            0.85,
            2.0 + index * 0.52,
            11.8,
            0.4,
            20,
            Color.ACCENT_2,
            name="Referencia enlazada",
        )
        box.text_frame.paragraphs[0].runs[0].hyperlink.address = url
    text(slide, "Revisión fijada del checkpoint", 0.85, 5.47, 11.8, 0.3, 14, bold=True)
    text(slide, cfg.pretrained_revision, 0.85, 5.88, 11.8, 0.3, 14)
    text(
        slide,
        "Curso: semanas 8 y 9 · Huber → MSE · TensorFlow/CelebA → PyTorch/anime",
        0.85,
        6.4,
        11.8,
        0.4,
        14,
        Color.DARK_2,
    )


def clock(seconds):
    return f"{seconds // 60}:{seconds % 60:02}"


def _script(plan, output, pending):
    lines = [
        "# Guion de exposición — PC3",
        "",
        "**Duración planificada: 5:30. Tope de ensayo: 6:00.** Los tiempos son un "
        "presupuesto, no una duración medida; ensayar en voz alta con cronómetro.",
        "",
        "**Estado: "
        + (
            "sin resultados reales; no defender como comparación ejecutada.**"
            if pending
            else "evidencias incorporadas; verificar interpretación.**"
        ),
        "",
        "| Diapositiva | Tramo | Duración |",
        "|---|---|---|",
    ]
    elapsed = 0
    for index, item in enumerate(plan, 1):
        lines.append(
            f"| {index}. {item['section']} | {clock(elapsed)}–"
            f"{clock(elapsed + item['seconds'])} | {item['seconds']} s |"
        )
        elapsed += item["seconds"]
    elapsed = 0
    for index, item in enumerate(plan, 1):
        lines += [
            "",
            f"## {index}. {item['title']} ({clock(elapsed)}–{clock(elapsed + item['seconds'])})",
            "",
            item["speech"],
            "",
            "**Apoyo visual:** " + item["cue"],
        ]
        elapsed += item["seconds"]
    lines += [
        "",
        "## Ensayo y relevo entre expositores",
        "",
        "- Recorrido principal: solo diapositivas 1–7; las 8–9 están ocultas para preguntas.",
        "- Si exponen dos personas: una presenta 1–3 (2:10); la otra 4–7 (3:20). "
        "El relevo está incluido en el presupuesto, no añade una introducción nueva.",
        "- A 4:50, pasar al cierre. Si hay retraso, omitir detalles de memoria o límites; "
        "no omitir el estado de ejecución ni afirmar un ganador sin resultados.",
        "- No hacer descarga, entrenamiento ni demo en vivo. Abrir PPTX y PDF antes de empezar.",
        "- Ensayar con la terminología: ruido, U-Net, MSE, EMA, DDIM, FID y KID.",
        "",
        "## Antes de la entrega final",
        "",
        "- Añadir los nombres reales del equipo en PowerPoint; no se inventaron integrantes.",
        "- Completar ambos entrenamientos y el flujo sample/evaluate/report del README.",
        "- Reconstruir con `--require-results`; este modo rechaza ausencias, mezcla de "
        "protocolos, presupuestos parciales y muestras alteradas.",
        "- Revisar memoria y tiempos reales, curvas, N de test, pasos EMA elegidos y "
        "limitaciones. El anexo aclara qué sí demuestran las pruebas CPU.",
        "- No confundir fine-tuning con sobreajuste ni std de KID con un intervalo de confianza.",
        "- Los gráficos ilustrativos de difusión no son imágenes generadas por el modelo.",
    ]
    (output / "guion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path, output: Path, *, require_results=False):
    config = load_config(root / "configs/rtx3060_8gb.toml")
    evidence = load_evidence(root, config)
    if require_results and not evidence:
        raise FileNotFoundError(
            "Faltan resultados reales: ejecutar sample/evaluate/report primero."
        )
    plan = talk_plan(evidence, config)
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333333), Inches(7.5)
    presentation.core_properties.title = "PC3 · Difusión para generar rostros anime"
    presentation.core_properties.author = "Equipo PC3"
    presentation.core_properties.subject = "Desde cero frente a fine-tuning; comparación controlada"
    cover, content = _layouts(presentation)
    renderers = [
        lambda s: _cover(s, not evidence),
        lambda s: _diffusion(s, config),
        lambda s: _experiment(s, config),
        lambda s: _training(s, config),
        lambda s: _evaluation(s, config),
        lambda s: _results(s, evidence),
        lambda s: _closing(s, evidence),
    ]
    for number, (item, render) in enumerate(zip(plan, renderers, strict=True), 1):
        render(new_slide(presentation, cover if number in {1, 7} else content, item, number))
    _appendices(presentation, content, config)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{NAME}.pptx"
    presentation.save(destination)
    _script(plan, output, not evidence)
    words = sum(len(item["speech"].split()) for item in plan)
    manifest = {
        "status": "executed_results" if evidence else "pending_results",
        "main_slides": 7,
        "hidden_appendices": 2,
        "planned_seconds": 330,
        "spoken_words": words,
        "estimated_words_per_minute": round(words / 5.5, 1),
        "timing_measured": False,
        "pc3_source_sha256": source_fingerprint(),
        "evidence": "reports/comparison/comparison.json" if evidence else None,
        "pptx_sha256": file_sha256(destination),
    }
    write_json(output / "estado.json", manifest)
    return destination


def export_pdf(pptx: Path, soffice: str | None = None):
    """Renderizar el PPTX real, no una recreación del diseño en otro motor de dibujo."""
    binary = soffice or shutil.which("soffice") or shutil.which("libreoffice")
    if not binary:
        raise FileNotFoundError("Para --pdf se necesita LibreOffice o --soffice /ruta/a/soffice.")
    import pymupdf

    with tempfile.TemporaryDirectory(prefix="pc3-render-", dir=pptx.parent) as directory:
        temporary = Path(directory).resolve()
        # Para QA se renderizan también los anexos; la copia no altera el PPTX entregable.
        deck = Presentation(pptx)
        for slide in deck.slides:
            slide._element.set("show", "1")
        copy = temporary / pptx.name
        deck.save(copy)
        subprocess.run(
            [
                str(Path(binary).resolve()),
                "--headless",
                f"-env:UserInstallation={(temporary / 'profile').as_uri()}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temporary),
                str(copy),
            ],
            check=True,
            timeout=120,
        )
        rendered = temporary / f"{NAME}.pdf"
        pages_dir = pptx.parent / "vistas"
        pages_dir.mkdir(exist_ok=True)
        with pymupdf.open(rendered) as document:
            if len(document) != 9:
                raise ValueError("El PDF renderizado debe contener las nueve diapositivas para QA.")
            thumbnails = []
            for number, page in enumerate(document, 1):
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                path = pages_dir / f"diapositiva_{number:02}.png"
                pixmap.save(path)
                with Image.open(path) as image:
                    image.thumbnail((640, 360))
                    thumbnails.append(image.copy())
            pdf = pymupdf.open()
            pdf.insert_pdf(document, from_page=0, to_page=6)
            pdf.save(pptx.with_suffix(".pdf"))
            pdf.close()
        montage = Image.new("RGB", (640 * 3, 360 * 3), "white")
        for index, image in enumerate(thumbnails):
            montage.paste(image, ((index % 3) * 640, (index // 3) * 360))
        montage.save(pptx.parent / "vista_general.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "presentacion")
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--soffice")
    args = parser.parse_args()
    try:
        output = build(ROOT, args.output, require_results=args.require_results)
        if args.pdf:
            export_pdf(output, args.soffice)
        print(f"Presentación: {output}\nGuion: {output.parent / 'guion.md'}")
        print("5:30 planificados; ensayar con cronómetro. No se ejecutaron modelos.")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Error: {error}\n")
