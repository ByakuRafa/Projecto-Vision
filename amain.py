"""
amain.py — Servidor FastAPI · Modo Blender

Versão 5.0 — Novos campos em CalibracaoData:
  • ortho_scale_mm  : Orthographic Scale do Blender em mm (câmara ortográfica real).
  • detectar_sombra : ativa/desativa detecção de sombra cast (padrão: True).
  • thresh_sombra_cast : sensibilidade da detecção (0.20=agressivo, 0.50=suave).
  • raio_contexto_px   : raio do filtro de contexto vizinho para detecção.

Execução:
    pip install fastapi uvicorn python-multipart
    uvicorn amain:app --reload --port 8000
"""

import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv_core
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("photostereo")

DIR_STATIC  = Path("static")
DIR_UPLOADS = Path("uploads")
for d in (DIR_STATIC, DIR_UPLOADS):
    d.mkdir(exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=2)
app       = FastAPI(title="PhotoStereo Blender", version="5.0")

app.mount("/static",  StaticFiles(directory=DIR_STATIC),  name="static")
app.mount("/uploads", StaticFiles(directory=DIR_UPLOADS), name="uploads")


# ──────────────────────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────────────────────

class CalibracaoData(BaseModel):
    arquivos:     List[str]
    vetores_luz:  List[List[float]]

    # ── Câmara Blender ───────────────────────────────────────
    focal_mm:        float         = Field(50.0,   gt=0,
        description="Focal length (mm) — só usado no modo perspectiva (fallback)")
    sensor_w_mm:     float         = Field(36.0,   gt=0,
        description="Sensor width (mm) — só usado no modo perspectiva (fallback)")
    camera_h_mm:     float         = Field(1000.0, gt=0,
        description="Altura da câmara acima da bancada (mm)")
    ortho_scale_mm:  Optional[float] = Field(814.286, gt=0,
        description="Orthographic Scale do Blender convertido em mm "
                    "(ex: Blender mostra 1.28 → passe 1280). "
                    "Se fornecido, usa cálculo ortográfico correto ignorando focal/sensor/altura.")

    # ── Bancada ──────────────────────────────────────────────
    wb_larg_mm: float = Field(1000.0, gt=0, description="Largura da bancada (mm)")
    wb_prof_mm: float = Field(600.0,  gt=0, description="Profundidade da bancada (mm)")

    # ── Opções de processamento ──────────────────────────────
    metodo_depth:     str  = "poisson"
    suavizar_normais: bool = False
    razao_sombra:     float = Field(0.60, ge=0.1, le=1.0,
        description="Drop-darkest: descarta luz se min < X*mediana (sombra própria)")
    thresh_residuo:   float = Field(0.20, ge=0.05, le=1.0,
        description="Rejeição por resíduo: máximo tolerado (0.10=agressivo, 0.40=suave)")
    thresh_variacao:  float = Field(0.04, ge=0.001, le=0.5,
        description="Sensibilidade da máscara de variação")

    # ── Detecção de sombra cast ──────────────────────────────
    detectar_sombra:    bool  = Field(True,
        description="Ativa detecção de sombras cast (projetadas por outros objetos). "
                    "Recomendado manter True para cenas com múltiplos objetos.")
    thresh_sombra_cast: float = Field(0.35, ge=0.10, le=0.90,
        description="Sensibilidade da detecção de sombra cast por inconsistência de normais. "
                    "Menor = remove mais pixels de sombra. 0.20=agressivo, 0.50=suave.")
    raio_contexto_px:   int   = Field(5, ge=2, le=20,
        description="Raio do filtro de contexto para detecção de sombra cast (pixels). "
                    "Aumente para objetos grandes; reduza para detalhes pequenos.")

    @field_validator("arquivos")
    @classmethod
    def validar_n_arquivos(cls, v):
        if len(v) != 4:
            raise ValueError(f"Esperado 4 arquivos, recebido {len(v)}.")
        return v

    @field_validator("vetores_luz")
    @classmethod
    def validar_vetores(cls, v):
        if len(v) != 4:
            raise ValueError(f"Esperado 4 vetores, recebido {len(v)}.")
        for i, vec in enumerate(v):
            if len(vec) != 3:
                raise ValueError(f"Vetor {i} deve ter 3 componentes [Lx, Ly, Lz].")
        return v

    @field_validator("metodo_depth")
    @classmethod
    def validar_metodo(cls, v):
        if v not in ("frankot", "poisson"):
            raise ValueError("metodo_depth deve ser 'frankot' ou 'poisson'.")
        return v


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def read_index():
    caminho = DIR_STATIC / "index.html"
    if not caminho.exists():
        raise HTTPException(404, detail="index.html não encontrado em static/")
    return FileResponse(caminho)


@app.post("/upload_imagens")
async def upload_imagens(imagens: List[UploadFile] = File(...)):
    if len(imagens) != 4:
        raise HTTPException(400, detail=f"Envie exatamente 4 imagens. Recebido: {len(imagens)}.")
    arquivos_salvos = []
    for img in imagens:
        extensao   = Path(img.filename).suffix.lower() or ".jpg"
        nome_unico = f"{uuid.uuid4().hex}{extensao}"
        caminho    = DIR_UPLOADS / nome_unico
        with caminho.open("wb") as buf:
            shutil.copyfileobj(img.file, buf)
        arquivos_salvos.append(nome_unico)
        log.info(f"Upload: {nome_unico}  (original: {img.filename})")
    return {"arquivos": arquivos_salvos}


@app.post("/processar")
async def processar_dados(dados: CalibracaoData):
    caminhos = []
    for nome in dados.arquivos:
        p = DIR_UPLOADS / nome
        if not p.exists():
            raise HTTPException(404, detail=f"Arquivo não encontrado: {nome}.")
        caminhos.append(str(p))

    def _executar_cv():
        return cv_core.processar_mapas(
            caminhos_imagens    = caminhos,
            vetores_luz         = dados.vetores_luz,
            diretorio_saida     = str(DIR_STATIC),
            # ── Câmara ──────────────────────────────────────
            camera_ortogonal    = True,
            wb_larg_mm          = dados.wb_larg_mm,
            wb_prof_mm          = dados.wb_prof_mm,
            camera_h_mm         = dados.camera_h_mm,
            focal_mm            = dados.focal_mm,
            sensor_w_mm         = dados.sensor_w_mm,
            ortho_scale_mm      = dados.ortho_scale_mm,
            # ── Qualidade ───────────────────────────────────
            metodo_depth        = dados.metodo_depth,
            suavizar_normais    = dados.suavizar_normais,
            thresh_variacao     = dados.thresh_variacao,
            thresh_residuo      = dados.thresh_residuo,
            razao_sombra        = dados.razao_sombra,
            # ── Sombra cast ─────────────────────────────────
            detectar_sombra     = dados.detectar_sombra,
            thresh_sombra_cast  = dados.thresh_sombra_cast,
            raio_contexto_px    = dados.raio_contexto_px,
        )

    log.info(
        f"Processando | método={dados.metodo_depth} | "
        f"ortho_scale={dados.ortho_scale_mm}mm | "
        f"bancada {dados.wb_larg_mm}×{dados.wb_prof_mm}mm | "
        f"sombra_cast={dados.detectar_sombra} thresh={dados.thresh_sombra_cast}"
    )

    try:
        loop = asyncio.get_event_loop()
        normal_img, depth_img, albedo_img = await loop.run_in_executor(_executor, _executar_cv)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        log.exception("Erro no pipeline CV")
        raise HTTPException(500, detail=f"Erro interno: {e}")

    log.info("Processamento concluído.")
    return {
        "status":     "sucesso",
        "normal_url": f"/static/{normal_img}",
        "depth_url":  f"/static/{depth_img}",
        "albedo_url": f"/static/{albedo_img}",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}