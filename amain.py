"""
amain.py — Servidor FastAPI · Modo Blender (câmara ortogonal fixa)

Diferenças em relação à versão anterior:
  • CalibracaoData não tem mais o campo `plano` (4 cantos manuais).
  • Novos campos: wb_larg_mm, wb_prof_mm, camera_h_mm, focal_mm, sensor_w_mm.
  • processar_mapas() é chamado com camera_ortogonal=True →
    ROI auto-calculada, homografia/rotação omitidas.

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
from typing import List

import cv_core
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("photostereo")

DIR_STATIC     = Path("static")
DIR_UPLOADS    = Path("uploads")

for d in (DIR_STATIC, DIR_UPLOADS):
    d.mkdir(exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=2)
app       = FastAPI(title="PhotoStereo Blender", version="4.0")

app.mount("/static",  StaticFiles(directory=DIR_STATIC),  name="static")
app.mount("/uploads", StaticFiles(directory=DIR_UPLOADS), name="uploads")


# ──────────────────────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────────────────────

class CalibracaoData(BaseModel):
    """
    Payload do frontend.
    Sem `plano`: a ROI é calculada automaticamente dos parâmetros da câmara.
    """
    arquivos:        List[str]
    vetores_luz:     List[List[float]]

    # ── Câmara Blender ───────────────────────────────────────
    focal_mm:        float = Field(50.0,  gt=0, description="Focal length (mm)")
    sensor_w_mm:     float = Field(36.0,  gt=0, description="Sensor width (mm)")
    camera_h_mm:     float = Field(1000.0,gt=0, description="Altura da câmara acima da bancada (mm)")

    # ── Bancada ──────────────────────────────────────────────
    wb_larg_mm:      float = Field(1000.0, gt=0, description="Largura da bancada (mm)")
    wb_prof_mm:      float = Field(600.0,  gt=0, description="Profundidade da bancada (mm)")

    # ── Opções de processamento ──────────────────────────────
    metodo_depth:    str  = "poisson"
    suavizar_normais: bool = False

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
            raise ValueError(f"Esperado 4 vetores de luz, recebido {len(v)}.")
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
    """Recebe as 4 imagens, salva com UUID e devolve os nomes."""
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
    """
    Executa o pipeline de Estéreo Fotométrico no modo Blender.
    ROI calculada automaticamente; sem homografia.
    """
    caminhos = []
    for nome in dados.arquivos:
        p = DIR_UPLOADS / nome
        if not p.exists():
            raise HTTPException(404, detail=f"Arquivo não encontrado: {nome}. Faça o upload novamente.")
        caminhos.append(str(p))

    def _executar_cv():
        return cv_core.processar_mapas(
            caminhos_imagens = caminhos,
            vetores_luz      = dados.vetores_luz,
            diretorio_saida  = str(DIR_STATIC),
            # ── Modo Blender ────────────────────────────────
            camera_ortogonal = True,
            wb_larg_mm       = dados.wb_larg_mm,
            wb_prof_mm       = dados.wb_prof_mm,
            camera_h_mm      = dados.camera_h_mm,
            focal_mm         = dados.focal_mm,
            sensor_w_mm      = dados.sensor_w_mm,
            # ── Opções ──────────────────────────────────────
            metodo_depth     = dados.metodo_depth,
            suavizar_normais = dados.suavizar_normais,
        )

    log.info(f"Processando | método={dados.metodo_depth} | "
             f"câmara h={dados.camera_h_mm}mm f={dados.focal_mm}mm | "
             f"bancada {dados.wb_larg_mm}×{dados.wb_prof_mm}mm")

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
