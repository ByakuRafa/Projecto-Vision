"""
main.py — Servidor FastAPI para o pipeline de Estéreo Fotométrico.

Execução:
    pip install fastapi uvicorn python-multipart
    uvicorn main:app --reload --port 8000

Estrutura de pastas esperada:
    main.py
    cv_core.py
    static/
        index.html       ← copie o index.html para cá
    uploads/             ← criada automaticamente
    resultados/          ← criada automaticamente
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
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("photostereo")

DIR_STATIC     = Path("static")
DIR_UPLOADS    = Path("uploads")
DIR_RESULTADOS = Path("static")   # Resultados servidos via /static

for d in (DIR_STATIC, DIR_UPLOADS, DIR_RESULTADOS):
    d.mkdir(exist_ok=True)

# Pool de threads para não bloquear o event loop durante o processamento CV.
# max_workers=2: evita sobrecarga de RAM em hardware limitado.
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="PhotoStereo Audit", version="3.0")

app.mount("/static",  StaticFiles(directory=DIR_STATIC),  name="static")
app.mount("/uploads", StaticFiles(directory=DIR_UPLOADS), name="uploads")


# ──────────────────────────────────────────────────────────────
# MODELOS PYDANTIC
# ──────────────────────────────────────────────────────────────

class PontoPlano(BaseModel):
    """Coordenada de um canto do plano de referência em pixels."""
    x: float
    y: float


class CalibracaoData(BaseModel):
    """
    Payload enviado pelo frontend após a calibração.

    vetores_luz: lista de 4 vetores [Lx, Ly, Lz], pré-calculados pelo
                 frontend a partir da geometria conhecida (h_mm, d_mm, ângulo).
                 Cada vetor já chega normalizado — cv_core renormaliza por
                 segurança de qualquer forma.
    """
    arquivos:        List[str]
    plano:           List[PontoPlano]
    vetores_luz:     List[List[float]]          # BUG #1 CORRIGIDO: era List[Any] sem import
    largura_mm:      float  = Field(210.0, gt=0)
    altura_mm:       float  = Field(297.0, gt=0)
    metodo_depth:    str    = "poisson"
    suavizar_normais: bool  = False

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

    @field_validator("plano")
    @classmethod
    def validar_plano(cls, v):
        if len(v) != 4:
            raise ValueError(f"Esperado 4 pontos do plano, recebido {len(v)}.")
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
    """Serve a interface HTML principal."""
    caminho = DIR_STATIC / "index.html"
    if not caminho.exists():
        raise HTTPException(404, detail="index.html não encontrado em static/")
    return FileResponse(caminho)


@app.post("/upload_imagens")
async def upload_imagens(imagens: List[UploadFile] = File(...)):
    """
    Recebe as 4 imagens, salva com nomes únicos e devolve os nomes.

    BUG #3 CORRIGIDO: a versão anterior salvava com img.filename original,
    causando colisão entre sessões com fotos de mesmo nome (ex: IMG_0001.jpg).
    Agora cada arquivo recebe um UUID como prefixo, garantindo unicidade.
    """
    if len(imagens) != 4:
        raise HTTPException(400, detail=f"Envie exatamente 4 imagens. Recebido: {len(imagens)}.")

    arquivos_salvos = []
    for img in imagens:
        # Sanitiza a extensão (mantém somente a extensão original)
        extensao   = Path(img.filename).suffix.lower() or ".jpg"
        nome_unico = f"{uuid.uuid4().hex}{extensao}"
        caminho    = DIR_UPLOADS / nome_unico

        with caminho.open("wb") as buf:
            shutil.copyfileobj(img.file, buf)

        arquivos_salvos.append(nome_unico)
        log.info(f"Upload salvo: {nome_unico} (original: {img.filename})")

    return {"arquivos": arquivos_salvos}


@app.post("/processar")
async def processar_dados(dados: CalibracaoData):
    """
    Executa o pipeline de Estéreo Fotométrico e devolve URLs dos resultados.

    BUG #4 CORRIGIDO: processar_mapas() é CPU-bound (numpy/opencv).
    Rodá-lo diretamente em async def bloqueia o event loop inteiro.
    run_in_executor() delega para um ThreadPoolExecutor e mantém o
    servidor responsivo durante o processamento.
    """
    # Verifica se todos os arquivos existem antes de iniciar
    caminhos = []
    for nome in dados.arquivos:
        p = DIR_UPLOADS / nome
        if not p.exists():
            raise HTTPException(404, detail=f"Arquivo não encontrado: {nome}. "
                                            "Faça o upload novamente.")
        caminhos.append(str(p))

    # Converte PontoPlano → dict para cv_core (que espera List[dict])
    plano_dicts = [{"x": pt.x, "y": pt.y} for pt in dados.plano]

    # BUG #2 CORRIGIDO:
    #   Versão anterior: imagens=, plano=   ← nomes errados → TypeError
    #   Versão correta:  caminhos_imagens=, plano_coords=
    def _executar_cv():
        return cv_core.processar_mapas(
            caminhos_imagens = caminhos,
            vetores_luz      = dados.vetores_luz,
            diretorio_saida  = str(DIR_RESULTADOS),
            plano_coords     = plano_dicts,
            largura_mm       = dados.largura_mm,
            altura_mm        = dados.altura_mm,
            metodo_depth     = dados.metodo_depth,
            suavizar_normais = dados.suavizar_normais,
        )

    log.info(f"Iniciando processamento CV | método: {dados.metodo_depth} | "
             f"ref: {dados.largura_mm}×{dados.altura_mm}mm")

    try:
        loop = asyncio.get_event_loop()
        normal_img, depth_img, albedo_img = await loop.run_in_executor(
            _executor, _executar_cv
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        log.exception("Erro no pipeline CV")
        raise HTTPException(500, detail=f"Erro interno no processamento: {e}")

    log.info("Processamento concluído com sucesso.")

    return {
        "status":     "sucesso",
        "normal_url": f"/static/{normal_img}",
        "depth_url":  f"/static/{depth_img}",
        "albedo_url": f"/static/{albedo_img}",
    }


@app.get("/health")
async def health():
    """Endpoint de saúde — útil para monitorar se o servidor está de pé."""
    return {"status": "ok"}