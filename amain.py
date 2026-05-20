from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
import shutil
import cv_core  # Importando o seu módulo de CV!

app = FastAPI()

# Criar pastas se não existirem
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

class CalibracaoData(BaseModel):
    arquivos: List[str]
    plano: List[dict] 
    # Dependendo de como a sua função calcVetorLuz retorna no JS (se é array [x,y,z] ou objeto {x,y,z}), 
    # deixamos como Any ou List[dict] para o FastAPI não chiar na conversão.
    vetores_luz: List[Any] 
    largura_mm: float
    altura_mm: float
    metodo_depth: str
    suavizar_normais: bool # Lista de {px, py} para cada imagem



@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/upload_imagens")
async def upload_imagens(imagens: List[UploadFile] = File(...)):
    arquivos_salvos = []
    for img in imagens:
        caminho = f"uploads/{img.filename}"
        with open(caminho, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        arquivos_salvos.append(img.filename)
    return {"arquivos": arquivos_salvos}

@app.post("/processar")
async def processar_dados(dados: CalibracaoData):
    
    # 1. Preparar caminhos absolutos das imagens
    caminhos_imagens = [f"uploads/{arquivo}" for arquivo in dados.arquivos]
    
    # 2. Chamar o processamento pesado de CV
    # Agora passamos os vetores de luz diretos do frontend e as novas configurações
    normal_img, depth_img, albedo_img = cv_core.processar_mapas(
        imagens=caminhos_imagens, 
        vetores_luz=dados.vetores_luz, 
        diretorio_saida="static", 
        plano=dados.plano,
        largura_mm=dados.largura_mm,
        altura_mm=dados.altura_mm,
        metodo_depth=dados.metodo_depth,
        suavizar_normais=dados.suavizar_normais
    )
    
    # 3. Retornar os caminhos para o frontend atualizar as tags <img>
    return {
        "status": "sucesso",
        "normal_url": f"/static/{normal_img}",
        "depth_url": f"/static/{depth_img}",
        "albedo_url": f"/static/{albedo_img}"
    }
