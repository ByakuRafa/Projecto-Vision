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
    cx: float
    cy: float
    raio: float
    luzes: List[dict] # Lista de {px, py} para cada imagem

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
    # 1. Calcular os 4 vetores de luz usando a função do cv_core
    vetores_luz = []
    for luz in dados.luzes:
        v = cv_core.calcular_vetor_luz(dados.cx, dados.cy, dados.raio, luz['px'], luz['py'])
        vetores_luz.append(v)
    
    # 2. Preparar caminhos absolutos das imagens
    caminhos_imagens = [f"uploads/{arquivo}" for arquivo in dados.arquivos]
    
    # 3. Chamar o processamento pesado de CV
    normal_img, depth_img, albedo_img = cv_core.processar_mapas(caminhos_imagens, vetores_luz, "static", dados.plano)
    
    # 4. Retornar os caminhos para o frontend visualizar
    return {
        "status": "sucesso",
        "normal_url": f"/static/{normal_img}",
        "depth_url": f"/static/{depth_img}",
        "albedo_url": f"/static/{albedo_img}"
    }
