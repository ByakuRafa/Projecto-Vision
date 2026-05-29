import os
import math
import sys

# Importa a função principal do nosso novo core
from cv_core_v2 import processar_perspectiva

def calcular_vetor_sol(azimute_graus: float, elevacao_graus: float = 45.0) -> list[float]:
    """
    Calcula a direção da luz.
    Azimute: 0=Norte, 90=Leste, 180=Sul, 270=Oeste
    """
    az = math.radians(azimute_graus)
    el = math.radians(elevacao_graus)
    
    # Fórmulas de conversão esférica para cartesiana
    lx = math.sin(az) * math.cos(el)
    ly = math.cos(az) * math.cos(el)
    lz = math.sin(el)
    
    mag = math.sqrt(lx**2 + ly**2 + lz**2)
    return [lx/mag, ly/mag, lz/mag]

def main():
    # 1. Configuração de Diretórios
    # Usa os.path.join para evitar problemas com barras (\ ou /)
    pasta_entrada = os.path.join("test_depth", "t9")
    pasta_saida = os.path.join("test_depth", "t9_resultados")
    
    # Cria a pasta de saída se ela não existir
    os.makedirs(pasta_saida, exist_ok=True)

    # 2. Mapeamento das Imagens
    # Assumindo formato .png (altere para .jpg se necessário)
    ext = ".png"
    caminhos_imagens = [
        os.path.join(pasta_entrada, f"1{ext}"), # Imagem 1: Luz do Norte
        os.path.join(pasta_entrada, f"2{ext}"), # Imagem 2: Luz do Leste
        os.path.join(pasta_entrada, f"3{ext}"), # Imagem 3: Luz do Sul
        os.path.join(pasta_entrada, f"4{ext}")  # Imagem 4: Luz do Oeste
    ]

    # Validação rápida de segurança
    for caminho in caminhos_imagens:
        if not os.path.exists(caminho):
            print(f"ERRO: A imagem {caminho} não foi encontrada!")
            sys.exit(1)

    # 3. Definição dos Vetores de Luz
    # Assumindo que as luzes no Blender estão a 45º de inclinação para baixo
    vetores_luz = [
        calcular_vetor_sol(0),   # Norte
        calcular_vetor_sol(90),  # Leste
        calcular_vetor_sol(180), # Sul
        calcular_vetor_sol(270)  # Oeste
    ]

    # 4. Execução do Pipeline
    print(f"Lendo imagens da pasta: {pasta_entrada}")
    print(f"Salvando resultados em: {pasta_saida}\n")
    print("-" * 40)
    
    processar_perspectiva(caminhos_imagens, vetores_luz, pasta_saida)

if __name__ == "__main__":
    main()