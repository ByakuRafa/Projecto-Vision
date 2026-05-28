"""
cv_core_v2.py — Estéreo Fotométrico com Segmentação Explícita de Sombras
"""
import cv2
import numpy as np
from scipy.fft import dctn, idctn
import os

# ==============================================================
# 1. SEGMENTAÇÃO E MÁSCARAS LÓGICAS
# ==============================================================
def segmentar_cena(imgs: list[np.ndarray], 
                   thresh_fundo_std: float = 0.03, 
                   razao_sombra: float = 0.70) -> dict:
    """
    Segmenta a cena usando topologia e queda de luz relativa.
    razao_sombra: Se a luz mínima do pixel for menor que X% da média dele, é sombra.
    """
    stack = np.stack(imgs, axis=0)
    I_std = np.std(stack, axis=0)
    I_min = np.min(stack, axis=0)
    I_mean = np.mean(stack, axis=0)

    H, W = imgs[0].shape

    # 1. Encontrar superfícies perfeitamente planas
    # Apenas olhamos para a variação (std), ignorando o quão claro ou escuro é.
    planos_totais = (I_std < thresh_fundo_std)

    # 2. Separar o chão verdadeiro dos topos (Flood Fill)
    im_planos = (planos_totais * 255).astype(np.uint8)
    mask_fill = np.zeros((H + 2, W + 2), np.uint8)

    cv2.floodFill(im_planos, mask_fill, (0, 0), 128)
    cv2.floodFill(im_planos, mask_fill, (W - 1, 0), 128)
    cv2.floodFill(im_planos, mask_fill, (0, H - 1), 128)
    cv2.floodFill(im_planos, mask_fill, (W - 1, H - 1), 128)

    mask_fundo_limpo = (im_planos == 128)
    mask_topos = (im_planos == 255)

    # 3. IDENTIFICAÇÃO DE SOMBRA RELATIVA
    # Em vez de um valor fixo, verificamos se a luz caiu drasticamente.
    # Ex: se razao_sombra=0.70, a luz mínima tem que ser menor que 70% da luz média.
    queda_de_luz = I_min < (I_mean * razao_sombra)
    
    # É sombra se teve queda de luz e NÃO é uma superfície plana limpa
    mask_sombra = (~planos_totais) & queda_de_luz

    # 4. Separar sombra no chão vs sombra no objeto
    kernel = np.ones((5,5), np.uint8)
    fundo_expandido = cv2.dilate(mask_fundo_limpo.astype(np.uint8), kernel, iterations=3).astype(bool)
    
    mask_sombra_chao = mask_sombra & fundo_expandido
    mask_sombra_objeto = mask_sombra & ~fundo_expandido

    # 5. Consolidar os Objetos
    # Objeto = o que sobrou (paredes) + as tampas detectadas no Flood Fill
    mask_objeto_paredes = ~(mask_fundo_limpo | mask_sombra | mask_topos)
    mask_objeto = mask_objeto_paredes | mask_topos

    mask_chao_total = mask_fundo_limpo | mask_sombra_chao

    return {
        'fundo_limpo': mask_fundo_limpo,
        'sombra_chao': mask_sombra_chao,
        'sombra_objeto': mask_sombra_objeto,
        'objeto': mask_objeto,
        'chao_total': mask_chao_total
    }
# ==============================================================
# 2. RESOLUÇÃO FOTOMÉTRICA (Apenas onde importa)
# ==============================================================

def calcular_normais_segmentadas(imgs: list[np.ndarray], 
                                 vetores_luz: list[list[float]], 
                                 mascaras: dict) -> tuple[np.ndarray, np.ndarray]:
    
    H, W = imgs[0].shape
    L = np.array(vetores_luz, dtype=np.float64)
    L_inv = np.linalg.pinv(L)
    
    stack = np.stack([img.flatten() for img in imgs], axis=0) # (4, N)
    
    # Vamos calcular apenas onde é OBJETO ou SOMBRA NO OBJETO.
    # O chão total será forçado para [0,0,1]
    mask_calc = mascaras['objeto'] | mascaras['sombra_objeto']
    idx_calc = np.where(mask_calc.flatten())[0]
    
    I_calc = stack[:, idx_calc]
    
    # Matemática Photometric Stereo (I = L * G)
    G_calc = L_inv @ I_calc
    albedo_calc = np.linalg.norm(G_calc, axis=0)
    N_calc = np.divide(G_calc, albedo_calc, out=np.zeros_like(G_calc), where=albedo_calc > 1e-9)
    
    # Remontar mapa completo
    N_full = np.zeros((3, H * W), dtype=np.float64)
    N_full[0, idx_calc] = N_calc[0]
    N_full[1, idx_calc] = N_calc[1]
    N_full[2, idx_calc] = N_calc[2]
    
    # Forçar o Chão Total a ser perfeitamente plano (Normal Z = 1)
    idx_chao = np.where(mascaras['chao_total'].flatten())[0]
    N_full[0, idx_chao] = 0.0
    N_full[1, idx_chao] = 0.0
    N_full[2, idx_chao] = 1.0
    
    nx = N_full[0].reshape(H, W)
    ny = N_full[1].reshape(H, W)
    nz = N_full[2].reshape(H, W)
    
    return nx, ny, nz


# ==============================================================
# 3. INTEGRAÇÃO DE PROFUNDIDADE (Blindada pela segmentação)
# ==============================================================

def integrar_depth(nx: np.ndarray, ny: np.ndarray, nz: np.ndarray, mascaras: dict) -> np.ndarray:
    
    # Evitar divisão por zero e artefatos de borda extrema
    nz_safe = np.where(np.abs(nz) < 0.15, 1.0, nz)
    
    mask_chao = mascaras['chao_total']
    
    # Gradientes p e q: Zero no chão, calculados apenas nos objetos
    p = np.zeros_like(nx)
    q = np.zeros_like(ny)
    
    p[~mask_chao] = np.clip(-nx[~mask_chao] / nz_safe[~mask_chao], -3.0, 3.0)
    q[~mask_chao] = np.clip(-ny[~mask_chao] / nz_safe[~mask_chao], -3.0, 3.0)
    
    # Solver de Poisson via Transformada Discreta de Cosseno (DCT)
    rows, cols = p.shape
    div = np.zeros((rows, cols))
    div[:-1, :] += q[1:, :] - q[:-1, :] 
    div[:, :-1] += p[:, 1:] - p[:, :-1] 
    
    ii = np.arange(rows).reshape(-1, 1)
    jj = np.arange(cols).reshape(1, -1)
    lam = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0,0] = 1.0 # Previne divisão por zero
    
    Z_dct = dctn(div, norm='ortho') / lam
    Z_dct[0,0] = 0.0
    depth_raw = idctn(Z_dct, norm='ortho')
    
    # TRAVAR O CHÃO
    if np.any(mask_chao):
        base_z = np.median(depth_raw[mask_chao])
        depth_raw -= base_z
        
    depth_raw[mask_chao] = 0.0
    depth_raw = np.maximum(depth_raw, 0.0)
    
    # Normalizar focando apenas na altura dos objetos
    depth_objeto = depth_raw[~mask_chao]
    if len(depth_objeto) > 0:
        d_max = np.percentile(depth_objeto, 98)
        depth_norm = np.clip((depth_raw) / (d_max + 1e-9) * 255.0, 0, 255)
    else:
        depth_norm = np.zeros_like(depth_raw)
        
    depth_norm[mask_chao] = 0.0
    
    return depth_norm


# ==============================================================
# 4. PIPELINE PRINCIPAL E EXPORTAÇÃO
# ==============================================================

def processar_perspectiva(caminhos_imagens: list[str], vetores_luz: list[list[float]], dir_saida: str):
    print("1. Carregando imagens...")
    imgs = [cv2.imread(c, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0 for c in caminhos_imagens]
    
    print("2. Segmentando cena (Fundo, Sombras, Objetos)...")
    mascaras = segmentar_cena(imgs)
    
    # Salvar mapas de debug para você visualizar como ele separou a cena!
    cv2.imwrite(os.path.join(dir_saida, 'debug_mask_chao_total.png'), mascaras['chao_total'].astype(np.uint8)*255)
    cv2.imwrite(os.path.join(dir_saida, 'debug_mask_sombras.png'), (mascaras['sombra_chao'] | mascaras['sombra_objeto']).astype(np.uint8)*255)
    cv2.imwrite(os.path.join(dir_saida, 'debug_mask_objetos.png'), mascaras['objeto'].astype(np.uint8)*255)

    print("3. Resolvendo Normais Fotométricas...")
    nx, ny, nz = calcular_normais_segmentadas(imgs, vetores_luz, mascaras)
    
    # Salvar Normal Map
    N_rgb = np.stack([
        np.clip(( nx + 1.0) * 127.5, 0, 255),
        np.clip((-ny + 1.0) * 127.5, 0, 255),
        np.clip(( nz + 1.0) * 127.5, 0, 255)
    ], axis=-1).astype(np.uint8)
    cv2.imwrite(os.path.join(dir_saida, 'resultado_normal.png'), cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))

    print("4. Integrando Mapa de Profundidade...")
    depth_norm = integrar_depth(nx, ny, nz, mascaras)
    
    depth_suave = cv2.GaussianBlur(depth_norm.astype(np.uint8), (5,5), sigmaX=1.0)
    depth_colormap = cv2.applyColorMap(depth_suave, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(dir_saida, 'resultado_depth.png'), depth_colormap)
    
    print("Concluído! Verifique a pasta de saída para os arquivos debug e resultados finais.")