"""
cv_core.py — Pipeline de Estéreo Fotométrico (versão revisada)

Melhorias principais em relação à versão anterior:
  1. Correção de perspectiva via HOMOGRAFIA (geometricamente exata)
     em vez de estimar a rotação pelas normais fotométricas (impreciso).
  2. Máscara de SOMBRA e SATURAÇÃO: exclui pixels corrompidos do least-squares.
  3. Remoção de LUZ AMBIENTE: subtrai o viés de iluminação difusa.
  4. Detecção SUBPIXEL do highlight da esfera cromada.
  5. Integrador de POISSON como alternativa ao Frankot-Chellappa.
  6. ALBEDO MAP gerado como produto extra sem custo adicional.
  7. Clipping de gradientes extremos antes da integração.
  8. Eliminação de código duplicado (N_image era gerado duas vezes).
"""

import cv2
import numpy as np
import os
import math
from scipy.ndimage import gaussian_filter


# ==============================================================
# MÓDULO 1 — CALIBRAÇÃO DA LUZ (Esfera Cromada)
# ==============================================================

def calcular_vetor_luz(cx, cy, raio, px, py):
    """
    Calcula o vetor 3D da direção da luz a partir do reflexo
    especular numa esfera cromada (mirror sphere / chrome ball).

    Princípio: o ponto brilhante na esfera é onde a lei de reflexão
    satisfaz L = 2(N·V)N − V, com V = câmera (eixo Z).

    Args:
        cx, cy : centro da esfera em pixels (imagem)
        raio   : raio da esfera em pixels
        px, py : pixel do highlight clicado pelo usuário

    Returns:
        list[float]: vetor de luz [Lx, Ly, Lz], magnitude ≈ 1
    """
    dx = (px - cx) / raio
    dy = (cy - py) / raio          # Y invertido: tela ↓, matemática ↑

    dist_sq = dx**2 + dy**2
    if dist_sq > 1.0:
        norm = math.sqrt(dist_sq)
        dx, dy = dx / norm, dy / norm
        dist_sq = 1.0

    dz = math.sqrt(max(0.0, 1.0 - dist_sq))

    N = np.array([dx, dy, dz])
    V = np.array([0.0, 0.0, 1.0])

    L = 2.0 * np.dot(N, V) * N - V
    L = L / np.linalg.norm(L)
    return L.tolist()


def detectar_highlight_subpixel(img_gray, cx_aprox, cy_aprox, raio_busca=30):
    """
    Refina a posição do highlight com precisão subpixel usando o
    centroide ponderado pela intensidade na região de busca.

    Isso elimina o erro de ~1–3 px do clique manual e melhora a
    precisão do vetor de luz resultante.

    Args:
        img_gray : imagem grayscale uint8 ou float
        cx_aprox, cy_aprox : posição aproximada do clique
        raio_busca : meia-janela de busca em pixels

    Returns:
        (cx_sub, cy_sub): posição subpixel do highlight
    """
    h, w = img_gray.shape
    x0, x1 = max(0, cx_aprox - raio_busca), min(w, cx_aprox + raio_busca)
    y0, y1 = max(0, cy_aprox - raio_busca), min(h, cy_aprox + raio_busca)

    roi = img_gray[y0:y1, x0:x1].astype(np.float64)
    threshold = np.percentile(roi, 95)        # Considera somente o topo 5%
    roi_w = np.where(roi >= threshold, roi, 0.0)

    total = roi_w.sum()
    if total < 1e-9:
        return cx_aprox, cy_aprox             # Fallback se ROI vazia

    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (xs * roi_w).sum() / total, (ys * roi_w).sum() / total


# ==============================================================
# MÓDULO 2 — CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==============================================================

def load_and_flatten(filepath):
    """
    Carrega imagem em escala de cinza e converte para float64 [0, 1].
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {filepath}")
    return img.astype(np.float64) / 255.0


def criar_mascara_valida(imgs, thresh_sombra=0.05, thresh_sat=0.98):
    """
    Gera uma máscara booleana excluindo pixels em sombra ou saturados
    em QUALQUER imagem do conjunto.

    Por que isso importa:
      • Pixels em SOMBRA violam o modelo lambertiano (I = ρ·L·N)
        e introduzem zeros espúrios no sistema linear.
      • Pixels SATURADOS (I ≥ 1) estão fora da região linear do sensor
        e inflam artificialmente a estimativa de albedo/normais.

    Args:
        imgs          : lista de arrays float64 [0, 1]
        thresh_sombra : intensidade mínima para considerar válido
        thresh_sat    : intensidade máxima para considerar válido

    Returns:
        mascara bool 2D, True = pixel válido para o least-squares
    """
    mascara = np.ones(imgs[0].shape, dtype=bool)
    for img in imgs:
        mascara &= (img > thresh_sombra)
        mascara &= (img < thresh_sat)
    return mascara


def subtrair_luz_ambiente(imgs, percentil=2):
    """
    Estima e subtrai a luz ambiente de cada imagem.

    Assume que os pixels mais escuros da imagem representam superfícies
    quase sem contribuição direcional, logo correspondem ao nível ambiente.
    Remover esse offset melhora a linearidade do modelo I = ρ·(L·N).

    Args:
        imgs      : lista de arrays float64 [0, 1]
        percentil : percentil usado como estimativa de ambiente

    Returns:
        lista de imagens corrigidas, ainda em [0, 1]
    """
    resultado = []
    for img in imgs:
        nivel_ambiente = np.percentile(img, percentil)
        resultado.append(np.clip(img - nivel_ambiente, 0.0, 1.0))
    return resultado


# ==============================================================
# MÓDULO 3 — CORREÇÃO DE PERSPECTIVA VIA HOMOGRAFIA
# ==============================================================

def calcular_rotacao_por_homografia(pontos_imagem, largura_mm=100.0, altura_mm=100.0):
    """
    Estima a orientação da câmera em relação ao plano da cena usando
    os 4 cantos de um retângulo físico de dimensões conhecidas.

    Aceita quadrados E retângulos (ex.: folha A4 = 210 × 297 mm).
    Os pontos devem ser fornecidos em ordem horária: TL → TR → BR → BL.

    O método extrai os vetores de rotação diretamente das colunas da
    homografia (decomposição de Faugeras, 1988), depois projeta para
    SO(3) via SVD para garantir ortogonalidade exata.

    Args:
        pontos_imagem : lista de 4 dicts {'x':..., 'y':...} (TL,TR,BR,BL)
        largura_mm    : dimensão horizontal do retângulo em milímetros
        altura_mm     : dimensão vertical  do retângulo em milímetros

    Returns:
        R: matriz de rotação 3×3 (identidade se cálculo falhar)
    """
    w, h = float(largura_mm), float(altura_mm)
    pts_mundo = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts_img   = np.array([[p['x'], p['y']] for p in pontos_imagem], dtype=np.float32)

    H, status = cv2.findHomography(pts_img, pts_mundo)
    if H is None:
        return np.eye(3)

    # Extrai e normaliza os vetores coluna da homografia
    h1 = H[:, 0]
    h2 = H[:, 1]
    lam = 1.0 / (np.linalg.norm(h1) + 1e-12)
    r1  = h1 * lam
    r2  = h2 * lam
    r3  = np.cross(r1, r2)

    # Projeta para SO(3) via SVD (garante rotação válida, sem shear)
    R_raw = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt

    # Garante det(R) = +1 (não uma reflexão)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def aplicar_rotacao_normais(N_flat, R):
    """
    Rotaciona o array de normais (3 × N_pixels) pela matriz R.
    Renormaliza após a rotação para manter norma unitária.
    """
    N_rot = R @ N_flat
    normas = np.linalg.norm(N_rot, axis=0)
    return np.divide(N_rot, normas, out=np.zeros_like(N_rot), where=normas > 1e-9)


# ==============================================================
# MÓDULO 4 — INTEGRAÇÃO DE GRADIENTES → DEPTH MAP
# ==============================================================

def frankot_chellappa(p, q):
    """
    Integra os campos de gradiente (p = ∂z/∂x, q = ∂z/∂y) para
    gerar o mapa de profundidade via transformada de Fourier.

    Complexidade: O(N log N) — muito rápido mesmo para imagens grandes.
    Ponto fraco: assume condições de contorno periódicas; gera
    artefatos de "ringing" em bordas de objetos com altura abrupta.
    Use este método para superfícies suaves e contínuas.
    """
    rows, cols = p.shape
    u = np.fft.fftfreq(cols) * 2 * np.pi
    v = np.fft.fftfreq(rows) * 2 * np.pi
    U, V = np.meshgrid(u, v)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    den = U**2 + V**2
    den[0, 0] = 1.0                     # Evita divisão por zero no DC

    Z_fft = -1j * (U * P + V * Q) / den
    Z_fft[0, 0] = 0.0                   # Âncora a altura base em zero

    return np.real(np.fft.ifft2(Z_fft))


def poisson_depth(p, q, iteracoes=300):
    """
    Integra os gradientes resolvendo a equação de Poisson
    ∇²Z = div(p, q) por diferenças finitas (Gauss-Seidel).

    Vantagem sobre Frankot-Chellappa: suporta descontinuidades e
    bordas de objetos, sem artefatos de periodicidade.
    Desvantagem: O(N·iteracoes), mais lento — use para objetos
    com bordas abruptas (moedas, peças maquinadas, etc.).

    Condição de contorno de Neumann: ∂Z/∂n = 0 nas bordas.
    """
    rows, cols = p.shape

    # Divergência do campo (∇·F) por diferenças finitas centralizadas
    div = np.zeros((rows, cols))
    div[:-1, :] += q[:-1, :] - q[1:, :]   # dq/dy
    div[:, :-1] += p[:, :-1] - p[:, 1:]   # dp/dx

    Z = np.zeros((rows, cols))
    for _ in range(iteracoes):
        Z_new = Z.copy()
        Z_new[1:-1, 1:-1] = (
            Z[:-2, 1:-1] + Z[2:, 1:-1] +
            Z[1:-1, :-2] + Z[1:-1, 2:] -
            div[1:-1, 1:-1]
        ) / 4.0
        # Condições de Neumann nas bordas
        Z_new[0, :]  = Z_new[1, :]
        Z_new[-1, :] = Z_new[-2, :]
        Z_new[:, 0]  = Z_new[:, 1]
        Z_new[:, -1] = Z_new[:, -2]
        Z = Z_new

    return Z


# ==============================================================
# MÓDULO 5 — PIPELINE PRINCIPAL
# ==============================================================

def processar_mapas(
    caminhos_imagens,
    vetores_luz,
    diretorio_saida,
    plano_coords=None,
    metodo_depth='frankot',
    suavizar_normais=False,
    sigma_suavizacao=1.0,
    tamanho_quadrado_mm=100.0,
):
    """
    Pipeline completo de Estéreo Fotométrico.

    Args:
        caminhos_imagens   : lista de caminhos para as imagens (greyscale)
        vetores_luz        : lista de vetores de luz [Lx, Ly, Lz] por imagem
        diretorio_saida    : pasta onde os resultados serão salvos
        plano_coords       : lista de 4 dicts {'x', 'y'} com os cantos do
                             quadrado de referência (TL, TR, BR, BL)
        metodo_depth       : 'frankot' (rápido) ou 'poisson' (mais preciso
                             em bordas abruptas)
        suavizar_normais   : aplica gaussian filter antes de gerar o depth map
        sigma_suavizacao   : desvio padrão do filtro gaussiano
        tamanho_quadrado_mm: tamanho real do lado do quadrado de referência

    Returns:
        tuple: nomes dos arquivos gerados (normal, depth, albedo)
    """

    # ----------------------------------------------------------
    # 1. CARREGAMENTO
    # ----------------------------------------------------------
    imgs = [load_and_flatten(c) for c in caminhos_imagens]
    altura, largura = imgs[0].shape
    n_pixels = altura * largura

    # ----------------------------------------------------------
    # 2. PRÉ-PROCESSAMENTO
    # ----------------------------------------------------------
    imgs = subtrair_luz_ambiente(imgs)
    mascara = criar_mascara_valida(imgs)          # True = pixel válido
    idx_validos = np.where(mascara.flatten())[0]

    # ----------------------------------------------------------
    # 3. VETORES DE LUZ
    # ----------------------------------------------------------
    L = np.array(vetores_luz, dtype=np.float64)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)   # Garante unitário

    # ----------------------------------------------------------
    # 4. LEAST-SQUARES FOTOMÉTRICO (somente pixels válidos)
    #
    # Resolve: I = L · (ρ·N)  para cada pixel
    # Onde:
    #   I = vetor de intensidades observadas (n_luzes × 1)
    #   L = matriz de vetores de luz (n_luzes × 3)
    #   ρ·N = pseudo-normal escalada pelo albedo (3 × 1)
    # ----------------------------------------------------------
    I_stack = np.stack([img.flatten() for img in imgs])    # (n_luzes, n_pixels)
    I_val   = I_stack[:, idx_validos]                      # Somente válidos

    G_val, _, _, _ = np.linalg.lstsq(L, I_val, rcond=None)   # (3, n_val)

    albedo_val = np.linalg.norm(G_val, axis=0)
    N_val = np.divide(G_val, albedo_val,
                      out=np.zeros_like(G_val),
                      where=albedo_val > 1e-9)

    # Reconstrói arrays completos (pixels inválidos = zero)
    N_flat     = np.zeros((3, n_pixels))
    albedo_flat = np.zeros(n_pixels)
    N_flat[:, idx_validos]    = N_val
    albedo_flat[idx_validos]  = albedo_val

    # ----------------------------------------------------------
    # 5. CORREÇÃO DE PERSPECTIVA VIA HOMOGRAFIA
    # ----------------------------------------------------------
    if plano_coords and len(plano_coords) >= 4:
        R = calcular_rotacao_por_homografia(plano_coords, tamanho_quadrado_mm)
        N_flat = aplicar_rotacao_normais(N_flat, R)

    # ----------------------------------------------------------
    # 6. SUAVIZAÇÃO OPCIONAL DAS NORMAIS
    # ----------------------------------------------------------
    if suavizar_normais:
        for i in range(3):
            canal = N_flat[i].reshape(altura, largura)
            N_flat[i] = gaussian_filter(canal, sigma=sigma_suavizacao).flatten()
        # Re-normaliza após filtro (gaussiano quebra a norma unitária)
        normas = np.linalg.norm(N_flat, axis=0)
        N_flat = np.divide(N_flat, normas,
                           out=np.zeros_like(N_flat), where=normas > 1e-9)

    # ----------------------------------------------------------
    # 7. SEPARAÇÃO DOS CANAIS DE NORMAL
    # ----------------------------------------------------------
    nx = N_flat[0].reshape(altura, largura)
    ny = N_flat[1].reshape(altura, largura)
    nz = N_flat[2].reshape(altura, largura)

    # ----------------------------------------------------------
    # 8. NORMAL MAP (padrão tangent-space: R=+X, G=+Y, B=+Z)
    # ----------------------------------------------------------
    N_rgb = np.stack([
        np.clip((nx + 1.0) * 127.5, 0, 255),
        np.clip((ny + 1.0) * 127.5, 0, 255),
        np.clip((nz + 1.0) * 127.5, 0, 255),
    ], axis=-1).astype(np.uint8)

    # ----------------------------------------------------------
    # 9. DEPTH MAP
    #
    # Gradientes: p = -nx/nz, q = -ny/nz
    # Problema: nz ≈ 0 em superfícies quase verticais → divisão instável.
    # Solução: clamp suave em 0.01 (era 0.05, muito agressivo).
    # Também: clipamos os gradientes em ±10 para evitar spikes de
    # integração que distorcem o depth map inteiro.
    # ----------------------------------------------------------
    nz_safe  = np.where(np.abs(nz) < 0.01, np.sign(nz + 1e-12) * 0.01, nz)
    p_grad   = np.clip(-nx / nz_safe, -10.0, 10.0)
    q_grad   = np.clip(-ny / nz_safe, -10.0, 10.0)

    if metodo_depth == 'poisson':
        depth_raw = poisson_depth(p_grad, q_grad)
    else:
        depth_raw = frankot_chellappa(p_grad, q_grad)

    # Normalização robusta por percentil (ignora outliers)
    d_min = np.percentile(depth_raw, 2)
    d_max = np.percentile(depth_raw, 98)
    depth_vis = np.clip(
        (depth_raw - d_min) / (d_max - d_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ----------------------------------------------------------
    # 10. ALBEDO MAP (produto extra sem custo adicional)
    # ----------------------------------------------------------
    albedo_2d = albedo_flat.reshape(altura, largura)
    if mascara.any():
        a_min = np.percentile(albedo_2d[mascara], 1)
        a_max = np.percentile(albedo_2d[mascara], 99)
    else:
        a_min, a_max = albedo_2d.min(), albedo_2d.max()

    albedo_vis = np.clip(
        (albedo_2d - a_min) / (a_max - a_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ----------------------------------------------------------
    # 11. SALVAR RESULTADOS
    # ----------------------------------------------------------
    caminho_normal = os.path.join(diretorio_saida, 'resultado_normal.png')
    caminho_depth  = os.path.join(diretorio_saida, 'resultado_depth.png')
    caminho_albedo = os.path.join(diretorio_saida, 'resultado_albedo.png')

    cv2.imwrite(caminho_normal, cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(caminho_depth,  depth_vis)
    cv2.imwrite(caminho_albedo, albedo_vis)

    return 'resultado_normal.png', 'resultado_depth.png', 'resultado_albedo.png'