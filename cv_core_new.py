"""
cv_core.py — Pipeline de Estéreo Fotométrico · Setup de Geometria Fixa
Otimizado para ambiente controlado (sala escura, luz pontual, câmera estática).

Mudanças em relação à versão anterior:
  1. Removidas calcular_vetor_luz() e detectar_highlight_subpixel() — esfera cromada
     não é mais usada; vetores chegam pré-calculados da geometria conhecida.
  2. criar_mascara_roi(): converte o polígono da folha de referência em máscara
     binária aplicada ANTES do least-squares — exclui fundo preto, mãos, bordas.
  3. subtrair_luz_ambiente() agora recebe a máscara ROI e estima o nível ambiente
     somente dentro da região válida (mais preciso com fundo quase negro).
  4. poisson_depth() substituído por solver DCT analítico: O(N log N) em vez de
     O(N · iterações). Boundary conditions de Neumann corretas, sem artefatos
     periódicos do Frankot-Chellappa.
  5. Gradientes zerados em pixels inválidos antes da integração — evita que o
     fundo mascarado "vaze" para a profundidade da região de interesse.
  6. processar_mapas() corrige bug silencioso: largura_mm + altura_mm separados
     em vez de tamanho_quadrado_mm (que era passado errado para a homografia).
  7. calcular_vetor_luz_geometria(): utilitário explícito para quem quiser validar
     ou gerar vetores fora do frontend.
"""

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn


# ==============================================================
# MÓDULO 1 — VETOR DE LUZ POR GEOMETRIA (sem esfera)
# ==============================================================

def calcular_vetor_luz_geometria(angulo_graus: float,
                                  h_mm: float,
                                  d_mm: float) -> list[float]:
    """
    Calcula o vetor unitário de iluminação a partir da geometria
    física do setup — sem necessidade de esfera cromada.

    Convenção de ângulo (0° = Norte = topo da imagem, horário):
        0°  → luz no topo    → L = [0, +d, h]
        90° → luz na direita → L = [+d, 0, h]
       180° → luz embaixo   → L = [0, -d, h]
       270° → luz na esquerda→ L = [-d, 0, h]

    Sistema de coordenadas (mesmo do estéreo fotométrico):
        +X = direita na imagem
        +Y = cima  na imagem  (invertido em relação a pixels)
        +Z = câmera (saindo da superfície)

    Args:
        angulo_graus : direção horizontal da luz (0° = Norte, horário)
        h_mm         : altura da fonte acima da superfície em mm
        d_mm         : distância horizontal da fonte ao centro em mm

    Returns:
        [Lx, Ly, Lz] normalizado
    """
    rad = angulo_graus * np.pi / 180.0
    lx  =  d_mm * np.sin(rad)
    ly  =  d_mm * np.cos(rad)   # N → +Y (positivo = topo da imagem)
    lz  =  h_mm
    mag = np.sqrt(lx**2 + ly**2 + lz**2)
    return [lx / mag, ly / mag, lz / mag]


# ==============================================================
# MÓDULO 2 — MÁSCARA DE REGIÃO DE INTERESSE (ROI)
# ==============================================================

def criar_mascara_roi(plano_coords: list[dict],
                      altura: int,
                      largura: int,
                      erosao_px: int = 8) -> np.ndarray:
    """
    Gera uma máscara binária booleana delimitando o polígono da
    folha/tapete de referência.

    Por que isso importa neste setup:
      O fundo da foto é quase negro (sala escura + papelão). Sem a
      máscara ROI, esses pixels sobrevivem ao thresh_sombra e
      introduzem normais zeradas que distorcem o depth map inteiro.
      A máscara elimina o problema na origem, antes de qualquer cálculo.

    Args:
        plano_coords : lista de 4 dicts {'x', 'y'} em ordem TL→TR→BR→BL
        altura, largura : dimensões da imagem
        erosao_px    : pixels de erosão interna para excluir bordas da folha
                       (bordas têm dobras/sombras que violam o modelo lambertiano)

    Returns:
        mascara bool (altura × largura), True = dentro do ROI
    """
    pts = np.array([[p['x'], p['y']] for p in plano_coords], dtype=np.int32)
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)

    if erosao_px > 0:
        kernel  = np.ones((erosao_px, erosao_px), dtype=np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)

    return mascara.astype(bool)


# ==============================================================
# MÓDULO 3 — CARREGAMENTO E PRÉ-PROCESSAMENTO
# ==============================================================

def load_and_flatten(filepath: str) -> np.ndarray:
    """Carrega imagem em escala de cinza como float64 [0, 1]."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {filepath}")
    return img.astype(np.float64) / 255.0


def subtrair_luz_ambiente(imgs: list[np.ndarray],
                           mascara_roi: np.ndarray | None = None,
                           percentil: int = 2) -> list[np.ndarray]:
    """
    Estima e subtrai a luz ambiente de cada imagem.

    Com mascara_roi fornecida, estima o nível ambiente SOMENTE dentro
    da região válida — evita que o fundo quase negro (que domina a
    imagem) puxe a estimativa para zero e subestime o ambiente real
    dentro da folha.

    Args:
        imgs        : lista de arrays float64 [0, 1]
        mascara_roi : máscara booleana da região de interesse (opcional)
        percentil   : percentil para estimativa de ambiente

    Returns:
        lista de imagens corrigidas, em [0, 1]
    """
    resultado = []
    for img in imgs:
        regiao = img[mascara_roi] if (mascara_roi is not None and mascara_roi.any()) else img.ravel()
        nivel_ambiente = np.percentile(regiao, percentil)
        resultado.append(np.clip(img - nivel_ambiente, 0.0, 1.0))
    return resultado


def criar_mascara_valida(imgs: list[np.ndarray],
                          mascara_roi: np.ndarray | None = None,
                          thresh_sombra: float = 0.03,
                          thresh_sat:    float = 0.97) -> np.ndarray:
    """
    Máscara booleana que exclui pixels em sombra, saturados ou fora do ROI.

    Threshold padrão ajustado para luz rasante (~14°):
      thresh_sombra = 0.03 (era 0.05): luz de baixa elevação produz
        pixels escuros mesmo em superfícies válidas; threshold mais
        conservador preserva mais dados.
      thresh_sat = 0.97 (era 0.98): papel branco diretamente iluminado
        satura facilmente; exclui mais cedo para proteger a linearidade.

    Args:
        imgs        : imagens float64 pós-subtração de ambiente
        mascara_roi : exclui pixels fora do polígono da folha
        thresh_sombra, thresh_sat : limiares de exclusão

    Returns:
        mascara bool (altura × largura)
    """
    mascara = np.ones(imgs[0].shape, dtype=bool)

    # ROI tem prioridade — exclui tudo fora da folha
    if mascara_roi is not None:
        mascara &= mascara_roi

    for img in imgs:
        mascara &= (img > thresh_sombra)
        mascara &= (img < thresh_sat)

    return mascara


# ==============================================================
# MÓDULO 4 — CORREÇÃO DE PERSPECTIVA VIA HOMOGRAFIA
# ==============================================================

def calcular_rotacao_por_homografia(pontos_imagem: list[dict],
                                     largura_mm: float = 210.0,
                                     altura_mm:  float = 297.0) -> np.ndarray:
    """
    Estima a matriz de rotação da câmera usando os 4 cantos de um
    retângulo plano de dimensões conhecidas (aceita A4, A3, qualquer rect).

    Pontos em ordem horária: TL → TR → BR → BL.
    Retorna identidade se o cálculo falhar.
    """
    w, h = float(largura_mm), float(altura_mm)
    pts_mundo = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    pts_img   = np.array([[p['x'], p['y']] for p in pontos_imagem], dtype=np.float32)

    H, _ = cv2.findHomography(pts_img, pts_mundo)
    if H is None:
        return np.eye(3)

    h1 = H[:, 0];  h2 = H[:, 1]
    lam = 1.0 / (np.linalg.norm(h1) + 1e-12)
    r1  = h1 * lam;  r2 = h2 * lam;  r3 = np.cross(r1, r2)

    R_raw = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def aplicar_rotacao_normais(N_flat: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotaciona o array de normais (3 × N_pixels) por R e renormaliza."""
    N_rot = R @ N_flat
    normas = np.linalg.norm(N_rot, axis=0)
    return np.divide(N_rot, normas, out=np.zeros_like(N_rot), where=normas > 1e-9)


# ==============================================================
# MÓDULO 5 — INTEGRAÇÃO DE GRADIENTES → DEPTH MAP
# ==============================================================

def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integra (p, q) = (∂z/∂x, ∂z/∂y) via FFT (Frankot-Chellappa 1988).
    O(N log N). Rápido para superfícies suaves e contínuas.
    Limitação: condições de contorno periódicas → ringing em bordas abruptas.
    """
    rows, cols = p.shape
    u = np.fft.fftfreq(cols) * 2 * np.pi
    v = np.fft.fftfreq(rows) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    den = U**2 + V**2;  den[0, 0] = 1.0
    Z_fft = -1j * (U * np.fft.fft2(p) + V * np.fft.fft2(q)) / den
    Z_fft[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z_fft))


def poisson_depth(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integra (p, q) resolvendo a equação de Poisson via DCT analítico.

    SUBSTITUIÇÃO do Gauss-Seidel iterativo (300 iterações O(N²)) por um
    solver direto O(N log N) baseado em Transformada Discreta de Cossenos:

      ∇²Z = div(p, q)
      → DCT(∇²Z) = λᵢⱼ · DCT(Z)   (λ = autovalores da Laplaciana)
      → Z = iDCT(DCT(div) / λ)

    Vantagens sobre Gauss-Seidel:
      • Mesma velocidade do Frankot-Chellappa (O(N log N))
      • Boundary conditions de Neumann corretas (∂Z/∂n = 0 nas bordas)
      • Sem artefatos periódicos do FFT
      • Lida bem com descontinuidades e bordas de objetos

    Args:
        p : gradiente ∂z/∂x  (altura × largura)
        q : gradiente ∂z/∂y  (altura × largura)

    Returns:
        Z : mapa de profundidade (mesma forma)
    """
    rows, cols = p.shape

    # 1. Divergência por diferenças finitas
    div = np.zeros((rows, cols))
    div[:-1, :] += q[:-1, :] - q[1:, :]    # dq/dy
    div[:, :-1] += p[:, :-1] - p[:, 1:]    # dp/dx

    # 2. Autovalores da Laplaciana 2D com Neumann boundaries
    ii = np.arange(rows).reshape(-1, 1)
    jj = np.arange(cols).reshape(1, -1)
    lam = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + \
          (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0, 0] = 1.0                         # Evita divisão por zero (componente DC)

    # 3. Solve analítico: Z = iDCT(DCT(div) / λ)
    Z_dct      = dctn(div, norm='ortho') / lam
    Z_dct[0,0] = 0.0                         # Ancora a altura base em zero
    return idctn(Z_dct, norm='ortho')


# ==============================================================
# MÓDULO 6 — PIPELINE PRINCIPAL
# ==============================================================

def processar_mapas(
    caminhos_imagens:  list[str],
    vetores_luz:       list[list[float]],
    diretorio_saida:   str,
    plano_coords:      list[dict] | None = None,
    largura_mm:        float = 210.0,       # A4 padrão (era tamanho_quadrado_mm)
    altura_mm:         float = 297.0,       # A4 padrão
    metodo_depth:      str   = 'poisson',   # padrão agora é poisson (DCT = mesmo custo)
    suavizar_normais:  bool  = False,
    sigma_suavizacao:  float = 1.0,
) -> tuple[str, str, str]:
    """
    Pipeline completo de Estéreo Fotométrico para setup de geometria fixa.

    Diferenças em relação à versão anterior:
      • Aceita largura_mm + altura_mm separados (suporte a retângulos reais)
      • Aplica máscara ROI desde o pré-processamento, não só na homografia
      • Ambiente estimado dentro do ROI (mais preciso com fundo escuro)
      • Poisson via DCT — mesmo custo do Frankot, sem artefatos periódicos
      • Gradientes zerados fora da máscara válida antes da integração

    Args:
        caminhos_imagens : 4 caminhos para as imagens em escala de cinza
        vetores_luz      : 4 vetores [Lx, Ly, Lz] (pré-calculados da geometria)
        diretorio_saida  : pasta de saída
        plano_coords     : 4 × {'x', 'y'} cantos TL→TR→BR→BL
        largura_mm       : largura real da referência em mm
        altura_mm        : altura  real da referência em mm
        metodo_depth     : 'poisson' (DCT, recomendado) ou 'frankot'
        suavizar_normais : gaussian filter antes do depth map
        sigma_suavizacao : σ do filtro gaussiano

    Returns:
        ('resultado_normal.png', 'resultado_depth.png', 'resultado_albedo.png')
    """

    # ── 1. CARREGAMENTO ────────────────────────────────────────
    imgs            = [load_and_flatten(c) for c in caminhos_imagens]
    altura, largura = imgs[0].shape
    n_pixels        = altura * largura

    # ── 2. MÁSCARA ROI ─────────────────────────────────────────
    # Criada aqui para ser reutilizada em todas as etapas seguintes.
    if plano_coords and len(plano_coords) >= 4:
        mascara_roi = criar_mascara_roi(plano_coords, altura, largura)
    else:
        mascara_roi = None   # Sem ROI → processa imagem inteira

    # ── 3. PRÉ-PROCESSAMENTO ───────────────────────────────────
    imgs    = subtrair_luz_ambiente(imgs, mascara_roi)
    mascara = criar_mascara_valida(imgs, mascara_roi)
    idx_val = np.where(mascara.flatten())[0]

    if len(idx_val) == 0:
        raise ValueError(
            "Nenhum pixel válido após mascaramento. "
            "Verifique thresh_sombra/sat ou os pontos do plano de referência."
        )

    # ── 4. VETORES DE LUZ ──────────────────────────────────────
    L = np.array(vetores_luz, dtype=np.float64)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)

    # ── 5. LEAST-SQUARES FOTOMÉTRICO ───────────────────────────
    # I = L · (ρ·N)  →  resolve para ρ·N por pixel
    # Processa somente pixels válidos para economia de memória e tempo.
    I_stack = np.stack([img.flatten() for img in imgs])   # (4, H×W)
    I_val   = I_stack[:, idx_val]                         # (4, n_val)

    G_val, _, _, _ = np.linalg.lstsq(L, I_val, rcond=None)   # (3, n_val)

    albedo_val = np.linalg.norm(G_val, axis=0)
    N_val      = np.divide(G_val, albedo_val,
                           out=np.zeros_like(G_val),
                           where=albedo_val > 1e-9)

    N_flat      = np.zeros((3, n_pixels))
    albedo_flat = np.zeros(n_pixels)
    N_flat[:, idx_val]   = N_val
    albedo_flat[idx_val] = albedo_val

    # ── 6. CORREÇÃO DE PERSPECTIVA ─────────────────────────────
    if plano_coords and len(plano_coords) >= 4:
        # BUG CORRIGIDO: versão anterior passava tamanho_quadrado_mm
        # onde a função espera largura_mm + altura_mm separados.
        R      = calcular_rotacao_por_homografia(plano_coords, largura_mm, altura_mm)
        N_flat = aplicar_rotacao_normais(N_flat, R)

    # ── 7. SUAVIZAÇÃO OPCIONAL ─────────────────────────────────
    if suavizar_normais:
        for i in range(3):
            canal      = N_flat[i].reshape(altura, largura)
            N_flat[i]  = gaussian_filter(canal, sigma=sigma_suavizacao).flatten()
        normas = np.linalg.norm(N_flat, axis=0)
        N_flat = np.divide(N_flat, normas, out=np.zeros_like(N_flat), where=normas > 1e-9)

    # ── 8. NORMAL MAP ──────────────────────────────────────────
    nx = N_flat[0].reshape(altura, largura)
    ny = N_flat[1].reshape(altura, largura)
    nz = N_flat[2].reshape(altura, largura)

    N_rgb = np.stack([
        np.clip((nx + 1.0) * 127.5, 0, 255),
        np.clip((ny + 1.0) * 127.5, 0, 255),
        np.clip((nz + 1.0) * 127.5, 0, 255),
    ], axis=-1).astype(np.uint8)

    # ── 9. DEPTH MAP ───────────────────────────────────────────
    nz_safe = np.where(np.abs(nz) < 0.01, np.sign(nz + 1e-12) * 0.01, nz)
    p_grad  = np.clip(-nx / nz_safe, -10.0, 10.0)
    q_grad  = np.clip(-ny / nz_safe, -10.0, 10.0)

    # Zera gradientes fora da máscara — evita que o fundo "vaze"
    # para o interior da região de interesse durante a integração.
    if mascara_roi is not None:
        p_grad[~mascara] = 0.0
        q_grad[~mascara] = 0.0

    depth_raw = poisson_depth(p_grad, q_grad) if metodo_depth == 'poisson' \
                else frankot_chellappa(p_grad, q_grad)

    d_min     = np.percentile(depth_raw[mascara] if mascara.any() else depth_raw, 2)
    d_max     = np.percentile(depth_raw[mascara] if mascara.any() else depth_raw, 98)
    depth_vis = np.clip(
        (depth_raw - d_min) / (d_max - d_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ── 10. ALBEDO MAP ─────────────────────────────────────────
    albedo_2d = albedo_flat.reshape(altura, largura)
    reg       = albedo_2d[mascara] if mascara.any() else albedo_2d.ravel()
    a_min, a_max = np.percentile(reg, 1), np.percentile(reg, 99)
    albedo_vis = np.clip(
        (albedo_2d - a_min) / (a_max - a_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ── 11. SALVAR ─────────────────────────────────────────────
    p_normal = os.path.join(diretorio_saida, 'resultado_normal.png')
    p_depth  = os.path.join(diretorio_saida, 'resultado_depth.png')
    p_albedo = os.path.join(diretorio_saida, 'resultado_albedo.png')

    cv2.imwrite(p_normal, cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p_depth,  depth_vis)
    cv2.imwrite(p_albedo, albedo_vis)

    return 'resultado_normal.png', 'resultado_depth.png', 'resultado_albedo.png'