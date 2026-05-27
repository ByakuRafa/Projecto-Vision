"""
cv_core.py — Pipeline de Estéreo Fotométrico · Modo Blender (câmara ortogonal fixa)

Diferenças em relação à versão anterior:
  1. calcular_roi_blender(): gera a máscara ROI automaticamente a partir dos
     parâmetros da câmara e da bancada — sem seleção manual de cantos.
  2. processar_mapas() aceita camera_ortogonal=True:
     quando verdadeiro, o passo de homografia/rotação é completamente omitido
     (câmara apontando direto para baixo = normais já no referencial correto).
  3. plano_coords é opcional; se ausente e camera_ortogonal=True, a ROI é
     calculada via calcular_roi_blender().
"""

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn
from fastapi import FastAPI

app = FastAPI() 

# ==============================================================
# MÓDULO 1 — VETOR DE LUZ POR GEOMETRIA
# ==============================================================

def calcular_vetor_luz_geometria(angulo_graus: float,
                                  h_mm: float,
                                  d_mm: float) -> list[float]:
    """
    Vetor unitário de iluminação calculado da geometria física do setup.

    Convenção (0°=Norte=topo da imagem, sentido horário):
        0°  → L = [ 0, +d, h]
        90° → L = [+d,  0, h]
       180° → L = [ 0, -d, h]
       270° → L = [-d,  0, h]
    """
    rad = angulo_graus * np.pi / 180.0
    lx  =  d_mm * np.sin(rad)
    ly  =  d_mm * np.cos(rad)
    lz  =  h_mm
    mag = np.sqrt(lx**2 + ly**2 + lz**2)
    return [lx / mag, ly / mag, lz / mag]


# ==============================================================
# MÓDULO 2 — ROI
# ==============================================================

def calcular_roi_blender(img_largura:     int,
                          img_altura:     int,
                          wb_larg_mm:     float,
                          wb_prof_mm:     float,
                          camera_h_mm:    float,
                          focal_mm:       float,
                          sensor_w_mm:    float,
                          erosao_px:      int = 8) -> np.ndarray:
    """
    Gera máscara ROI booleana automaticamente a partir dos parâmetros
    da câmara Blender e das dimensões físicas da bancada.

    Geometria (câmara perspectiva apontada para baixo, centrada):
        phys_w = camera_h_mm * sensor_w_mm / focal_mm   ← largura visível
        phys_h = phys_w * img_altura / img_largura       ← altura visível (aspect)
        roi_hw = (wb_larg_mm / phys_w) * img_largura / 2 ← meia-largura em px
        roi_hh = (wb_prof_mm / phys_h) * img_altura  / 2 ← meia-altura  em px

    Se a bancada for maior que o campo de visão (ratio > 1), a ROI
    ocupa a imagem toda naquela dimensão.

    Args:
        img_largura, img_altura : resolução da imagem em pixels
        wb_larg_mm, wb_prof_mm  : dimensões físicas da bancada em mm
        camera_h_mm             : altura da câmara acima da bancada em mm
        focal_mm                : comprimento focal da lente (mm)
        sensor_w_mm             : largura do sensor (mm) — padrão Blender: 36
        erosao_px               : erosão interna para excluir bordas

    Returns:
        máscara bool (img_altura × img_largura)
    """
    phys_w  = camera_h_mm * sensor_w_mm / focal_mm
    phys_h  = phys_w * img_altura / img_largura

    roi_hw = min((wb_larg_mm / phys_w) * img_largura / 2.0,  img_largura / 2.0)
    roi_hh = min((wb_prof_mm / phys_h) * img_altura  / 2.0,  img_altura  / 2.0)

    cx, cy = img_largura / 2.0, img_altura / 2.0

    pts = np.array([
        [cx - roi_hw, cy - roi_hh],   # TL
        [cx + roi_hw, cy - roi_hh],   # TR
        [cx + roi_hw, cy + roi_hh],   # BR
        [cx - roi_hw, cy + roi_hh],   # BL
    ], dtype=np.int32)

    mascara = np.zeros((img_altura, img_largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)

    if erosao_px > 0:
        kernel  = np.ones((erosao_px, erosao_px), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)

    return mascara.astype(bool)


def criar_mascara_roi(plano_coords: list[dict],
                      altura: int,
                      largura: int,
                      erosao_px: int = 8) -> np.ndarray:
    """ROI a partir de 4 cantos marcados manualmente (mantido para compatibilidade)."""
    pts = np.array([[p['x'], p['y']] for p in plano_coords], dtype=np.int32)
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara, [pts], 255)
    if erosao_px > 0:
        kernel  = np.ones((erosao_px, erosao_px), np.uint8)
        mascara = cv2.erode(mascara, kernel, iterations=1)
    return mascara.astype(bool)


# ==============================================================
# MÓDULO 3 — PRÉ-PROCESSAMENTO
# ==============================================================

def load_and_flatten(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {filepath}")
    return img.astype(np.float64) / 255.0


def subtrair_luz_ambiente(imgs: list[np.ndarray],
                           mascara_roi: np.ndarray | None = None) -> list[np.ndarray]:
    """
    Remove a componente de luz ambiente pixel a pixel.

    Método: min(I1, I2, I3, I4) em cada pixel — é a melhor estimativa possível
    da luz ambiente, pois em pelo menos uma das 4 direções o pixel recebe
    somente a componente difusa mínima (ou está em sombra).

    Muito mais eficaz do que um percentil global para renders Blender onde
    a iluminação ambiente do World pode variar pela cena.
    """
    stack    = np.stack(imgs, axis=0)          # (N, H, W)
    ambiente = np.min(stack, axis=0)           # mínimo pixel-a-pixel entre as N imagens
    return [np.clip(img - ambiente, 0.0, 1.0) for img in imgs]


def criar_mascara_valida(imgs: list[np.ndarray],
                          mascara_roi:      np.ndarray | None = None,
                          thresh_variacao:  float = 0.04,
                          thresh_sat:       float = 0.95) -> np.ndarray:
    """
    Pixel válido = apresenta variação real entre as 4 imagens.

    std(I1..IN) > thresh_variacao garante que o pixel foi efetivamente
    iluminado de formas diferentes e não está:
      - sempre em sombra (std ≈ 0, baixo)
      - sempre saturado  (std ≈ 0, alto)
      - num shadow cast de outro objeto (variação mínima)

    Isto substitui o threshold absoluto anterior que excluía pixels
    escuros legítimos e deixava passar pixels com sombra constante.
    """
    stack    = np.stack(imgs, axis=0)          # (N, H, W)
    variacao = np.std(stack, axis=0)           # desvio padrão por pixel

    mascara = variacao > thresh_variacao
    if mascara_roi is not None:
        mascara &= mascara_roi
    # Excluir saturados em qualquer das imagens
    for img in imgs:
        mascara &= (img < thresh_sat)
    return mascara


# ==============================================================
# MÓDULO 4 — CORREÇÃO DE PERSPECTIVA (somente câmara não-ortogonal)
# ==============================================================

def calcular_rotacao_por_homografia(pontos_imagem: list[dict],
                                     largura_mm: float = 210.0,
                                     altura_mm:  float = 297.0) -> np.ndarray:
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
    N_rot = R @ N_flat
    normas = np.linalg.norm(N_rot, axis=0)
    return np.divide(N_rot, normas, out=np.zeros_like(N_rot), where=normas > 1e-9)


# ==============================================================
# MÓDULO 5 — INTEGRAÇÃO DE GRADIENTES → DEPTH MAP
# ==============================================================

def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    rows, cols = p.shape
    u = np.fft.fftfreq(cols) * 2 * np.pi
    v = np.fft.fftfreq(rows) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    den = U**2 + V**2;  den[0, 0] = 1.0
    Z_fft = -1j * (U * np.fft.fft2(p) + V * np.fft.fft2(q)) / den
    Z_fft[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z_fft))


def poisson_depth(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solver DCT analítico O(N log N) com boundary conditions de Neumann."""
    rows, cols = p.shape
    div = np.zeros((rows, cols))
    div[:-1, :] += q[:-1, :] - q[1:, :]
    div[:, :-1] += p[:, :-1] - p[:, 1:]
    ii  = np.arange(rows).reshape(-1, 1)
    jj  = np.arange(cols).reshape(1, -1)
    lam = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + \
          (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0, 0] = 1.0
    Z_dct      = dctn(div, norm='ortho') / lam
    Z_dct[0,0] = 0.0
    return idctn(Z_dct, norm='ortho')


# ==============================================================
# MÓDULO 6 — PIPELINE PRINCIPAL
# ==============================================================

def processar_mapas(
    caminhos_imagens:  list[str],
    vetores_luz:       list[list[float]],
    diretorio_saida:   str,
    # ── ROI manual (compatibilidade) ─────────────────────────
    plano_coords:      list[dict] | None = None,
    largura_mm:        float = 210.0,
    altura_mm:         float = 297.0,
    # ── ROI automático (modo Blender) ─────────────────────────
    camera_ortogonal:  bool  = False,   # True = câmara reta para baixo, sem homografia
    wb_larg_mm:        float | None = None,
    wb_prof_mm:        float | None = None,
    camera_h_mm:       float | None = None,
    focal_mm:          float = 50.0,
    sensor_w_mm:       float = 36.0,
    # ── Opções ────────────────────────────────────────────────
    metodo_depth:      str   = 'poisson',
    suavizar_normais:  bool  = False,
    sigma_suavizacao:  float = 1.0,
    # Controle de qualidade da máscara:
    # Aumente thresh_variacao (ex: 0.08) se aparecerem artefatos no fundo.
    # Diminua (ex: 0.02) se objetos de cor uniforme sumam da máscara.
    thresh_variacao:   float = 0.04,
    thresh_sat:        float = 0.95,
) -> tuple[str, str, str]:
    """
    Pipeline completo de Estéreo Fotométrico.

    Modo Blender (camera_ortogonal=True):
      • ROI calculada automaticamente via calcular_roi_blender()
      • Passo de homografia/rotação de normais é OMITIDO —
        câmara já está alinhada com a normal da superfície.
      • Requer: wb_larg_mm, wb_prof_mm, camera_h_mm.

    Modo manual (camera_ortogonal=False, plano_coords fornecido):
      • ROI = polígono marcado manualmente.
      • Rotação de normais calculada por homografia.
    """

    # ── 1. CARREGAMENTO ──────────────────────────────────────
    imgs            = [load_and_flatten(c) for c in caminhos_imagens]
    altura, largura = imgs[0].shape

    # ── 2. MÁSCARA ROI ──────────────────────────────────────
    if camera_ortogonal and all(v is not None for v in [wb_larg_mm, wb_prof_mm, camera_h_mm]):
        # Modo Blender: ROI calculada automaticamente
        mascara_roi = calcular_roi_blender(
            img_largura  = largura,
            img_altura   = altura,
            wb_larg_mm   = wb_larg_mm,
            wb_prof_mm   = wb_prof_mm,
            camera_h_mm  = camera_h_mm,
            focal_mm     = focal_mm,
            sensor_w_mm  = sensor_w_mm,
        )
    elif plano_coords and len(plano_coords) >= 4:
        # Modo manual
        mascara_roi = criar_mascara_roi(plano_coords, altura, largura)
    else:
        mascara_roi = None   # Sem ROI — processa imagem inteira

    # ── 3. PRÉ-PROCESSAMENTO ────────────────────────────────
    imgs    = subtrair_luz_ambiente(imgs, mascara_roi)
    mascara = criar_mascara_valida(imgs, mascara_roi,
                                   thresh_variacao=thresh_variacao,
                                   thresh_sat=thresh_sat)
    idx_val = np.where(mascara.flatten())[0]

    pct_val = 100.0 * len(idx_val) / (altura * largura)
    print(f"  [mask] {len(idx_val)} px válidos ({pct_val:.1f}% da imagem)")

    if len(idx_val) == 0:
        raise ValueError(
            "Nenhum pixel válido após mascaramento. "
            "Reduza thresh_variacao ou verifique os parâmetros da câmara/bancada."
        )

    # ── 4. VETORES DE LUZ ───────────────────────────────────
    L = np.array(vetores_luz, dtype=np.float64)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)

    # ── 5. LEAST-SQUARES FOTOMÉTRICO ────────────────────────
    n_pixels = altura * largura
    I_stack  = np.stack([img.flatten() for img in imgs])
    I_val    = I_stack[:, idx_val]
    G_val, _, _, _ = np.linalg.lstsq(L, I_val, rcond=None)

    albedo_val = np.linalg.norm(G_val, axis=0)
    N_val      = np.divide(G_val, albedo_val,
                           out=np.zeros_like(G_val),
                           where=albedo_val > 1e-9)

    N_flat      = np.zeros((3, n_pixels))
    albedo_flat = np.zeros(n_pixels)
    N_flat[:, idx_val]   = N_val
    albedo_flat[idx_val] = albedo_val

    # ── 6. CORREÇÃO DE PERSPECTIVA (só modo manual) ─────────
    # Modo Blender: câmara já está perpendicular à bancada →
    # as normais calculadas já estão no referencial correto.
    # Aplicar a homografia aqui introduziria erro, não corrigiria nada.
    if not camera_ortogonal and plano_coords and len(plano_coords) >= 4:
        R      = calcular_rotacao_por_homografia(plano_coords, largura_mm, altura_mm)
        N_flat = aplicar_rotacao_normais(N_flat, R)

    # ── 7. SUAVIZAÇÃO OPCIONAL ──────────────────────────────
    if suavizar_normais:
        for i in range(3):
            canal     = N_flat[i].reshape(altura, largura)
            N_flat[i] = gaussian_filter(canal, sigma=sigma_suavizacao).flatten()
        normas = np.linalg.norm(N_flat, axis=0)
        N_flat = np.divide(N_flat, normas, out=np.zeros_like(N_flat), where=normas > 1e-9)

    # ── 8. NORMAL MAP ───────────────────────────────────────
    nx = N_flat[0].reshape(altura, largura)
    ny = N_flat[1].reshape(altura, largura)
    nz = N_flat[2].reshape(altura, largura)
    N_rgb = np.stack([
        np.clip((nx + 1.0) * 127.5, 0, 255),
        np.clip((ny + 1.0) * 127.5, 0, 255),
        np.clip((nz + 1.0) * 127.5, 0, 255),
    ], axis=-1).astype(np.uint8)

    # ── 9. DEPTH MAP ────────────────────────────────────────
    nz_safe = np.where(np.abs(nz) < 0.01, np.sign(nz + 1e-12) * 0.01, nz)
    p_grad  = np.clip(-nx / nz_safe, -10.0, 10.0)
    q_grad  = np.clip(-ny / nz_safe, -10.0, 10.0)

    # Forçar gradientes a zero fora da máscara para não contaminar a integração
    p_grad[~mascara] = 0.0
    q_grad[~mascara] = 0.0

    depth_raw = poisson_depth(p_grad, q_grad) if metodo_depth == 'poisson' \
                else frankot_chellappa(p_grad, q_grad)

    reg_m = mascara if mascara.any() else np.ones_like(mascara)
    d_min = np.percentile(depth_raw[reg_m], 2)
    d_max = np.percentile(depth_raw[reg_m], 98)
    depth_norm = np.clip(
        (depth_raw - d_min) / (d_max - d_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # Suavização leve do depth final (σ=1) para reduzir ruído de integração
    # sem distorcer as normais — só aplicada dentro da máscara válida
    depth_smooth = cv2.GaussianBlur(depth_norm, (5, 5), sigmaX=1.0)
    depth_vis    = np.where(reg_m, depth_smooth, depth_norm)

    # ── 10. ALBEDO MAP ──────────────────────────────────────
    albedo_2d = albedo_flat.reshape(altura, largura)
    reg_a     = albedo_2d[reg_m]
    a_min, a_max = np.percentile(reg_a, 1), np.percentile(reg_a, 99)
    albedo_vis = np.clip(
        (albedo_2d - a_min) / (a_max - a_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ── 11. SALVAR ──────────────────────────────────────────
    p_normal = os.path.join(diretorio_saida, 'resultado_normal.png')
    p_depth  = os.path.join(diretorio_saida, 'resultado_depth.png')
    p_albedo = os.path.join(diretorio_saida, 'resultado_albedo.png')

    cv2.imwrite(p_normal, cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p_depth,  depth_vis)
    cv2.imwrite(p_albedo, albedo_vis)

    return 'resultado_normal.png', 'resultado_depth.png', 'resultado_albedo.png'

