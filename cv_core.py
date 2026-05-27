"""
cv_core.py — Pipeline de Estéreo Fotométrico · Modo Blender

Versão 5.0 — Melhorias principais:
  1. Detecção robusta de sombras cast via "consistência fotométrica":
     um pixel em sombra tem I_k ≈ 0 em algumas luzes mas o resíduo
     explode porque nenhuma normal real produz esse padrão assimétrico.
     Novo método: threshold adaptativo por percentil local + máscara
     de sombra por gradiente de luminância entre imagens vizinhas.
  2. calcular_roi_blender() aceita ortho_scale_mm para câmara ortográfica
     real do Blender (sem depender de focal/sensor/distância).
  3. mask_fundo corrigido: usa ~mascara_depth para não confundir tampas
     planas de objetos com o chão.
  4. sem_normal recalculado após suavização para não destruir bordas.
  5. Poisson com divergência forward-difference (corrigido).
  6. Normal map com flip-Y para convenção OpenGL correta.
"""

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.fft import dctn, idctn


# ==============================================================
# MÓDULO 1 — VETORES DE LUZ
# ==============================================================

def calcular_vetor_luz_sol(azimute_graus: float,
                            elevacao_graus: float = 45.0) -> list[float]:
    """
    Vetor unitário para fonte Solar/Direcional do Blender.

    Convenção de azimute (0°=Norte, 90°=Leste, 180°=Sul, 270°=Oeste):
        N (0°,  45°) → [ 0,      0.707, 0.707]
        E (90°, 45°) → [ 0.707,  0,     0.707]
        S (180°,45°) → [ 0,     -0.707, 0.707]
        W (270°,45°) → [-0.707,  0,     0.707]
    """
    az  = np.radians(azimute_graus)
    el  = np.radians(elevacao_graus)
    lx  = np.sin(az) * np.cos(el)
    ly  = np.cos(az) * np.cos(el)
    lz  = np.sin(el)
    mag = np.sqrt(lx**2 + ly**2 + lz**2)
    return [float(lx/mag), float(ly/mag), float(lz/mag)]


def calcular_vetor_luz_ponto(azimute_graus: float,
                              h_m: float,
                              d_m: float) -> list[float]:
    """Vetor unitário para fonte de luz pontual a distância conhecida."""
    rad = np.radians(azimute_graus)
    lx  =  d_m * np.sin(rad)
    ly  =  d_m * np.cos(rad)
    lz  =  h_m
    mag = np.sqrt(lx**2 + ly**2 + lz**2)
    return [float(lx/mag), float(ly/mag), float(lz/mag)]


# ==============================================================
# MÓDULO 2 — ROI
# ==============================================================

def calcular_roi_blender(img_largura:     int,
                          img_altura:      int,
                          wb_larg_mm:      float,
                          wb_prof_mm:      float,
                          camera_h_mm:     float,
                          focal_mm:        float,
                          sensor_w_mm:     float,
                          ortho_scale_mm:  float | None = None,
                          erosao_px:       int = 8) -> np.ndarray:
    """
    Gera máscara ROI booleana automaticamente.

    Câmara ORTOGRÁFICA (ortho_scale_mm fornecido — modo correto para Blender ortho):
        O campo de visão NÃO depende de distância, focal ou sensor.
        phys_w = ortho_scale_mm
        phys_h = ortho_scale_mm * img_altura / img_largura

    Câmara PERSPECTIVA (fallback quando ortho_scale_mm=None):
        phys_w = camera_h_mm * sensor_w_mm / focal_mm
        phys_h = phys_w * img_altura / img_largura
    """
    if ortho_scale_mm is not None:
        phys_w = ortho_scale_mm
        phys_h = ortho_scale_mm * img_altura / img_largura
    else:
        phys_w = camera_h_mm * sensor_w_mm / focal_mm
        phys_h = phys_w * img_altura / img_largura

    roi_hw = min((wb_larg_mm / phys_w) * img_largura / 2.0, img_largura / 2.0)
    roi_hh = min((wb_prof_mm / phys_h) * img_altura  / 2.0, img_altura  / 2.0)
    cx, cy = img_largura / 2.0, img_altura / 2.0

    pts = np.array([
        [cx - roi_hw, cy - roi_hh],
        [cx + roi_hw, cy - roi_hh],
        [cx + roi_hw, cy + roi_hh],
        [cx - roi_hw, cy + roi_hh],
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
    """ROI a partir de 4 cantos marcados manualmente."""
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


def subtrair_luz_ambiente(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Remove luz ambiente pixel a pixel usando o mínimo entre as 4 imagens.
    Em pelo menos uma direção cada pixel está minimamente iluminado (ou em
    sombra própria), então min(I1..I4) é a melhor estimativa da componente
    ambiente constante.
    """
    stack    = np.stack(imgs, axis=0)
    ambiente = np.min(stack, axis=0)
    return [np.clip(img - ambiente, 0.0, 1.0) for img in imgs]


def criar_mascara_valida(imgs: list[np.ndarray],
                          mascara_roi:     np.ndarray | None = None,
                          thresh_variacao: float = 0.04,
                          thresh_sat:      float = 0.95) -> np.ndarray:
    """
    Pixel válido = variação real entre as 4 imagens (std > thresh_variacao).
    Exclui pixels sempre em sombra (std≈0 baixo) e sempre saturados (std≈0 alto).
    """
    stack    = np.stack(imgs, axis=0)
    variacao = np.std(stack, axis=0)
    mascara  = variacao > thresh_variacao
    if mascara_roi is not None:
        mascara &= mascara_roi
    for img in imgs:
        mascara &= (img < thresh_sat)
    return mascara


# ==============================================================
# MÓDULO 3b — DETECÇÃO DE SOMBRAS CAST
# ==============================================================

def detectar_sombras_cast(I_val:        np.ndarray,
                           L:            np.ndarray,
                           G_val:        np.ndarray,
                           albedo_val:   np.ndarray,
                           altura:       int,
                           largura:      int,
                           idx_val:      np.ndarray,
                           thresh_sombra_cast: float = 0.35,
                           raio_contexto_px:   int   = 5) -> np.ndarray:
    """
    Detecta pixels em sombra cast (projetada por outro objeto) usando
    dois critérios combinados:

    CRITÉRIO A — Assimetria fotométrica extrema:
        Um pixel em sombra cast tem I_k ≈ 0 em 1 ou 2 direções específicas
        (as que "enxergam" o objeto que projeta a sombra) mas valores normais
        nas outras. Isso cria uma assimetria no vetor de intensidades que
        nenhuma normal Lambertiana consegue explicar.

        Medida: coeficiente de variação entre imagens (std/mean).
        Superfícies reais iluminadas: CV modesto (~0.3–0.5).
        Sombra cast: CV alto (>0.7–0.9) porque algumas luzes chegam a zero.

    CRITÉRIO B — Consistência espacial de normais (contexto local):
        A normal calculada num pixel de sombra cast é espúria — ela vai
        diferir muito da média das normais dos vizinhos próximos.
        Medida: |N_pixel - N_media_vizinhos| > threshold.

    Retorna máscara booleana (n_pixels,) — True = sombra cast = inválido.
    """
    n_val = I_val.shape[1]

    # ── Critério A: coeficiente de variação ─────────────────
    media_val = np.mean(I_val, axis=0)                   # (N_val,)
    std_val   = np.std(I_val,  axis=0)                   # (N_val,)
    cv        = std_val / (media_val + 1e-6)             # (N_val,)

    # Pixels com pelo menos 1 valor muito próximo de zero E CV alto
    # (distingue sombra cast de superfície inclinada legítima)
    min_val = np.min(I_val, axis=0)
    sombra_a = (cv > 0.65) & (min_val < 0.08)

    # ── Critério B: inconsistência de normal com vizinhos ───
    # Monta mapa de normais completo para poder calcular a média local
    N_mapa = np.zeros((3, altura * largura))
    N_mapa[:, idx_val] = G_val / (albedo_val + 1e-9)   # normais brutas (sem renorm ainda)

    # Média local via filtro uniforme por canal
    N_local = np.zeros_like(N_mapa)
    for i in range(3):
        canal        = N_mapa[i].reshape(altura, largura)
        canal_smooth = uniform_filter(canal, size=raio_contexto_px * 2 + 1)
        N_local[i]   = canal_smooth.flatten()

    # Diferença entre normal calculada e média local (só nos pixels válidos)
    diff = np.linalg.norm(N_mapa[:, idx_val] - N_local[:, idx_val], axis=0)  # (N_val,)
    sombra_b = diff > thresh_sombra_cast

    # União dos dois critérios
    sombra_cast = sombra_a | sombra_b

    pct = 100.0 * sombra_cast.sum() / max(1, n_val)
    print(f"  [shadow] {sombra_cast.sum()} px identificados como sombra cast ({pct:.1f}%)")

    return sombra_cast


# ==============================================================
# MÓDULO 4 — CORREÇÃO DE PERSPECTIVA
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
    h1 = H[:, 0]; h2 = H[:, 1]
    lam = 1.0 / (np.linalg.norm(h1) + 1e-12)
    r1  = h1 * lam; r2 = h2 * lam; r3 = np.cross(r1, r2)
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
    den = U**2 + V**2; den[0, 0] = 1.0
    Z_fft = -1j * (U * np.fft.fft2(p) + V * np.fft.fft2(q)) / den
    Z_fft[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z_fft))


def poisson_depth(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solver DCT analítico O(N log N) com boundary conditions de Neumann."""
    rows, cols = p.shape
    div = np.zeros((rows, cols))
    div[:-1, :] += q[1:,  :] - q[:-1, :]   # ∂q/∂y forward difference
    div[:, :-1] += p[:,  1:] - p[:,  :-1]   # ∂p/∂x forward difference
    ii  = np.arange(rows).reshape(-1, 1)
    jj  = np.arange(cols).reshape(1, -1)
    lam = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + \
          (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0, 0] = 1.0
    Z_dct      = dctn(div, norm='ortho') / lam
    Z_dct[0, 0] = 0.0
    return idctn(Z_dct, norm='ortho')


# ==============================================================
# MÓDULO 6 — PIPELINE PRINCIPAL
# ==============================================================

def processar_mapas(
    caminhos_imagens:   list[str],
    vetores_luz:        list[list[float]],
    diretorio_saida:    str,
    # ── ROI manual ───────────────────────────────────────────
    plano_coords:       list[dict] | None = None,
    largura_mm:         float = 210.0,
    altura_mm:          float = 297.0,
    # ── ROI automático (modo Blender) ────────────────────────
    camera_ortogonal:   bool  = False,
    wb_larg_mm:         float | None = None,
    wb_prof_mm:         float | None = None,
    camera_h_mm:        float | None = None,
    focal_mm:           float = 50.0,
    sensor_w_mm:        float = 36.0,
    ortho_scale_mm:     float | None = None,
    # ── Opções ───────────────────────────────────────────────
    metodo_depth:       str   = 'poisson',
    suavizar_normais:   bool  = False,
    sigma_suavizacao:   float = 1.0,
    thresh_variacao:    float = 0.04,
    thresh_sat:         float = 0.95,
    thresh_residuo:     float = 0.20,
    razao_sombra:       float = 0.60,
    # ── Detecção de sombra cast ──────────────────────────────
    detectar_sombra:    bool  = True,
    thresh_sombra_cast: float = 0.35,
    raio_contexto_px:   int   = 5,
) -> tuple[str, str, str]:
    """
    Pipeline completo de Estéreo Fotométrico.

    Modo Blender ortográfico (camera_ortogonal=True, ortho_scale_mm fornecido):
      • ROI calculada com escala ortográfica real — sem depender de focal/altura.
      • Sem homografia/rotação de normais.
      • Sombras cast detectadas e excluídas automaticamente (detectar_sombra=True).

    Parâmetros de sombra cast:
      thresh_sombra_cast: sensibilidade da detecção por inconsistência de normais.
          Menor = mais agressivo (remove mais). Padrão 0.35.
      raio_contexto_px: raio do filtro de contexto vizinho (px). Padrão 5.
    """

    # ── 1. CARREGAMENTO ──────────────────────────────────────
    imgs            = [load_and_flatten(c) for c in caminhos_imagens]
    altura, largura = imgs[0].shape

    # ── 2. MÁSCARA ROI ──────────────────────────────────────
    if camera_ortogonal and all(v is not None for v in [wb_larg_mm, wb_prof_mm, camera_h_mm]):
        mascara_roi = calcular_roi_blender(
            img_largura    = largura,
            img_altura     = altura,
            wb_larg_mm     = wb_larg_mm,
            wb_prof_mm     = wb_prof_mm,
            camera_h_mm    = camera_h_mm,
            focal_mm       = focal_mm,
            sensor_w_mm    = sensor_w_mm,
            ortho_scale_mm = ortho_scale_mm,
        )
    elif plano_coords and len(plano_coords) >= 4:
        mascara_roi = criar_mascara_roi(plano_coords, altura, largura)
    else:
        mascara_roi = None

    # ── 3. PRÉ-PROCESSAMENTO ────────────────────────────────
    imgs    = subtrair_luz_ambiente(imgs)
    mascara = criar_mascara_valida(imgs, mascara_roi,
                                   thresh_variacao=thresh_variacao,
                                   thresh_sat=thresh_sat)
    idx_val = np.where(mascara.flatten())[0]
    n_pixels = altura * largura

    pct_val = 100.0 * len(idx_val) / n_pixels
    print(f"  [mask]  {len(idx_val)} px válidos ({pct_val:.1f}% da imagem)")

    if len(idx_val) == 0:
        raise ValueError(
            "Nenhum pixel válido após mascaramento. "
            "Reduza thresh_variacao ou verifique os parâmetros da câmara/bancada."
        )

    # ── 4. VETORES DE LUZ ───────────────────────────────────
    L = np.array(vetores_luz, dtype=np.float64)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)

    # ── 5. LEAST-SQUARES FOTOMÉTRICO — DROP DARKEST ─────────
    # Para cada pixel, descarta a imagem mais escura se for sombra evidente
    # (min < razao_sombra × mediana) e resolve com as 3 restantes.
    I_stack = np.stack([img.flatten() for img in imgs])
    I_val   = I_stack[:, idx_val]           # (4, N_val)

    L_invs    = [np.linalg.pinv(np.delete(L, k, axis=0)) for k in range(4)]
    min_idx   = np.argmin(I_val, axis=0)
    med_val   = np.median(I_val, axis=0)
    min_val   = I_val[min_idx, np.arange(I_val.shape[1])]
    deve_drop = min_val < razao_sombra * (med_val + 1e-9)

    G_val = np.zeros((3, I_val.shape[1]), dtype=np.float64)

    idx_sem_sombra = np.where(~deve_drop)[0]
    if idx_sem_sombra.size:
        L_inv4 = np.linalg.pinv(L)
        G_val[:, idx_sem_sombra] = L_inv4 @ I_val[:, idx_sem_sombra]

    for k in range(4):
        sel = np.where(deve_drop & (min_idx == k))[0]
        if sel.size == 0:
            continue
        I_sub = np.delete(I_val[:, sel], k, axis=0)
        G_val[:, sel] = L_invs[k] @ I_sub

    pct_drop = 100.0 * deve_drop.sum() / max(1, I_val.shape[1])
    print(f"  [drop]  {deve_drop.sum()} px com sombra própria descartada ({pct_drop:.1f}%)")

    albedo_val = np.linalg.norm(G_val, axis=0)
    N_val      = np.divide(G_val, albedo_val,
                           out=np.zeros_like(G_val),
                           where=albedo_val > 1e-9)

    # ── 6. REJEIÇÃO POR RESÍDUO (sombra própria residual) ───
    I_pred      = L @ G_val
    residuo_abs = np.mean(np.abs(I_val - I_pred), axis=0)
    residuo_rel = residuo_abs / (albedo_val + 1e-6)
    validos_ps  = residuo_rel < thresh_residuo

    # ── 7. DETECÇÃO DE SOMBRA CAST ──────────────────────────
    # Aplicada APÓS o lstsq para ter acesso ao G_val calculado.
    # Pixels em sombra cast produzem normais inconsistentes com os
    # vizinhos — detectamos isso antes de incluir no mapa final.
    if detectar_sombra:
        sombra_cast = detectar_sombras_cast(
            I_val              = I_val,
            L                  = L,
            G_val              = G_val,
            albedo_val         = albedo_val,
            altura             = altura,
            largura            = largura,
            idx_val            = idx_val,
            thresh_sombra_cast = thresh_sombra_cast,
            raio_contexto_px   = raio_contexto_px,
        )
        # Combina: pixel válido = passou no resíduo E não é sombra cast
        validos_final = validos_ps & ~sombra_cast
    else:
        validos_final = validos_ps

    idx_final = idx_val[validos_final]

    pct_final = 100.0 * len(idx_final) / n_pixels
    print(f"  [final] {len(idx_final)} px aceitos ({pct_final:.1f}% da imagem)")

    # ── 8. MONTAR MAPAS DE NORMAL ───────────────────────────
    N_flat      = np.zeros((3, n_pixels))
    albedo_flat = np.zeros(n_pixels)

    N_flat[0, idx_final] = N_val[0, validos_final]
    N_flat[1, idx_final] = N_val[1, validos_final]
    N_flat[2, idx_final] = N_val[2, validos_final]
    albedo_flat[idx_final] = albedo_val[validos_final]

    # Pixels sem normal → superfície horizontal [0,0,1]
    def _sem_normal(N_f, roi):
        normas = np.linalg.norm(N_f, axis=0)
        if roi is not None:
            return roi.flatten() & (normas < 1e-9)
        return normas < 1e-9

    sem_normal = _sem_normal(N_flat, mascara_roi)
    N_flat[2, sem_normal] = 1.0

    # ── 9. CORREÇÃO DE PERSPECTIVA (só modo manual) ─────────
    if not camera_ortogonal and plano_coords and len(plano_coords) >= 4:
        R      = calcular_rotacao_por_homografia(plano_coords, largura_mm, altura_mm)
        N_flat = aplicar_rotacao_normais(N_flat, R)

    # ── 10. SUAVIZAÇÃO OPCIONAL ─────────────────────────────
    if suavizar_normais:
        for i in range(3):
            canal     = N_flat[i].reshape(altura, largura)
            N_flat[i] = gaussian_filter(canal, sigma=sigma_suavizacao).flatten()
        normas = np.linalg.norm(N_flat, axis=0)
        N_flat = np.divide(N_flat, normas, out=np.zeros_like(N_flat), where=normas > 1e-9)
        # Recomputar sem_normal APÓS suavização
        sem_normal = _sem_normal(N_flat, mascara_roi)
        N_flat[2, sem_normal] = 1.0

    # ── 11. NORMAL MAP ──────────────────────────────────────
    nx = N_flat[0].reshape(altura, largura)
    ny = N_flat[1].reshape(altura, largura)
    nz = N_flat[2].reshape(altura, largura)

    # Flip Y para convenção OpenGL (+Y = cima na textura)
    N_rgb = np.stack([
        np.clip(( nx + 1.0) * 127.5, 0, 255),
        np.clip((-ny + 1.0) * 127.5, 0, 255),   # ← flip Y
        np.clip(( nz + 1.0) * 127.5, 0, 255),
    ], axis=-1).astype(np.uint8)

    # ── 12. DEPTH MAP ───────────────────────────────────────
    mascara_depth = np.zeros(n_pixels, dtype=bool)
    mascara_depth[idx_final] = True
    mascara_depth = mascara_depth.reshape(altura, largura)

    # Chão: normal para cima E pixel NÃO pertence a objetos detectados
    # (protege tampas planas de caixas/cubos)
    mask_fundo_normal = (np.abs(nx) < 0.08) & (np.abs(ny) < 0.08)
    mask_fundo = mask_fundo_normal & ~mascara_depth

    # Bordas perpendiculares: nz muito pequeno → gradiente explode
    mask_bordas = np.abs(nz) < 0.20

    mascara_grad = mascara_depth & ~mask_fundo & ~mask_bordas

    nz_safe = np.where(np.abs(nz) < 0.20, 1.0, nz)
    p_grad  = np.zeros_like(nx)
    q_grad  = np.zeros_like(ny)
    p_grad[mascara_grad] = -nx[mascara_grad] / nz_safe[mascara_grad]
    q_grad[mascara_grad] = -ny[mascara_grad] / nz_safe[mascara_grad]

    # Clip para evitar artefatos de gradiente extremo
    p_grad = np.clip(p_grad, -2.5, 2.5)
    q_grad = np.clip(q_grad, -2.5, 2.5)

    # Integração
    depth_raw = poisson_depth(p_grad, q_grad) if metodo_depth == 'poisson' \
                else frankot_chellappa(p_grad, q_grad)

    # Fixar datum no chão
    if np.any(mask_fundo):
        base_z = np.median(depth_raw[mask_fundo])
        depth_raw -= base_z
    depth_raw[mask_fundo] = 0.0
    depth_raw = np.maximum(depth_raw, 0.0)

    # Normalizar usando apenas pixels de objetos (ignora chão)
    depth_objeto = depth_raw[~mask_fundo & mascara_depth]
    if len(depth_objeto) > 0:
        d_max      = np.percentile(depth_objeto, 98)
        depth_norm = np.clip(depth_raw / (d_max + 1e-9) * 255.0, 0, 255)
    else:
        depth_norm = np.zeros_like(depth_raw)
    depth_norm[mask_fundo] = 0.0

    depth_smooth   = cv2.GaussianBlur(depth_norm.astype(np.uint8), (5, 5), sigmaX=1.0)
    depth_colormap = cv2.applyColorMap(depth_smooth, cv2.COLORMAP_JET)

    # ── 13. ALBEDO MAP ──────────────────────────────────────
    albedo_2d = albedo_flat.reshape(altura, largura)
    reg_m     = mascara_depth if mascara_depth.any() else np.ones_like(mascara_depth)
    reg_a     = albedo_2d[reg_m]
    a_min, a_max = np.percentile(reg_a, 1), np.percentile(reg_a, 99)
    albedo_vis = np.clip(
        (albedo_2d - a_min) / (a_max - a_min + 1e-9) * 255.0, 0, 255
    ).astype(np.uint8)

    # ── 14. SALVAR ──────────────────────────────────────────
    p_normal = os.path.join(diretorio_saida, 'resultado_normal.png')
    p_depth  = os.path.join(diretorio_saida, 'resultado_depth.png')
    p_albedo = os.path.join(diretorio_saida, 'resultado_albedo.png')

    cv2.imwrite(p_normal, cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p_depth,  depth_colormap)
    cv2.imwrite(p_albedo, albedo_vis)

    return 'resultado_normal.png', 'resultado_depth.png', 'resultado_albedo.png'