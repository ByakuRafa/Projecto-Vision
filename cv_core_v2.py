"""
cv_core_v2.py — Estéreo Fotométrico com Segmentação por Flood Fill
===================================================================

DIAGNÓSTICO DA VERSÃO ANTERIOR:
    O bug central era queda_de_luz = I_min < (I_mean * 0.70).
    Após subtração de ambiente, uma parede voltada para Norte recebe
    luz ≈ 0 de 3 das 4 direções → I_min ≈ 0 → sempre flagada como sombra.
    As paredes dos objetos eram removidas ANTES do cálculo de normais.

CORREÇÕES APLICADAS:
    1. segmentar_cena() — mantém o flood fill (boa ideia), remove queda_de_luz.
       Resultado: 3 regiões limpas: chão, topos, paredes.
    2. calcular_normais_segmentadas() — Drop-Darkest para sombra própria,
       resíduo fotométrico para sombra cast. Topos e chão → [0,0,1].
    3. integrar_depth() — usa CAMPO COMPLETO de normais (não só paredes),
       corrige nz_safe, remove np.maximum(0) que destruía geometria relativa.
"""

import cv2
import numpy as np
from scipy.fft import dctn, idctn
import os


# ==============================================================
# 1. SEGMENTAÇÃO POR FLOOD FILL
# ==============================================================

def segmentar_cena(imgs: list[np.ndarray],
                   thresh_fundo_std: float = 0.03) -> dict:
    """
    Segmenta a cena em 3 regiões usando variação entre imagens + flood fill.

    REGIÃO 1 — CHÃO (fundo_limpo):
        Pixels planos (std < thresh) conectados às bordas da imagem.
        São o chão/bancada visto de cima.

    REGIÃO 2 — TOPOS (topos_objetos):
        Pixels planos (std < thresh) NÃO conectados às bordas.
        São as tampas horizontais dos objetos (cubos, cilindros).

    REGIÃO 3 — PAREDES (paredes_objetos):
        Todos os pixels com variação (std ≥ thresh).
        São as faces inclinadas/verticais dos objetos E as sombras cast.
        O Drop-Darkest + resíduo separam sombra de parede real nesta etapa.

    POR QUE REMOVER queda_de_luz:
        I_min ≈ 0 é a condição NORMAL de qualquer superfície que não vê
        todas as 4 luzes — isto é, TODA parede. Com razao_sombra=0.70,
        qualquer pixel com I_min < 70% da média vira "sombra", o que
        inclui 100% das paredes. Não há como distinguir sombra cast de
        shadow-own por esta métrica depois da subtração de ambiente.
    """
    stack  = np.stack(imgs, axis=0)
    I_std  = np.std(stack, axis=0)
    H, W   = imgs[0].shape

    # Pixels com baixa variação = iluminados igualmente por todas as direções
    planos = (I_std < thresh_fundo_std)
    im     = (planos * 255).astype(np.uint8)

    # Flood fill a partir dos 4 cantos para identificar chão (conectado à borda)
    mask_fill = np.zeros((H + 2, W + 2), np.uint8)
    for corner in [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]:
        cv2.floodFill(im, mask_fill, corner, 128)

    mask_chao   = (im == 128)   # plano + conectado à borda = chão
    mask_topos  = (im == 255)   # plano + ilhado = tampa de objeto
    mask_paredes = ~planos      # variação alta = parede, slope, ou sombra cast

    # Chão "total": chão limpo + região vizinha ao chão (sombras cast ficam aqui)
    # Não zeramos diretamente — deixamos o solver decidir pela normal [0,0,1].
    kernel = np.ones((3, 3), np.uint8)
    mask_chao_expandido = cv2.dilate(
        mask_chao.astype(np.uint8), kernel, iterations=2
    ).astype(bool)

    return {
        'chao':     mask_chao,
        'topos':    mask_topos,
        'paredes':  mask_paredes,
        'chao_exp': mask_chao_expandido,   # para debug
    }


# ==============================================================
# 2. RESOLUÇÃO FOTOMÉTRICA COM DROP-DARKEST + RESÍDUO
# ==============================================================

def calcular_normais_segmentadas(imgs: list[np.ndarray],
                                  vetores_luz: list[list[float]],
                                  mascaras: dict,
                                  razao_sombra: float = 0.60,
                                  thresh_residuo: float = 0.25) -> tuple:
    """
    Calcula normais fotométricas aplicando dois filtros de sombra:

    DROP-DARKEST (sombra própria):
        Para cada pixel, descarta a luz mais escura se for < razao_sombra × mediana.
        Superfícies que "viram as costas" para uma luz recebem zero nessa direção —
        solucionar com 3 luzes é mais estável que com 4 incluindo o zero.

    RESÍDUO FOTOMÉTRICO (sombra cast):
        Depois de calcular G, prediz as intensidades esperadas (Î = L·G).
        Cast shadows têm Î >> I medido → resíduo alto → excluídos.
        Sombra própria já foi tratada pelo Drop-Darkest, então o resíduo
        restante identifica sombras cast de outros objetos.

    PREENCHIMENTO:
        Chão e topos → normal [0,0,1] (superfície horizontal).
        Paredes excluídas pelo resíduo → normal [0,0,1] (interpolação futura).
    """
    H, W = imgs[0].shape
    n_pixels = H * W

    L     = np.array(vetores_luz, dtype=np.float64)
    L     = L / np.linalg.norm(L, axis=1, keepdims=True)
    L_inv = np.linalg.pinv(L)                               # (3, 4)
    L_invs = [np.linalg.pinv(np.delete(L, k, axis=0)) for k in range(4)]

    # Subtrair luz ambiente pixel a pixel (mínimo entre as 4 imagens)
    stack    = np.stack(imgs, axis=0)
    ambiente = np.min(stack, axis=0)
    imgs_corr = [np.clip(img - ambiente, 0.0, 1.0) for img in imgs]

    I_stack = np.stack([img.flatten() for img in imgs_corr], axis=0)  # (4, N)

    # Computar apenas nas paredes (variação real)
    mask_calc = mascaras['paredes']
    idx_calc  = np.where(mask_calc.flatten())[0]
    I_calc    = I_stack[:, idx_calc]                         # (4, N_calc)

    # ── DROP-DARKEST ────────────────────────────────────────
    min_idx   = np.argmin(I_calc, axis=0)
    med_val   = np.median(I_calc, axis=0)
    min_val   = I_calc[min_idx, np.arange(I_calc.shape[1])]
    deve_drop = min_val < razao_sombra * (med_val + 1e-9)

    G_calc = np.zeros((3, I_calc.shape[1]), dtype=np.float64)

    idx_sem = np.where(~deve_drop)[0]
    if idx_sem.size:
        G_calc[:, idx_sem] = L_inv @ I_calc[:, idx_sem]

    for k in range(4):
        sel = np.where(deve_drop & (min_idx == k))[0]
        if not sel.size:
            continue
        I_sub = np.delete(I_calc[:, sel], k, axis=0)        # (3, |sel|)
        G_calc[:, sel] = L_invs[k] @ I_sub

    albedo = np.linalg.norm(G_calc, axis=0)
    N_calc = np.divide(G_calc, albedo,
                       out=np.zeros_like(G_calc),
                       where=albedo > 1e-9)

    # ── RESÍDUO: filtrar sombra cast ─────────────────────────
    I_pred     = L @ G_calc                                  # (4, N_calc)
    res_abs    = np.mean(np.abs(I_calc - I_pred), axis=0)   # (N_calc,)
    res_rel    = res_abs / (albedo + 1e-6)
    validos    = res_rel < thresh_residuo                    # bool (N_calc,)

    idx_final  = idx_calc[validos]

    pct_drop = 100.0 * deve_drop.sum() / max(1, I_calc.shape[1])
    pct_res  = 100.0 * (~validos).sum() / max(1, I_calc.shape[1])
    pct_ok   = 100.0 * validos.sum() / max(1, n_pixels)
    print(f"  drop-darkest: {pct_drop:.1f}%  resíduo: {pct_res:.1f}%  aceitos: {pct_ok:.1f}%")

    # ── MONTAR MAPA COMPLETO ─────────────────────────────────
    N_flat = np.zeros((3, n_pixels), dtype=np.float64)

    # Paredes válidas → normal calculada
    N_flat[0, idx_final] = N_calc[0, validos]
    N_flat[1, idx_final] = N_calc[1, validos]
    N_flat[2, idx_final] = N_calc[2, validos]

    # Chão + Topos + paredes excluídas → normal [0,0,1]
    # [0,0,1] = superfície horizontal apontando para a câmara
    sem_normal = np.linalg.norm(N_flat, axis=0) < 1e-9
    N_flat[2, sem_normal] = 1.0

    nx = N_flat[0].reshape(H, W)
    ny = N_flat[1].reshape(H, W)
    nz = N_flat[2].reshape(H, W)

    return nx, ny, nz, mascaras['chao']


# ==============================================================
# 3. INTEGRAÇÃO DE PROFUNDIDADE
# ==============================================================

def integrar_depth(nx: np.ndarray,
                   ny: np.ndarray,
                   nz: np.ndarray,
                   mask_chao: np.ndarray) -> np.ndarray:
    """
    Integra gradientes de superfície para obter mapa de profundidade.

    USA O MAPA COMPLETO de normais (não só paredes).
    Pixels planos (chão, topos) têm gradiente = 0 → o solver Poisson
    propaga altura das bordas das paredes para o interior dos topos.
    Sem isso o solver recebe só "slivers" e produz depth nulo.

    CORREÇÕES em relação à versão anterior:
      - nz_safe: clamp correto (±0.05) em vez de 1.0
        (1.0 introduzia gradientes falsos fora da máscara ativa)
      - sem np.maximum(depth_raw, 0.0): o solver dá valores relativos;
        clipar em zero destrói a geometria — a subtract de base_z já corrige.
    """
    # Clamp correto: evita |gradiente| → ∞ sem introduzir valor falso
    nz_safe = np.where(np.abs(nz) < 0.05,
                       np.sign(nz + 1e-12) * 0.05,
                       nz)

    # Gradientes em toda a imagem (zeros no chão = informação plana válida)
    tem_normal = np.abs(nz) > 1e-9     # todos os pixels com qualquer normal
    p = np.where(tem_normal, np.clip(-nx / nz_safe, -4.0, 4.0), 0.0)
    q = np.where(tem_normal, np.clip(-ny / nz_safe, -4.0, 4.0), 0.0)

    # Solver Poisson via DCT
    rows, cols = p.shape
    div = np.zeros((rows, cols))
    div[:-1, :] += q[1:,  :] - q[:-1, :]   # ∂q/∂y forward
    div[:, :-1] += p[:,  1:] - p[:,  :-1]   # ∂p/∂x forward

    ii  = np.arange(rows).reshape(-1, 1)
    jj  = np.arange(cols).reshape(1, -1)
    lam = (2.0 * np.cos(np.pi * ii / rows) - 2.0) + \
          (2.0 * np.cos(np.pi * jj / cols) - 2.0)
    lam[0, 0]   = 1.0
    Z_dct       = dctn(div, norm='ortho') / lam
    Z_dct[0, 0] = 0.0
    depth_raw   = idctn(Z_dct, norm='ortho')

    # Fixar datum: subtrair mediana do chão → chão fica em Z≈0
    if np.any(mask_chao):
        base_z    = np.median(depth_raw[mask_chao])
        depth_raw = depth_raw - base_z

    # Zeramos o chão APÓS a subtração (elimina ringing residual do solver)
    depth_raw[mask_chao] = 0.0
    # NÃO fazemos np.maximum(0): objetos cujo topo ficou levemente negativo
    # por imprecisão seriam zerados, perdendo-se a distinção de altura.

    # Normalizar usando percentis dos pixels de objetos (não chão)
    pixels_obj = depth_raw[~mask_chao]
    if len(pixels_obj) > 0 and pixels_obj.max() > pixels_obj.min():
        d_min = np.percentile(pixels_obj, 1)
        d_max = np.percentile(pixels_obj, 99)
        depth_norm = np.clip(
            (depth_raw - d_min) / (d_max - d_min + 1e-9) * 255.0, 0, 255
        )
    else:
        depth_norm = np.zeros_like(depth_raw)

    depth_norm[mask_chao] = 0.0
    return depth_norm


# ==============================================================
# 4. PIPELINE PRINCIPAL
# ==============================================================

def processar_perspectiva(caminhos_imagens: list[str],
                           vetores_luz: list[list[float]],
                           dir_saida: str,
                           thresh_fundo_std: float = 0.03,
                           razao_sombra: float = 0.60,
                           thresh_residuo: float = 0.25):
    """
    Pipeline completo: 4 imagens → normal map + depth map + masks de debug.

    Parâmetros ajustáveis:
      thresh_fundo_std : desvio padrão máximo para pixel ser "plano".
                         Suba (0.05) se o chão não for detectado.
                         Baixe (0.02) se objetos forem absorvidos no chão.
      razao_sombra     : agressividade do drop-darkest (0.60 = moderado).
      thresh_residuo   : tolerância para cast shadows (0.25 = moderado).
    """
    print("1. Carregando imagens...")
    imgs = []
    for c in caminhos_imagens:
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Não encontrado: {c}")
        imgs.append(img.astype(np.float64) / 255.0)

    print("2. Segmentando cena (flood fill)...")
    mascaras = segmentar_cena(imgs, thresh_fundo_std)

    # Mapas de debug (ajudam a calibrar thresh_fundo_std)
    cv2.imwrite(os.path.join(dir_saida, 'debug_chao.png'),
                mascaras['chao'].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(dir_saida, 'debug_topos.png'),
                mascaras['topos'].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(dir_saida, 'debug_paredes.png'),
                mascaras['paredes'].astype(np.uint8) * 255)

    print("3. Calculando normais fotométricas...")
    nx, ny, nz, mask_chao = calcular_normais_segmentadas(
        imgs, vetores_luz, mascaras, razao_sombra, thresh_residuo
    )

    # Normal map (convenção OpenGL: flip Y)
    N_rgb = np.stack([
        np.clip(( nx + 1.0) * 127.5, 0, 255),
        np.clip((-ny + 1.0) * 127.5, 0, 255),
        np.clip(( nz + 1.0) * 127.5, 0, 255),
    ], axis=-1).astype(np.uint8)
    cv2.imwrite(os.path.join(dir_saida, 'resultado_normal.png'),
                cv2.cvtColor(N_rgb, cv2.COLOR_RGB2BGR))

    print("4. Integrando depth map...")
    depth_norm = integrar_depth(nx, ny, nz, mask_chao)

    depth_suave    = cv2.GaussianBlur(depth_norm.astype(np.uint8), (5, 5), sigmaX=1.0)
    depth_colormap = cv2.applyColorMap(depth_suave, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(dir_saida, 'resultado_depth.png'), depth_colormap)

    print("Concluído. Verifique debug_chao.png e debug_paredes.png para calibrar.")