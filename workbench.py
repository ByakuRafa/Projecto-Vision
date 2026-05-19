"""
workbench.py — Visão Computacional para Bancada com Iluminação Controlada
Compatível com render do Blender ou câmera física estática.

╔══════════════════════════════════════════════════════════════╗
║  GUIA DE CONFIGURAÇÃO POR MODO DE CÂMERA                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MODO 'topdown'  (recomendado para começar)                  ║
║  Blender: Camera > Type = Orthographic, Rotation X = 0°     ║
║  Sem distorção de perspectiva, homografia é só escala.       ║
║  Altura dos objetos não afeta o centroide detectado.         ║
║                                                              ║
║  MODO 'perspective'                                          ║
║  Blender: Camera > Type = Perspective, inclinação ~80-90°   ║
║  Homografia corrige distorção do plano do chão.              ║
║  Objetos altos têm centroide visual deslocado da base.       ║
║                                                              ║
║  MODO 'isometric'  (câmera a 45°)                           ║
║  Blender: Camera > Type = Orthographic, Rotation X = 45°    ║
║  Idem perspective, mas sem distorção radial.                 ║
║  Usa ponto inferior do contorno como âncora no chão.         ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  CONFIGURAÇÃO BLENDER RECOMENDADA (todos os modos):         ║
║  • Render Engine: EEVEE (mais rápido para testes)            ║
║  • Resolution: 1280×720 ou 1920×1080                        ║
║  • Color Management > View Transform: Standard              ║
║  • Iluminação: Sun lamp de cima (força = 2-5)               ║
║  • Background: Surface Material, cor sólida (facilita HSV)  ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


# ──────────────────────────────────────────────────────────────
# TIPOS
# ──────────────────────────────────────────────────────────────

class ModoCamera(str, Enum):
    TOPDOWN     = 'topdown'      # Ortográfico de cima, ângulo 0°
    PERSPECTIVE = 'perspective'  # Perspectivo inclinado, qualquer ângulo
    ISOMETRIC   = 'isometric'    # Ortográfico a 45° (sem distorção radial)


@dataclass
class FaixaHSV:
    """
    Representa um intervalo de cor no espaço HSV do OpenCV.

    O vermelho exige DUAS faixas porque seu matiz (H) fica nos dois
    extremos da escala circular: H≈0 (vermelho puro) e H≈175–180
    (vermelho escuro / bordô). Uma única inRange() perde metade dos pixels.

    Para outras cores (azul, verde, amarelo): use apenas lower/upper1,
    mantenha lower2=None.
    """
    lower1: np.ndarray
    upper1: np.ndarray
    lower2: Optional[np.ndarray] = None
    upper2: Optional[np.ndarray] = None

    @classmethod
    def para_vermelho(cls) -> 'FaixaHSV':
        """
        Faixa calibrada para o cubo vermelho do render Blender.
        Cobre H=[0-10] (vermelho puro) e H=[165-180] (vermelho escuro/sombra).
        """
        return cls(
            lower1=np.array([  0, 120,  80], dtype=np.uint8),
            upper1=np.array([ 10, 255, 255], dtype=np.uint8),
            lower2=np.array([165, 120,  80], dtype=np.uint8),
            upper2=np.array([180, 255, 255], dtype=np.uint8),
        )

    @classmethod
    def para_verde(cls) -> 'FaixaHSV':
        return cls(
            lower1=np.array([ 40, 80, 80], dtype=np.uint8),
            upper1=np.array([ 80, 255, 255], dtype=np.uint8),
        )

    @classmethod
    def para_azul(cls) -> 'FaixaHSV':
        return cls(
            lower1=np.array([100, 80, 80], dtype=np.uint8),
            upper1=np.array([140, 255, 255], dtype=np.uint8),
        )


@dataclass
class DeteccaoResultado:
    """Resultado completo da detecção de uma peça em um frame."""
    encontrado:    bool
    x_mm:          float = 0.0
    y_mm:          float = 0.0
    x_pixel:       int   = 0
    y_pixel:       int   = 0
    area_pixels:   float = 0.0
    angulo_graus:  float = 0.0
    frame_debug:   Optional[np.ndarray] = None

    def __repr__(self):
        if not self.encontrado:
            return "DeteccaoResultado(não encontrado)"
        return (f"DeteccaoResultado(x={self.x_mm:.1f}mm, y={self.y_mm:.1f}mm, "
                f"area={self.area_pixels:.0f}px², ang={self.angulo_graus:.1f}°)")


# ──────────────────────────────────────────────────────────────
# SISTEMA PRINCIPAL
# ──────────────────────────────────────────────────────────────

class WorkbenchVisionSetup:
    """
    Sistema de visão para bancada com câmera estática e iluminação controlada.

    Exemplo rápido (cubo vermelho do render Blender):
        sistema = WorkbenchVisionSetup(
            modo=ModoCamera.TOPDOWN,
            cor=FaixaHSV.para_vermelho(),
        )
        resultado = sistema.process_frame(cv2.imread("render.png"))
        print(resultado)
    """

    # ──────────────────────────────────────────────────────────
    # PONTOS DE CALIBRAÇÃO
    #
    # COMO OBTER NO BLENDER:
    #   1. Renderize uma imagem com a bancada visível
    #   2. Abra no Image Editor do Blender
    #   3. Passe o mouse nos 4 cantos da área de trabalho
    #      e anote as coordenadas X,Y da barra de status
    #   4. Esses são os PONTOS_IMAGEM_PX
    #   5. PONTOS_REAIS_MM vêm do grid do Blender (em mm)
    #
    # DICA: coloque 4 esferas coloridas pequenas nos cantos da
    # bancada no Blender — são fáceis de identificar no render.
    # ──────────────────────────────────────────────────────────

    # Pixels dos 4 cantos da área de trabalho, ORDEM HORÁRIA: TL→TR→BR→BL
    # Ajuste para a resolução do seu render (abaixo: 1280×720)
    PONTOS_IMAGEM_PX = np.array([
        [  64,  40],   # TL
        [1216,  40],   # TR
        [1216, 680],   # BR
        [  64, 680],   # BL
    ], dtype=np.float32)

    # Coordenadas reais desses mesmos cantos em mm (origem = TL)
    PONTOS_REAIS_MM = np.array([
        [  0,   0],    # TL
        [600,   0],    # TR  — bancada 500mm de largura
        [600, 400],    # BR  — bancada 350mm de profundidade
        [  0, 400],    # BL
    ], dtype=np.float32)

    AREA_MINIMA_PX = 300

    def __init__(
        self,
        modo: ModoCamera = ModoCamera.TOPDOWN,
        cor:  FaixaHSV   = None,
    ):
        self.modo = modo
        self.cor  = cor if cor is not None else FaixaHSV.para_vermelho()
        self.H    = self._calcular_homografia()

    # ──────────────────────────────────────────────────────────
    # CALIBRAÇÃO
    # ──────────────────────────────────────────────────────────

    def _calcular_homografia(self) -> np.ndarray:
        H, _ = cv2.findHomography(self.PONTOS_IMAGEM_PX, self.PONTOS_REAIS_MM)
        if H is None:
            raise ValueError(
                "Homografia inválida. Verifique os pontos de calibração: "
                "devem ser 4 pontos não-colineares no formato (N,2)."
            )
        return H

    def recalibrar(self, pontos_px: np.ndarray, pontos_mm: np.ndarray) -> None:
        """
        Recalibra dinamicamente a homografia com novos pontos.
        Útil quando a câmera é movida ou o espaço muda entre sessões.
        """
        H, _ = cv2.findHomography(
            pontos_px.astype(np.float32),
            pontos_mm.astype(np.float32)
        )
        if H is not None:
            self.H = H
            print("Homografia recalibrada.")

    @staticmethod
    def calibrar_hsv(imagem_bgr: np.ndarray) -> None:
        """
        Utilitário interativo: clique na peça para ver os valores HSV.
        Pressione qualquer tecla para sair.
        """
        hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

        def ao_clicar(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                h, s, v = hsv[y, x]
                print(f"\nHSV em ({x},{y}): H={h}  S={s}  V={v}")
                if h < 15 or h > 165:
                    print("  ⚠  Vermelho detectado — use FaixaHSV.para_vermelho() como base")
                    print(f"  lower1=[0, {max(0,s-50)}, {max(0,v-50)}]  upper1=[10, 255, 255]")
                    print(f"  lower2=[165, {max(0,s-50)}, {max(0,v-50)}]  upper2=[180, 255, 255]")
                else:
                    print(f"  lower=[{max(0,h-12)}, {max(0,s-50)}, {max(0,v-50)}]")
                    print(f"  upper=[{min(180,h+12)}, 255, 255]")

        titulo = "Calibrar HSV — clique na peca, tecla para sair"
        cv2.namedWindow(titulo)
        cv2.setMouseCallback(titulo, ao_clicar)
        cv2.imshow(titulo, imagem_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ──────────────────────────────────────────────────────────
    # SEGMENTAÇÃO
    # ──────────────────────────────────────────────────────────

    def _segmentar(self, frame: np.ndarray) -> np.ndarray:
        """Gera máscara binária da peça. Suporta faixa dupla (vermelho)."""
        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, self.cor.lower1, self.cor.upper1)

        if self.cor.lower2 is not None and self.cor.upper2 is not None:
            mascara2 = cv2.inRange(hsv, self.cor.lower2, self.cor.upper2)
            mascara  = cv2.bitwise_or(mascara, mascara2)

        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mascara

    # ──────────────────────────────────────────────────────────
    # PONTO DE ANCORAGEM NO CHÃO
    # ──────────────────────────────────────────────────────────

    def _ponto_ancora(self, contorno: np.ndarray, cx: int, cy: int) -> Tuple[int, int]:
        """
        Retorna o pixel que representa a posição da peça NO CHÃO.

        TOPDOWN:            centroide — sem paralaxe de altura
        PERSPECTIVE/ISO:    ponto mais BAIXO do contorno
                            (maior Y na imagem = base do objeto na cena 3D)

        Por que não usar o centroide em câmeras inclinadas?
        O cubo vermelho da imagem Blender tem ~50mm de altura. Sua face
        superior fica ~25px acima da base na projeção — o centroide visual
        aponta para o meio do cubo, não para onde ele está no chão.
        O ponto mais baixo do contorno coincide exatamente com a borda
        inferior do objeto, que está em contato com o chão.
        """
        if self.modo == ModoCamera.TOPDOWN:
            return cx, cy

        pontos = contorno.reshape(-1, 2)
        idx    = np.argmax(pontos[:, 1])
        return int(pontos[idx][0]), int(pontos[idx][1])

    # ──────────────────────────────────────────────────────────
    # PROCESSAMENTO
    # ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DeteccaoResultado:
        """
        Detecta a peça no frame e retorna coordenadas reais em mm.

        Args:
            frame: imagem BGR (render Blender ou captura de câmera)
        """
        nao_encontrado = DeteccaoResultado(encontrado=False, frame_debug=frame)
        mascara = self._segmentar(frame)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        validos = [c for c in contornos if cv2.contourArea(c) >= self.AREA_MINIMA_PX]
        if not validos:
            return nao_encontrado

        maior = max(validos, key=cv2.contourArea)
        M     = cv2.moments(maior)
        if M["m00"] == 0:
            return nao_encontrado

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        _, _, angulo = cv2.minAreaRect(maior)

        ax, ay  = self._ponto_ancora(maior, cx, cy)
        pt_real = cv2.perspectiveTransform(
            np.array([[[ax, ay]]], dtype=np.float32), self.H
        )
        x_mm = float(pt_real[0][0][0])
        y_mm = float(pt_real[0][0][1])

        # Debug visual
        debug = frame.copy()
        cv2.drawContours(debug, [maior], -1, (0, 220, 0), 2)
        cv2.circle(debug, (cx, cy), 5, (255, 100, 0), -1)

        if self.modo != ModoCamera.TOPDOWN:
            cv2.circle(debug, (ax, ay), 6, (0, 220, 220), -1)
            cv2.line(debug, (cx, cy), (ax, ay), (0, 220, 220), 1)

        label  = f"{self.modo.value.upper()}  X={x_mm:.1f}  Y={y_mm:.1f}mm"
        origem = (max(ax - 80, 4), max(ay - 16, 14))
        cv2.putText(debug, label, origem, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(debug, label, origem, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return DeteccaoResultado(
            encontrado=True, x_mm=x_mm, y_mm=y_mm,
            x_pixel=ax, y_pixel=ay,
            area_pixels=cv2.contourArea(maior),
            angulo_graus=angulo,
            frame_debug=debug,
        )

    def process_multiplas_pecas(self, frame: np.ndarray) -> List[DeteccaoResultado]:
        """
        Detecta TODAS as peças da cor configurada, ordenadas por tamanho.
        """
        mascara   = self._segmentar(frame)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        validos   = sorted(
            [c for c in contornos if cv2.contourArea(c) >= self.AREA_MINIMA_PX],
            key=cv2.contourArea, reverse=True
        )

        resultados = []
        debug      = frame.copy()

        for contorno in validos:
            M = cv2.moments(contorno)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            ax, ay = self._ponto_ancora(contorno, cx, cy)
            _, _, angulo = cv2.minAreaRect(contorno)

            pt_real = cv2.perspectiveTransform(
                np.array([[[ax, ay]]], dtype=np.float32), self.H
            )
            x_mm = float(pt_real[0][0][0])
            y_mm = float(pt_real[0][0][1])

            cv2.drawContours(debug, [contorno], -1, (0, 220, 0), 2)
            cv2.circle(debug, (ax, ay), 5, (0, 220, 220), -1)
            cv2.putText(debug, f"{x_mm:.0f},{y_mm:.0f}mm",
                        (ax - 40, ay - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,0,0), 3)
            cv2.putText(debug, f"{x_mm:.0f},{y_mm:.0f}mm",
                        (ax - 40, ay - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255,255,255), 1)

            resultados.append(DeteccaoResultado(
                encontrado=True, x_mm=x_mm, y_mm=y_mm,
                x_pixel=ax, y_pixel=ay,
                area_pixels=cv2.contourArea(contorno),
                angulo_graus=angulo,
                frame_debug=debug,
            ))

        return resultados


# ──────────────────────────────────────────────────────────────
# TESTE
# ──────────────────────────────────────────────────────────────

def _criar_imagem_teste(resolucao=(1280, 720)) -> np.ndarray:
    """Imagem sintética: fundo cinza + cubo vermelho (imita o render Blender)."""
    w, h   = resolucao
    imagem = np.full((h, w, 3), 110, dtype=np.uint8)

    # Cubo vermelho — BGR(40, 40, 180)  →  HSV ≈ (0, 200, 180)
    cx, cy = int(w * 0.77), int(h * 0.26)
    m = 60
    cv2.rectangle(imagem, (cx - m, cy - m), (cx + m, cy + m), (40, 40, 180), -1)
    # Sombra lateral
    sombra = np.array([
        (cx - m - 8, cy + m - 10), (cx + m + 4, cy + m - 10),
        (cx + m + 4, cy + m + 16), (cx - m - 8, cy + m + 16)
    ], dtype=np.int32)
    overlay = imagem.copy()
    cv2.fillPoly(overlay, [sombra], (70, 70, 70))
    cv2.addWeighted(overlay, 0.5, imagem, 0.5, 0, imagem)
    return imagem


if __name__ == "__main__":
    print("=" * 60)
    print("  TESTE — WorkbenchVisionSetup")
    print("=" * 60)

    imagem = _criar_imagem_teste()

    for modo in ModoCamera:
        # sistema   = WorkbenchVisionSetup(modo=modo, cor=FaixaHSV.para_vermelho())
        # resultado = sistema.process_frame(imagem)

        sistema = WorkbenchVisionSetup(modo=ModoCamera.TOPDOWN, cor=FaixaHSV.para_vermelho())          
        resultado = sistema.process_frame(cv2.imread('wbtest.png'))
        print(resultado)   


        ok = "✓" if resultado.encontrado else "✗"
        if resultado.encontrado:
            print(f"  [{ok}] {modo.value:12s}  "
                  f"X={resultado.x_mm:7.1f}mm  Y={resultado.y_mm:7.1f}mm  "
                  f"area={resultado.area_pixels:.0f}px²  "
                  f"ang={resultado.angulo_graus:.1f}°")
        else:
            print(f"  [{ok}] {modo.value:12s}  Não encontrado — "
                  "ajuste PONTOS_IMAGEM_PX para sua resolução.")

    print()
    print("Para usar com seu render:")
    print("  sistema = WorkbenchVisionSetup(modo=ModoCamera.TOPDOWN,")
    print("                                 cor=FaixaHSV.para_vermelho())")
    print("  r = sistema.process_frame(cv2.imread('render.png'))")
    print("  print(r)")
    print()
    print("Para calibrar o HSV interativamente:")
    print("  WorkbenchVisionSetup.calibrar_hsv(cv2.imread('render.png'))")