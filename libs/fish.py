import pygame
import math

from libs.extra import *

class Fish(pygame.sprite.Sprite):
    def __init__(self, npeixe):
        super().__init__()
        self.npeixe = npeixe
        self.w = cm_to_pixels(2)  # cm
        self.h = cm_to_pixels(.3)  # cm

        self.imageo = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.ellipse(self.imageo, (255, 255, 255), [0, 0, self.w, self.h])
        self.image = self.imageo.copy()
        self.rect = self.image.get_rect()

        self.vr = 180  # graus
        self.angulo_atual = 0
        self.vx = 0
        self.vy = 0

        # Polarização
        self.px = 0
        self.py = 10
        self.mag = 2

        self.mcx, self.mcy = 0, 0

    def load_trajectory(self, trajectory, velocity, aquarium):
        """ Carrega uma série temporal de pontos (x, y) para a elipse seguir. """
        self.trajectory = trajectory
        self.velocity = velocity
        self.max_index = len(self.trajectory) - 1
        self.current_index = 0  # Começa do primeiro ponto
        self.mcx, self.mcy = aquarium.center_px
        self.rect.center = aquarium.center_px

    def update_position(self):
        """ Atualiza a posição da elipse para o ponto exato na trajetória (em cm -> convertendo para pixels). """
        if self.current_index < len(self.trajectory):
            # Obtém o próximo ponto da trajetória (em cm)
            x_cm, y_cm = self.trajectory[self.current_index]
            self.vx, self.vy = self.velocity[self.current_index]

            # Converte para pixels
            x_px = cm_to_pixels(x_cm) + self.mcx
            y_px = cm_to_pixels(y_cm) + self.mcy

            # Atualiza a posição do peixe exatamente no ponto
            self.rect.center = (x_px, y_px)

            # Avança para o próximo ponto da trajetória
            self.current_index += 1

    def update_polarization(self):
        """ Garante que o vetor de polarização tenha sempre a mesma magnitude sem mudanças bruscas """
        velocidade_magnitude = math.sqrt(self.vx ** 2 + self.vy ** 2)

        if velocidade_magnitude > 0.7:  # Define um limiar mínimo para considerar movimento
            # Interpolação suave para evitar mudanças bruscas
            alpha = 0.4  # Quanto menor, mais suave é a transição (0 < alpha <= 1)
            
            novo_px = (self.vx / velocidade_magnitude) * self.mag
            novo_py = (self.vy / velocidade_magnitude) * self.mag

            # Interpolação entre o vetor atual e o novo vetor
            self.px = (1 - alpha) * self.px + alpha * novo_px
            self.py = (1 - alpha) * self.py + alpha * novo_py
        else:
            # Mantém a polarização anterior se a velocidade for muito pequena
            pass

    def update_image_rotation(self):
        """ Rotaciona a imagem para alinhar com a direção da polarização """
        angulo_desejado = math.degrees(math.atan2(self.py, self.px))
        
        diferenca_angular = (angulo_desejado - self.angulo_atual + 180) % 360 - 180  # Garante rotação no menor caminho

        # Suaviza a rotação (evita mudanças bruscas)
        passo_rotacao = min(abs(diferenca_angular), self.vr)  # self.vr é a rotação máxima permitida por frame
        if diferenca_angular > 0:
            self.angulo_atual += passo_rotacao
        else:
            self.angulo_atual -= passo_rotacao

        # Aplica a rotação à imagem do peixe
        self.image = pygame.transform.rotate(self.imageo, -self.angulo_atual)
        self.rect = self.image.get_rect(center=self.rect.center)

    def desenha(self, screen):
        """ Desenha o peixe na tela. """
        screen.blit(self.image, self.rect)

    def update(self):
        """ Atualiza o estado do peixe. """
        self.update_position()
        self.update_polarization()
        self.update_image_rotation()

    # Método de colisão para detecção pelos sensores do agente
    def collide(self, p1, p2):
        """ Detecta colisões com o peixe (para os sensores do agente). """
        p1x, p1y = p1
        p2x, p2y = p2

        dx, dy = p2x - p1x, p2y - p1y
        fx, fy = p1x - self.rect.centerx, p1y - self.rect.centery

        a, b = self.w / 2, self.h / 2  # Semi-eixos maior e menor

        # Calcular o ângulo da elipse com base no vetor de polarização
        theta = math.atan2(self.py, self.px)

        # Aplicar rotação inversa nos pontos
        fx_rot = fx * math.cos(theta) + fy * math.sin(theta)
        fy_rot = -fx * math.sin(theta) + fy * math.cos(theta)

        dx_rot = dx * math.cos(theta) + dy * math.sin(theta)
        dy_rot = -dx * math.sin(theta) + dy * math.cos(theta)

        # Equação da elipse para colisão
        A = (dx_rot ** 2) / a ** 2 + (dy_rot ** 2) / b ** 2
        B = 2 * ((fx_rot * dx_rot) / a ** 2 + (fy_rot * dy_rot) / b ** 2)
        C = (fx_rot ** 2) / a ** 2 + (fy_rot ** 2) / b ** 2 - 1

        delta = B ** 2 - 4 * A * C

        if delta < 0:
            return False, None

        sqrt_delta = math.sqrt(delta)
        t1 = (-B - sqrt_delta) / (2 * A)
        t2 = (-B + sqrt_delta) / (2 * A)

        t = [t_val for t_val in (t1, t2) if 0 <= t_val <= 1]

        if t:
            collisions = [(p1x + tx * dx, p1y + tx * dy) for tx in t]
            dist = lambda c: math.sqrt((c[0] - p1x) ** 2 + (c[1] - p1y) ** 2)
            closest_collision = min(collisions, key=dist)
            return True, closest_collision

        return False, None
