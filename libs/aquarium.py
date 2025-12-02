import pygame
import math

from libs.extra import *
from libs.fish import Fish
from libs.agent import Agent

class Aquarium:
    def __init__(self, radius_cm=25):
        self.radius_cm = radius_cm
        self.radius_px = cm_to_pixels(radius_cm)
        self.center_px = (cm_to_pixels(25), cm_to_pixels(25))
        
        self.fishes = pygame.sprite.Group()  # Peixes reais
        self.agent = None  # Agente controlável

    def add_fish(self, fish_id, trajectory, velocity):
        f = Fish(fish_id)
        f.load_trajectory(trajectory, velocity, self)
        self.fishes.add(f)

    def add_agent(self):
        """Adiciona um agente controlável ao aquário."""
        self.agent = Agent(npeixe=999)  # ID alto para diferenciar
        self.agent.rect.center = self.center_px  # Começa no centro
        self.agent.reset()



    def update(self, dt, events=None):
        """Atualiza peixes e agente."""
        # Atualizar peixes
        self.fishes.update()
        
        # Atualizar agente se existir
        if self.agent:
            # Passar eventos para o agente
            if events:
                for event in events:
                    if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
                        self.agent.handle_event(event)
            
            # Atualizar agente com lista de outros peixes
            other_fishes = [f for f in self.fishes]
            sn, act, ep = self.agent.update(dt, self, other_fishes)
            return ep
            


    def draw(self, screen,**kwargs):
        """Desenha aquário, peixes e agente."""
        # Desenhar aquário
        pygame.draw.circle(screen, (255, 255, 255), self.center_px, self.radius_px, 2)
        
        # Desenhar peixes reais
        for fish in self.fishes:
            fish.desenha(screen)
        
        # Desenhar agente se existir
        if self.agent:
            self.agent.desenha(screen,aquarium=self,**kwargs)
    
    def check_simulation_end(self):
        """Verifica se todos os peixes completaram sua trajetória."""
        if len(self.fishes) == 0:
            return False
            
        t = []
        for fish in self.fishes:
            if fish.max_index <= fish.current_index:
                t.append(True)
            else:
                t.append(False)
        return all(t)

    def reset_group(self):
        """Reseta o grupo de peixes para o próximo experimento."""
        self.fishes = pygame.sprite.Group()

    def collide(self, p1, p2):
        """Detecta colisões com o aquário."""
        cx, cy = self.center_px  # Centro do aquário
        r = self.radius_px       # Raio do aquário

        p1x, p1y = p1
        p2x, p2y = p2

        dx, dy = p2x - p1x, p2y - p1y
        fx, fy = p1x - cx, p1y - cy

        a = dx**2 + dy**2
        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - r**2

        delta = b**2 - 4*a*c

        if delta < 0:
            return False, None  # Sem interseção

        sqrt_delta = math.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2 * a)
        t2 = (-b + sqrt_delta) / (2 * a)

        t_values = [t for t in (t1, t2) if 0 <= t <= 1]

        if t_values:
            collisions = [(p1x + t * dx, p1y + t * dy) for t in t_values]
            closest_collision = min(collisions, key=lambda p: math.dist(p, p1))
            return True, closest_collision

        return False, None
