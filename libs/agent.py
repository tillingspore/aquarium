import pygame
import math
import os

import numpy as np

from libs.extra import *
from libs.fishAI import *

class Agent(pygame.sprite.Sprite):
    def __init__(self, npeixe, directory="./models"):
        super().__init__()
        self.npeixe = npeixe
        self.w = cm_to_pixels(2)  # cm
        self.h = cm_to_pixels(.3)  # cm
        self.MAX_PUNISHMENT_VALUE = -1000.0

        self.imageo = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.ellipse(self.imageo, (255, 255, 0), [0, 0, self.w, self.h])
        self.image = self.imageo.copy()
        self.rect = self.image.get_rect()
        #print(self.rect)
        # Controles de movimento - AUMENTADOS para resposta mais rápida
        self.max_speed = 13     # cm/s (aumentado)
        self.acceleration = 13  # cm/s² (aumentado)
        self.rotation_speed = 540  # Graus por segundo (aumentado)
        self.monitoring = False

        # Estado de movimento
        self.vx = 0
        self.vy = 0
        self.v = 0
        self.angulo_atual = 0
        #print(self.angulo_atual)

        # Polarização (vetor de direção)
        self.polarization_magnitude = 2
        self.px = 0
        self.py = self.polarization_magnitude

        # Sensores (apenas no agente)
        self.sensors = {
            "front": {
                "N": 128,  # Número de sensores
                "sector": 180,  # graus
                "range": cm_to_pixels(55),  # cm
                "sensors": []
            }
        }

        # Controles ativos
        self.keys = {
            pygame.K_w: False,  # Acelerar
            pygame.K_s: False,  # Desacelerar
            pygame.K_a: False,  # Girar esquerda
            pygame.K_d: False   # Girar direita
        }

        # --- Lógica de Save/Load PADRONIZADA ---
        self.model_dir = directory
        # Nome do arquivo base padronizado: ddpg_agente_0
        self.model_filename = f"ddpg_agente_{self.npeixe}" 
        os.makedirs(self.model_dir, exist_ok=True) 
        
        # Rede neural DDPG
        self.ddpg = DDPG(state_dim=130, action_dim=2)
        self.rb = ReplayBuffer(max_size=500_000)
        
        # Tenta carregar os pesos existentes
        if self._load_model():
            print(f"Pesos da Rede Neural carregados para o agente {self.npeixe}.")
        else:
            print(f"Agente {self.npeixe} iniciado com pesos aleatórios.")

        # Modo de controle
        self.use_neural_network = True

    def _get_model_path(self):
        """Retorna o caminho completo e prefixo padronizado (ex: ./models/ddpg_agente_0)."""
        return os.path.join(self.model_dir, self.model_filename)

    def _load_model(self):
        """Tenta carregar os parâmetros da rede neural E o Replay Buffer."""
        model_path = self._get_model_path()
        
        # 1. Carregar Rede Neural (Ator e Crítico)
        weights_loaded = False
        if os.path.exists(f"{model_path}_actor.pt") and os.path.exists(f"{model_path}_critic.pt"):
            self.ddpg.load(model_path)
            weights_loaded = True

        # 2. Carregar Replay Buffer
        
        rb_file = f"{model_path}_rb.pkl"
        rb_loaded = self.rb.load(rb_file)
            
        # Retorna True se pelo menos os pesos foram carregados.
        return weights_loaded

    def save_model(self):
        """
        Salva os parâmetros da rede neural E o Replay Buffer.
        Chamado no momento da punição (no método update).
        """
        model_path = self._get_model_path()
        
        # 1. Salvar Rede Neural (Ator e Crítico)
        self.ddpg.save(model_path)
        
        # 2. Salvar Replay Buffer
        rb_file = f"{model_path}_rb.pkl"
        self.rb.save(rb_file)
        
        #print(f"💾 Agente {self.npeixe}: Pesos e Replay Buffer (Tamanho {len(self.rb)}) salvos.")


    def handle_event(self, event):
        """Captura eventos de teclado para controle do agente."""
        if event.type == pygame.KEYDOWN:
            if event.key in self.keys:
                self.keys[event.key] = True
            # Tecla para alternar entre modos
            elif event.key == pygame.K_n:
                self.use_neural_network = not self.use_neural_network
                print(f"Modo rede neural: {self.use_neural_network}")
        elif event.type == pygame.KEYUP:
            if event.key in self.keys:
                self.keys[event.key] = False

    def get_state_from_sensors(self):
        """Converte as leituras dos sensores em estado para a rede neural."""
        sensor_type = "front"
        sensor_info = self.sensors[sensor_type]
        
        if not sensor_info["sensors"]:
            return np.zeros(130)
        
        state = []
        origin = np.array(self.rect.center)
        
        for sensor_point in sensor_info["sensors"]:
            # Calcular distância normalizada do sensor
            end_point = np.array(sensor_point)
            distance = np.linalg.norm(end_point - origin)
            normalized_distance = distance / sensor_info["range"]  # Normalizar para [0, 1]
            state.append(normalized_distance)
        
        # Garantir que o estado tenha exatamente 128 dimensões
        if len(state) < 130:
            # Preencher com zeros se necessário
            state.extend([0] * (130 - len(state)))
        elif len(state) > 130:
            # Reduzir se necessário (pegar amostras equidistantes)
            indices = np.linspace(0, len(state)-1, 130, dtype=int)
            state = [state[i] for i in indices]
            state.insert(0,self.angulo_atual)
            state.insert(0,self.v)
        
        return np.array(state)

    
    def update_movement_nn(self, dt):

        # Obter estado dos sensores
        state = self.get_state_from_sensors()
        
        # Obter ação da rede neural
        action = self.ddpg.select_action(state, add_noise=True)
        
        # Interpretar a ação da rede neural
        # action[0]: aceleração (-1 a 1)
        # action[1]: rotação (-1 a 1)  

        # Rotação

         # Rotação baseada na ação neural
        rotation_action = action[1]  # -1 (esquerda) a 1 (direita)
        rotation_step = self.rotation_speed * dt * rotation_action # Aplica o valor contínuo
        self.angulo_atual += rotation_step
        #print(self.angulo_atual)
        # Aceleração baseada na ação neural
        acceleration_action = action[0]  # -1 (ré) a 1 (frente)
        acceleration_step = self.acceleration * dt * acceleration_action # Aplica o valor contínuo
        
        angle_rad = math.radians(self.angulo_atual)
        self.vx += math.cos(angle_rad) * acceleration_step
        self.vy += math.sin(angle_rad) * acceleration_step


        # Recalcular a direção da velocidade baseada no ângulo atual
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        self.v = current_speed
        
        if current_speed > 0:
            angle_rad = math.radians(self.angulo_atual)
            self.vx = math.cos(angle_rad) * current_speed
            self.vy = math.sin(angle_rad) * current_speed

        # Limitar velocidade máxima
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale

        # Atualizar posição
        self.rect.x += self.vx * dt * PIXELS_PER_CM
        self.rect.y += self.vy * dt * PIXELS_PER_CM

        # Atualizar vetor de polarização
        angle_rad = math.radians(self.angulo_atual)
        self.px = math.cos(angle_rad) * self.polarization_magnitude
        self.py = math.sin(angle_rad) * self.polarization_magnitude

        return state, action


    def update_movement(self, dt):
        """Seleciona o método de movimento baseado no modo atual."""
        return self.update_movement_nn(dt)


    def update_image_rotation(self):
        """Atualiza a rotação da imagem do agente."""
        self.image = pygame.transform.rotate(self.imageo, -self.angulo_atual)
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center

    def generate_sensors(self, aquarium, other_fishes):
        """Gera sensores para o agente, detectando colisões com peixes e aquário."""
        sensor_type = "front"
        sensor_info = self.sensors[sensor_type]
        N = sensor_info["N"]
        sector = math.radians(sensor_info["sector"])
        max_range_px = sensor_info["range"]
        origin = self.rect.center

        base_angle = math.atan2(self.py, self.px)
        angles = np.linspace(base_angle - sector/2, base_angle + sector/2, N)

        sensor_points = []
        for angle in angles:
            end_x = origin[0] + max_range_px * math.cos(angle)
            end_y = origin[1] + max_range_px * math.sin(angle)
            end_point = (end_x, end_y)

            final_point = self.general_collide(origin, end_point, aquarium, other_fishes)
            sensor_points.append(final_point)

        self.sensors[sensor_type]["sensors"] = sensor_points

    def general_collide(self, p1, p2, aquarium, other_fishes):
        """Detecta colisões com o aquário e outros peixes."""
        closest_collision = None
        min_dist = float('inf')
        
        for fish in other_fishes:
            if fish == self:
                continue
                
            collides, collision_point = fish.collide(p1, p2)
            if collides:
                dist = math.dist(p1, collision_point)
                if dist < min_dist:
                    closest_collision = collision_point
                    min_dist = dist

        if closest_collision is None:
            collides, collision_point = aquarium.collide(p1, p2)
            if collides:
                closest_collision = collision_point

        return closest_collision if closest_collision else p2
    
    def reset(self):
        self.vx = 0
        self.vy = 0
        self.v = 0
        self.angulo_atual = 0
        #print("center:",self.rect.center)
        self.rect.center = (cm_to_pixels(25),cm_to_pixels(25))

    def update(self, dt, aquarium, other_fishes):
        """
        Atualiza o agente, gerencia o ciclo de RL (armazenamento e treino) 
        e aplica a punição de fronteira.
        """
        
        state_old = self.get_state_from_sensors()
        
        action_tuple = self.update_movement(dt)

        # Se estiver no modo rede neural, action_tuple será uma tupla (state, action)
        if action_tuple is not None:
            # Desempacota a tupla para obter o estado (st) e a AÇÃO (action) correta
            st, action = action_tuple
            self.last_action = action # Armazena a ação correta para visualização
        else:
            action = None # Se for modo teclado, a ação é None.
        self.update_image_rotation()
        self.generate_sensors(aquarium, other_fishes)
        
        # 3. Obter o PRÓXIMO ESTADO (S') após o movimento
        state_new = self.get_state_from_sensors()

        # O ciclo de RL só ocorre se estivermos no modo Rede Neural E se a ação foi determinada (não nula)
        if self.use_neural_network and action is not None:
            
            # Recompensa base (ex: sobrevivência)
            reward_original = 0.1
            done_original = False 
            
            # 4. CHECAGEM DE PUNIÇÃO DE SEGURANÇA
            is_punished = self.punish(aquarium) # Usa o método de checagem
            
            if is_punished:
                # Se violou o limite:
                reward_final = -100                       # Punição forte
                done_final = True                        # Finaliza o episódio
                #print("VIOLAÇÃO! Punição aplicada e episódio encerrado.")
                self.save_model()
                self.reset()
 
            else:
                # Se seguro:
                reward_final = reward_original
                done_final = done_original
            
            # 5. ARMAZENAMENTO DA EXPERIÊNCIA CORRIGIDA (S, A, R_final, S', Done_final)
            self.rb.push(state_old, action, reward_final, state_new, done_final)

            # 6. EXECUTAR TREINAMENTO (Treinamento off-policy do DDPG)
            # Você precisa de um tamanho mínimo de buffer antes de começar a treinar
            MIN_SAMPLES_FOR_TRAIN = 2000
            BATCH_SIZE = 512
            TRAIN_ITERATIONS = 1
            
            if len(self.rb) > MIN_SAMPLES_FOR_TRAIN:
                #print("treinando")
                for _ in range(TRAIN_ITERATIONS):
                    batch = self.rb.sample(BATCH_SIZE)
                    self.ddpg.update_parameters(batch)
                
            # Retorna a flag de término para que o loop principal da simulação possa encerrar o episódio
            return state_new, action, done_final
        
        return state_new, action, False # Retorna False para 'done' se estiver no modo teclado

    """def desenha(self, screen):
       Desenha o agente e seus componentes
        # Desenhar o agente
        screen.blit(self.image, self.rect)

        # Desenhar sensores
        origin = self.rect.center
        for sensor_type, sensor_info in self.sensors.items():
            for point in sensor_info["sensors"]:
                #pygame.draw.line(screen, (0, 255, 0), origin, point, 1)
                pygame.draw.circle(screen, (255, 0, 0), point, 3)

        # Desenhar vetor de polarização
        start = self.rect.center
        end = (start[0] + cm_to_pixels(self.px)*3, start[1] + cm_to_pixels(self.py)*3)
        pygame.draw.line(screen, (255, 255, 0), start, end, 2)
        
        # --- Visualização da Rede Neural (DDPG) ---
        if self.use_neural_network and hasattr(self, 'last_action'):
            
            # Posição base para o texto
            text_x = self.rect.right + 10
            text_y = self.rect.top
            
            # 1. Configurar a fonte (pode ser feito no __init__ para eficiência)
            font = pygame.font.Font(None, 24) 
            
            action = self.last_action
            
            # 2. Interpretar a Ação (Decisão de Movimento)
            
            # Rotação
            if action[1] > 0:
                rot_text = "GIRAR ESQUERDA"
                rot_color = (0, 255, 0) # Verde
            elif action[1] < 0:
                rot_text = "GIRAR DIREITA"
                rot_color = (255, 165, 0) # Laranja
            else:
                rot_text = "SEM ROTAÇÃO"
               rot_color = (150, 150, 150) # Cinza

            # Aceleração
            if action[0] > 0:
                accel_text = "ACELERAR"
                accel_color = (0, 255, 0) # Verde
            elif action[0] < 0:
                 accel_text = "FREAR/RÉ"
                accel_color = (255, 0, 0) # Vermelho
            else:
                accel_text = "MANTER VELOCIDADE"
                accel_color = (150, 150, 150) # Cinza
            
            # 3. Desenhar os textos (Saída Bruta)
            
            # Texto Saída Bruta
            action_raw_text = f"Ação Bruta: [{action[0]:.2f}, {action[1]:.2f}]"
            text_surface = font.render(action_raw_text, True, (255, 255, 255))
            screen.blit(text_surface, (text_x, text_y))
            text_y += 20

            # 4. Desenhar os textos (Decisão Interpretada)
            
            # Texto Rotação
            text_surface_rot = font.render(f"Rotação: {rot_text}", True, rot_color)
            screen.blit(text_surface_rot, (text_x, text_y))
            text_y += 20
            
            # Texto Aceleração
            text_surface_accel = font.render(f"Aceleração: {accel_text}", True, accel_color)
            screen.blit(text_surface_accel, (text_x, text_y))
            
            # Desenha o modo atual (já estava na sua função, mas comentei)
            mode_text = "Rede Neural (DDPG)" if self.use_neural_network else "Teclado"
            text_surface = font.render(f"Modo: {mode_text}", True, (255, 255, 255))
            screen.blit(text_surface, (self.rect.x, self.rect.y - 30))"""

    def desenha(self, screen, aquarium=None, punish=False,sensor=False,dotcol=False,monitr=False):
        """Desenha o agente e seus componentes.
        
        Args:
            screen: Surface do Pygame para desenhar
            aquarium: Objeto do aquário (opcional, necessário para debug de colisão)
            debug_collision: Se True, desenha informações de colisão
        """
        # Desenhar o agente
        screen.blit(self.image, self.rect)

        self.ddpg.monitoring_change(monitr)

        # Desenhar sensores
        origin = self.rect.center
        for sensor_type, sensor_info in self.sensors.items():
            for point in sensor_info["sensors"]:
                if sensor:
                    pygame.draw.line(screen, (0, 255, 0), origin, point, 1)
                if dotcol:
                    pygame.draw.circle(screen, (255, 0, 0), point, 3)

        # Desenhar vetor de polarização
        start = self.rect.center
        end = (start[0] + cm_to_pixels(self.px)*3, start[1] + cm_to_pixels(self.py)*3)
        pygame.draw.line(screen, (255, 255, 0), start, end, 2)
        
        # Desenhar área de colisão se debug estiver ativado
        if punish and aquarium is not None:
            self.draw_collision_area(screen, aquarium)
        
        # --- Visualização da Rede Neural (DDPG) ---
        if self.use_neural_network and hasattr(self, 'last_action'):
            
            # Posição base para o texto
            text_x = self.rect.right + 10
            text_y = self.rect.top
            
            # 1. Configurar a fonte (pode ser feito no __init__ para eficiência)
            font = pygame.font.Font(None, 24) 
            
            action = self.last_action
            
            # 2. Interpretar a Ação (Decisão de Movimento)
            
            # Rotação
            if action[1] > 0:
                rot_text = "GIRAR ESQUERDA"
                rot_color = (0, 255, 0) # Verde
            elif action[1] < 0:
                rot_text = "GIRAR DIREITA"
                rot_color = (255, 165, 0) # Laranja
            else:
                rot_text = "SEM ROTAÇÃO"
                rot_color = (150, 150, 150) # Cinza

            # Aceleração
            if action[0] > 0:
                accel_text = "ACELERAR"
                accel_color = (0, 255, 0) # Verde
            elif action[0] < 0:
                accel_text = "FREAR/RÉ"
                accel_color = (255, 0, 0) # Vermelho
            else:
                accel_text = "MANTER VELOCIDADE"
                accel_color = (150, 150, 150) # Cinza
            
            # 3. Desenhar os textos (Saída Bruta)
            
            # Texto Saída Bruta
            action_raw_text = f"Ação Bruta: [{action[0]:.2f}, {action[1]:.2f}]"
            text_surface = font.render(action_raw_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 705))
            text_y += 20

            # 4. Desenhar os textos (Decisão Interpretada)
            
            # Texto Rotação
            text_surface_rot = font.render(f"Rotação: {rot_text}", True, rot_color)
            screen.blit(text_surface_rot, (10, 730))
            text_y += 20
            
            # Texto Aceleração
            text_surface_accel = font.render(f"Aceleração: {accel_text}", True, accel_color)
            screen.blit(text_surface_accel, (10, 750))
            
            # Desenha o modo atual (já estava na sua função, mas comentei)
            mode_text = "Rede Neural (DDPG)" if self.use_neural_network else "Teclado"
            text_surface = font.render(f"Modo: {mode_text}", True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))

    def draw_collision_area(self, screen, aquarium):
        """Desenha a área de colisão do agente (elipse) e sua relação com o aquário."""
        
        # Configurações de cores
        COLLISION_COLOR = (255, 0, 0, 100)  # Vermelho transparente
        SAFE_COLOR = (0, 255, 0, 100)       # Verde transparente
        CLOSEST_POINT_COLOR = (255, 255, 0)  # Amarelo
        NORMAL_COLOR = (0, 0, 255)          # Azul
        
        # Cria uma surface temporária para desenho com transparência
        collision_surface = pygame.Surface((self.w * 2, self.h * 2), pygame.SRCALPHA)
        
        # Centro da elipse
        center_x, center_y = self.rect.center
        a = self.w / 2  # semi-eixo maior
        b = self.h / 2  # semi-eixo menor
        theta = math.radians(self.angulo_atual)
        
        # 1. Desenhar a elipse rotacionada
        points = []
        for angle in range(0, 360, 5):  # 5° de incremento para suavidade
            rad_angle = math.radians(angle)
            
            # Ponto na elipse não rotacionada
            x_local = a * math.cos(rad_angle)
            y_local = b * math.sin(rad_angle)
            
            # Aplica rotação
            x_rot = x_local * math.cos(theta) - y_local * math.sin(theta)
            y_rot = x_local * math.sin(theta) + y_local * math.cos(theta)
            
            # Translada para a posição global
            x_global = center_x + x_rot
            y_global = center_y + y_rot
            
            points.append((x_global, y_global))
        
        # Desenha a elipse
        if len(points) > 2:
            pygame.draw.polygon(screen, COLLISION_COLOR, points, 1)
        
        # 2. Calcular e desenhar o ponto mais próximo do aquário
        cx, cy = aquarium.center_px
        dx = center_x - cx
        dy = center_y - cy
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Coordenadas no sistema rotacionado
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        
        # Encontra ponto mais próximo na elipse
        closest_local = self._find_closest_point_on_ellipse(x_rot, y_rot, a, b)
        
        # Converte de volta para coordenadas globais
        closest_x_global = center_x + (closest_local[0] * cos_theta - closest_local[1] * sin_theta)
        closest_y_global = center_y + (closest_local[0] * sin_theta + closest_local[1] * cos_theta)
        
        # Desenha o ponto mais próximo
        pygame.draw.circle(screen, CLOSEST_POINT_COLOR, (int(closest_x_global), int(closest_y_global)), 5)
        
        # 3. Desenhar linha do centro do aquário até o ponto mais próximo
        pygame.draw.line(screen, CLOSEST_POINT_COLOR, (cx, cy), (closest_x_global, closest_y_global), 2)
        
        # 4. Desenhar vetor normal no ponto mais próximo
        # O vetor normal aponta na direção do centro do aquário
        normal_length = 20
        normal_end_x = closest_x_global + (cx - closest_x_global) / aquarium.radius_px * normal_length
        normal_end_y = closest_y_global + (cy - closest_y_global) / aquarium.radius_px * normal_length
        
        pygame.draw.line(screen, NORMAL_COLOR, 
                        (closest_x_global, closest_y_global),
                        (normal_end_x, normal_end_y), 2)
        
        # 5. Desenhar círculo de segurança (bounding circle)
        circumradius = math.sqrt(a**2 + b**2)
        pygame.draw.circle(screen, (255, 255, 0, 50), (center_x, center_y), int(circumradius), 1)
        
        # 6. Desenhar informações textuais
        font = pygame.font.Font(None, 24)
        
        # Verifica colisão atual
        is_colliding = self._ellipse_circle_collision(aquarium)
        
        status_text = f"Colisão: {'N' if is_colliding else 'S'}"
        status_color = (255, 0, 0) if is_colliding else (0, 255, 0)
        
        text_surface = font.render(status_text, True, status_color)
        screen.blit(text_surface, (10, 30))
        
        # Distância até a borda
        distance_to_edge = math.sqrt((closest_x_global - cx)**2 + (closest_y_global - cy)**2)
        distance_text = f"Dist: {distance_to_edge:.1f}/{aquarium.radius_px:.1f}"
        text_surface = font.render(distance_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 45))
        
        # Ângulo atual
        angle_text = f"Ângulo: {self.angulo_atual:.1f}°"
        text_surface = font.render(angle_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 60))

    def punish(self, aquarium):
        """Verifica se o agente (elipse rotacionada) colidiu com a borda do aquário (círculo).
        Retorna True se deve ser punido (colidiu/saiu), False caso contrário."""

        if np.abs(self.angulo_atual) >= 1800:
            return True
        
        # 1. Verificação rápida usando bounding circle - O(1)
        center_x, center_y = self.rect.center
        cx, cy = aquarium.center_px
        
        # Distância do centro do agente ao centro do aquário
        distance_to_center = math.sqrt((center_x - cx)**2 + (center_y - cy)**2)
        
        # Raio do círculo circunscrito ao agente (maior distância possível do centro até a borda da elipse)
        circumradius = math.sqrt((self.w/2)**2 + (self.h/2)**2)
        
        # Verificação otimista: se estiver claramente dentro
        if distance_to_center + circumradius < aquarium.radius_px:
            return False
        
        # Verificação pessimista: se estiver claramente fora
        if distance_to_center - circumradius > aquarium.radius_px:
            return True
        
        # 2. Verificação precisa usando geometria analítica - O(1)
        return self._ellipse_circle_collision(aquarium)

    def _ellipse_circle_collision(self, aquarium):
        """Detecção precisa de colisão entre elipse rotacionada e círculo."""
        # Centro do aquário e do agente
        cx, cy = aquarium.center_px
        ex, ey = self.rect.center
        
        # Semi-eixos da elipse
        a = self.w / 2  # semi-eixo maior
        b = self.h / 2  # semi-eixo menor
        
        # Ângulo de rotação da elipse em radianos
        theta = math.radians(self.angulo_atual)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Translada e rotaciona o sistema de coordenadas para colocar a elipse na origem e alinhada com os eixos
        dx = ex - cx
        dy = ey - cy
        
        # Aplica a rotação inversa para alinhar a elipse com os eixos coordenados
        x_rot = dx * cos_theta + dy * sin_theta
        y_rot = -dx * sin_theta + dy * cos_theta
        
        # Equação da elipse no sistema transladado e rotacionado: (x_rot/a)^2 + (y_rot/b)^2 = 1
        # Precisamos encontrar o ponto na elipse mais próximo do centro do círculo (que está na origem neste sistema)
        
        # Se o ponto (x_rot, y_rot) está dentro da elipse, então o círculo contém o centro da elipse
        ellipse_eq = (x_rot/a)**2 + (y_rot/b)**2
        if ellipse_eq <= 1:
            # Centro do círculo está dentro da elipse - colisão certamente ocorre
            return True
        
        # Encontra o ponto na elipse mais próximo da origem (centro do círculo)
        # Usa o método do gradiente com busca binária para precisão
        closest_point = self._find_closest_point_on_ellipse(x_rot, y_rot, a, b)
        
        # Calcula a distância do ponto mais próximo até a origem
        distance_to_closest = math.sqrt(closest_point[0]**2 + closest_point[1]**2)
        
        # Se a distância for menor que o raio do aquário, há colisão
        return distance_to_closest < aquarium.radius_px

    def _find_closest_point_on_ellipse(self, x, y, a, b, tolerance=1e-8, max_iterations=50):
        """Encontra o ponto na elipse (x/a)^2 + (y/b)^2 = 1 mais próximo do ponto (x,y) dado.
        
        Usa o método de Newton com busca binária para garantir convergência.
        Complexidade: O(max_iterations) = O(1) para max_iterations fixo.
        """
        # Inicializa o parâmetro t (ângulo paramétrico)
        if x == 0 and y == 0:
            return (a, 0)  # Caso especial: ponto na origem
        
        # Ângulo inicial em direção ao ponto (x,y)
        t = math.atan2(a * y, b * x)
        
        for i in range(max_iterations):
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            
            # Ponto na elipse: (a*cos_t, b*sin_t)
            ellipse_x = a * cos_t
            ellipse_y = b * sin_t
            
            # Vetor do ponto na elipse ao ponto alvo
            dx = ellipse_x - x
            dy = ellipse_y - y
            
            # Derivadas do ponto na elipse
            dx_dt = -a * sin_t
            dy_dt = b * cos_t
            
            # Função f(t) = (P_ellipse(t) - P) · dP/dt
            f = dx * dx_dt + dy * dy_dt
            
            # Derivada f'(t)
            d2x_dt2 = -a * cos_t
            d2y_dt2 = -b * sin_t
            df_dt = (dx_dt * dx_dt + dx * d2x_dt2 + 
                    dy_dt * dy_dt + dy * d2y_dt2)
            
            # Método de Newton
            if abs(df_dt) > tolerance:
                delta_t = -f / df_dt
                t += delta_t
                
                # Garante que t permaneça no intervalo [0, 2π]
                if t < 0:
                    t += 2 * math.pi
                elif t >= 2 * math.pi:
                    t -= 2 * math.pi
            
            # Verifica convergência
            if abs(f) < tolerance:
                break
        
        # Retorna o ponto mais próximo na elipse
        return (a * math.cos(t), b * math.sin(t))



    def _calculate_rotation_punishment(self, dt):
        """Pune rotação excessiva e oscilações"""
        # Taxa de rotação atual (graus por segundo)
        current_rotation_rate = abs(self.angulo_atual - getattr(self, '_last_angle', self.angulo_atual)) / dt
        
        # Salvar ângulo para próximo cálculo
        self._last_angle = self.angulo_atual
        
        # Limites aceitáveis de rotação (graus por segundo)
        MAX_ROTATION_RATE = 180  # 180°/s
        IDEAL_ROTATION_RATE = 45  # 45°/s
        
        if current_rotation_rate > MAX_ROTATION_RATE:
            # Punição severa por rotação muito rápida
            return -2.0
        elif current_rotation_rate > IDEAL_ROTATION_RATE * 1.5:
            # Punição moderada por rotação acima do ideal
            return -0.5
        elif current_rotation_rate < IDEAL_ROTATION_RATE * 0.5:
            # Pequena recompensa por rotação suave
            return 0.1
        else:
            # Recompensa por rotação no range ideal
            return 0.2

    def _calculate_inactivity_punishment(self, dt):
        """Pune inatividade prolongada"""
        speed = math.sqrt(self.vx**2 + self.vy**2)
        
        # Limites de velocidade (cm/s)
        MIN_ACCEPTABLE_SPEED = 5   # Muito lento
        IDEAL_MIN_SPEED = 15       # Velocidade mínima ideal
        IDEAL_MAX_SPEED = 50       # Velocidade máxima ideal
        
        # Contador de inatividade
        if not hasattr(self, '_inactivity_timer'):
            self._inactivity_timer = 0
            
        if speed < MIN_ACCEPTABLE_SPEED:
            self._inactivity_timer += dt
        else:
            self._inactivity_timer = max(0, self._inactivity_timer - dt * 2)
        
        # Punição progressiva por inatividade
        if self._inactivity_timer > 10:  # 10 segundos parado
            return -2.0
        elif self._inactivity_timer > 5:  # 5 segundos parado
            return -0.5
        elif speed < IDEAL_MIN_SPEED:
            return -0.1
        elif IDEAL_MIN_SPEED <= speed <= IDEAL_MAX_SPEED:
            return 0.3  # Recompensa por velocidade adequada
        else:
            return -0.2  # Punição leve por velocidade muito alta

    def _calculate_speed_reward(self):
        """Recompensa por manter velocidade consistente"""
        speed = math.sqrt(self.vx**2 + self.vy**2)
        IDEAL_SPEED = 30  # cm/s
        
        # Recompensa baseada na proximidade da velocidade ideal
        speed_diff = abs(speed - IDEAL_SPEED)
        if speed_diff < 5:
            return 0.2
        elif speed_diff < 15:
            return 0.0
        else:
            return -0.1

    def _calculate_smoothness_punishment(self, dt):
        """Pune mudanças bruscas de direção e velocidade"""
        if not hasattr(self, '_last_vx'):
            self._last_vx, self._last_vy = self.vx, self.vy
            return 0
        
        # Calcula aceleração
        acc_x = (self.vx - self._last_vx) / dt
        acc_y = (self.vy - self._last_vy) / dt
        acceleration = math.sqrt(acc_x**2 + acc_y**2)
        
        # Salva valores para próxima iteração
        self._last_vx, self._last_vy = self.vx, self.vy
        
        MAX_ACCEPTABLE_ACCELERATION = 50  # cm/s²
        
        if acceleration > MAX_ACCEPTABLE_ACCELERATION:
            return -1.0
        elif acceleration > MAX_ACCEPTABLE_ACCELERATION * 0.7:
            return -0.3
        else:
            return 0.1  # Recompensa por movimento suave

    def _calculate_socialization_reward(self, other_fishes):
        """Recompensa por nadar próximo a outros peixes (comportamento de cardume)"""
        if not other_fishes:
            return 0
        
        min_distance = float('inf')
        center_x, center_y = self.rect.center
        
        for fish in other_fishes:
            if fish == self:
                continue
                
            fish_x, fish_y = fish.rect.center
            distance = math.sqrt((center_x - fish_x)**2 + (center_y - fish_y)**2)
            min_distance = min(min_distance, distance)
        
        # Distâncias em pixels - converter para cm
        distance_cm = pixels_to_cm(min_distance)
        
        # Comportamento social ideal: 10-30 cm de distância
        if 10 <= distance_cm <= 30:
            return 0.5  # Recompensa por distância social ideal
        elif 5 <= distance_cm < 10:
            return 0.2  # Recompensa menor por muito próximo
        elif 30 < distance_cm <= 50:
            return 0.1  # Recompensa menor por um pouco longe
        elif distance_cm < 5:
            return -0.3  # Punição por muito próximo (risco de colisão)
        else:
            return 0

    def _calculate_isolation_punishment(self, other_fishes):
        """Pune isolamento excessivo do cardume"""
        if not other_fishes:
            return 0
        
        center_x, center_y = self.rect.center
        total_distance = 0
        count = 0
        
        for fish in other_fishes:
            if fish == self:
                continue
                
            fish_x, fish_y = fish.rect.center
            distance = math.sqrt((center_x - fish_x)**2 + (center_y - fish_y)**2)
            total_distance += distance
            count += 1
        
        if count == 0:
            return 0
            
        avg_distance = total_distance / count
        avg_distance_cm = pixels_to_cm(avg_distance)
        
        # Punição por ficar muito isolado
        if avg_distance_cm > 100:  # Mais de 1m de distância média
            return -0.8
        elif avg_distance_cm > 70:
            return -0.4
        elif avg_distance_cm > 50:
            return -0.1
        else:
            return 0

        """def punish(self, aquarium):
            "Verifica se o agente colidiu ou saiu do aquário.
            Retorna True se deve ser punido (colidiu/saiu), False caso contrário."
            
            # Verificar se o centro do agente está fora do círculo do aquário
            center_x, center_y = self.rect.center
            cx, cy = aquarium.center_px
            distance_to_center = math.sqrt((center_x - cx)**2 + (center_y - cy)**2)
            
            # Considerar uma margem de segurança (raio do agente)
            agent_radius = max(self.w, self.h) / 2
            safe_distance = aquarium.radius_px - agent_radius
            
            # Se a distância for maior que o raio seguro, está fora
            if distance_to_center > safe_distance:
                return True
            
            # Verificação adicional: verificar se algum vértice da bounding box está fora
            vertices = [
                (self.rect.left, self.rect.top),
                (self.rect.right, self.rect.top),
                (self.rect.right, self.rect.bottom),
                (self.rect.left, self.rect.bottom)
            ]
            
            for vx, vy in vertices:
                vertex_distance = math.sqrt((vx - cx)**2 + (vy - cy)**2)
                if vertex_distance > aquarium.radius_px:
                    return True
            
            return False"""