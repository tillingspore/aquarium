import sqlite3
import pygame
import math
import time
import sys

import numpy as np
from numba import jit

from libs.extra import *

from libs.aquarium import Aquarium
#from libs.agent import Agent
#from libs.fish import Fish

TMAX = 0


class Simulation:
    def __init__(self, db_path):
        self.db_path = db_path
        self.aquarium = Aquarium()
        self.experiments = self.load_experiments()
        self.degug_keys ={
            "punish":True,
            "sensor":True,
            "dotcol":True,
            "monitr":False
        }
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)
        self.screen = pygame.display.set_mode((750, 750))
        pygame.display.set_caption("Simulação de Peixes com Agente")

    def load_experiments(self):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT numero_peixes, experimento
        FROM peixes
        ORDER BY numero_peixes ASC;
        """
        cursor.execute(query)
        experiments = [{"numero_peixes": row[0], "experimento": row[1]} for row in cursor.fetchall()]
        
        conn.close()
        return experiments

    def load_experiment_data(self, numero_peixes, experiment_id):

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for fish_id in range(numero_peixes):

            print(f"loading fish {fish_id}")
            # Carregar trajetória
            query = """
            SELECT pos_x, pos_y FROM peixes
            WHERE experimento = ? AND peixe = ? AND numero_peixes = ?
            ORDER BY tempo;
            """
            cursor.execute(query, (experiment_id, fish_id, numero_peixes))
            trajectory = cursor.fetchall()

            # Carregar velocidade
            query = """
            SELECT vel_x, vel_y FROM peixes
            WHERE experimento = ? AND peixe = ? AND numero_peixes = ?
            ORDER BY tempo;
            """
            cursor.execute(query, (experiment_id, fish_id, numero_peixes))
            velocity = cursor.fetchall()

            if trajectory:
                self.aquarium.add_fish(fish_id, trajectory, velocity)

            #print("loaded!")
        conn.close()

    def run_display(self):
        global TMAX
        clock = pygame.time.Clock()
        
        for experiment in self.experiments:
            numero_peixes = experiment["numero_peixes"]
            experiment_id = experiment["experimento"]

            #print(f"Iniciando simulação para Experimento {experiment_id} com {numero_peixes} peixes")

            caso = 0
            # No início do treino
            noise_scale = 1.0
            decay_rate = 0.995 # Reduz 0.5% a cada episódio


            while caso < 5_000:

                caso += 1


                print(f"exp-{numero_peixes}-{experiment_id}-{caso}")
            
                # Carregar dados do experimento
                self.load_experiment_data(numero_peixes, experiment_id)

                # Adicionar agente controlável
                if self.aquarium.agent == None:
                    self.aquarium.add_agent()

                #action = self.aquarium.agent.select_action(state, noise_scale=noise_scale)


                running = True
                tempo_atual = 0

                t = time.time()
                while running:
                    dt = clock.tick(120) / 1000.0  # Delta time em segundos
                    
                    # Coletar eventos
                    events = pygame.event.get()
                    
                    for event in events:
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_s:
                                self.degug_keys["sensor"] = not self.degug_keys["sensor"]
                            if event.key == pygame.K_d:
                                self.degug_keys["dotcol"] = not self.degug_keys["dotcol"]
                            if event.key == pygame.K_f:
                                self.degug_keys["punish"] = not self.degug_keys["punish"]  
                            if event.key == pygame.K_m:
                                self.degug_keys["monitr"] = not self.degug_keys["monitr"]                          

                    # Desenhar
                    self.screen.fill((0, 0, 0))
                    self.aquarium.draw(self.screen,**self.degug_keys)
                    
                    # Atualizar simulação
                    ep = self.aquarium.update(dt, events)

                    if ep:
                        break
                        
                    
                    # Verificar fim da simulação
                    if self.aquarium.check_simulation_end():
                        running = False


                    """
                    # Mostrar controles na tela
                    controls_text = [
                        "Controles do Agente:",
                        "W - Acelerar",
                        "S - Desacelerar", 
                        "A - Girar esquerda",
                        "D - Girar direita"
                    ]
                    
                    for i, text in enumerate(controls_text):
                        text_surface = self.font.render(text, True, (255, 255, 255))
                        self.screen.blit(text_surface, (10, 10 + i*20))
                    """
                    pygame.display.flip()
                    tempo_atual += dt
                    time.sleep(1/30)

                # Resetar para próximo experimento
                # No final do episódio, atualize o scale
                #noise_scale = max(0.1, noise_scale * decay_rate) # Nunca desce abaixo de 0.1

                t2 = time.time()
                tf = t2-t
                if tf > TMAX:
                    TMAX = tf
                ##print(f"duração: {tf} segundos(record:{TMAX})")
                self.aquarium.reset_group()
                self.aquarium.agent.reset()
               # self.aquarium.agent = None

        pygame.quit()
        #print("Simulação concluída para todos os experimentos.")

if __name__ == "__main__":
    while True:
        sim = Simulation("data.db")  # Passa o caminho do banco de dados
        sim.run_display()  # Executa a simulação para todos os experimentos