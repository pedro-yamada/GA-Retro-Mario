import neat
import os
import pickle
import retro

from classes.worker import Worker

'''
Classe para rodar algoritmo genetico
'''
class Evolution:
    
    game_name : str 
    phase_name : str
    config_file : str
    checkpoint_path : str
    best_path : str
    best_name : str
    env : retro.RetroEnv
    
    def __init__(self,
                 game_name : str, 
                 phase_name : str,
                 config_file : str,
                 checkpoint_path : str,
                 best_path : str,
                 best_name : str,
                 render = True
                 ) -> None:
        '''
        Construtor da evolucao
        * game_name -> nome do jogo .rom
        * phase_name -> nome da fase jogada
        * config_file -> caminho para txt de configuracoes do neat
        * checkpoint_path -> caminho para pasta para salvar os checkpoints
        * best_path -> caminho para salvar melhor individuo
        * best_name -> nome do melhor individuo
        * render = True -> Caso queira acompanhar a evolução (apenas run single thread)
        '''
        if os.path.exists(f"{best_path}/{best_name}.pkl"):
            raise Exception(f"File already exists {best_path}/{best_name}.pkl\nUse a different 'best_name'!")
        os.makedirs(f'{best_path}', exist_ok= True)
        
        self.game_name = game_name
        self.phase_name = phase_name                 
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.best_name = best_name
        self.render = render
        self.env = retro.make(self.game_name, self.phase_name)
    
    def eval_population(self, genomes, config_file):
        '''
        Funcao utilizada para avaliar um conjunto de genomas
        * genomes : lista de neat genomes
        * config_file : caminho para txt de configuracao        
        '''
        for genome_id, genome in genomes:
            worker = Worker(
                    genome = genome,
                    config_file_path= config_file
                )
            fitness = worker.eval_genome(env = self.env, render = self.render)
            print(genome_id, fitness)
    
    def run(self, 
            num_generations : int = 20,
            checkpoint_interval : int = 10,
            restore_checkpoint : str = None):
        '''Roda em uma unica thread o GA
        * num_generations -> total de gerações até salvar o melhor individuo
        * checkpoint interval -> intervalo entre gerações para salvar o checkpoint
        * restore checkpont -> Caminho do checkpoint utilizado para continuar o treino, caso None treinamos do zero
        '''    
    
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                    self.config_file)
        
        os.makedirs(self.checkpoint_path, exist_ok= True)
        
        if restore_checkpoint is not None:
            p = neat.Checkpointer.restore_checkpoint(restore_checkpoint)
        else:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     self.config_file)
            p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(checkpoint_interval, 
                                         filename_prefix = f"{self.checkpoint_path}/neat-checkpoint-"
                                    ))

        winner = p.run(self.eval_population, num_generations)

        with open(f'{self.best_path}/{self.best_name}.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)
