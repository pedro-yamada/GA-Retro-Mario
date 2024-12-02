import neat
import os
import pickle
import retro

from classes.worker import Worker

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
        
        if os.path.exists(f"{best_path}/{best_name}.pkl"):
            raise Exception(f"File already exists {best_path}/{best_name}.pkl")
        
        self.game_name = game_name
        self.phase_name = phase_name                 
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.best_name = best_name
        self.render = render
        self.env = retro.make(self.game_name, self.phase_name)
    
    def eval_population(self, genomes, config_file):

        for genome_id, genome in genomes:
            worker = Worker(
                    game_name= self.game_name, 
                    phase_name= self.phase_name,
                    genome = genome,
                    config_file_path= config_file
                )
            fitness = worker.eval_genome(env = self.env, render = self.render)
            print(genome_id, fitness)
    
    def run(self, num_generations = 20, checkpoint_interval = 10):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                    self.config_file)
        
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(checkpoint_interval))

        winner = p.run(self.eval_population, num_generations)

        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)