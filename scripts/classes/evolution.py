import neat
import os
from worker import Worker

class Evolution:
    
    game_name : str 
    phase_name : str
    config_file : str
    checkpoint_path : str
    best_path : str
    best_name : str
    def __init__(self,
                 game_name : str, 
                 phase_name : str,
                 config_file : str,
                 checkpoint_path : str,
                 best_path : str,
                 best_name : str
                 ) -> None:
        
        if os.path.exists(f"{best_path}/{best_name}.pkl"):
            raise Exception(f"File already exists {best_path}/{best_name}.pkl")
        
        self.game_name = game_name
        self.phase_name = phase_name                 
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.best_name = best_name
    
    def eval_population(self, genomes, config_file):
        for genome_id, genome in genomes:
            worker = Worker(
                    game_name= self.game_name, 
                    phase_name= self.phase_name,
                    genome = genome,
                    config_file_path= config_file
                )