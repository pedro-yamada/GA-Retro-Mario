## Libs 
from classes.worker import Worker
import pickle 
import neat
import retro 

## Configuracoes
winner_path = 'best_models/best_individual.pkl'
game_name = 'SuperMarioWorld-Snes'
phase_name = 'YoshiIsland2.state'
config_file = 'config-feedfoward.txt'

## Rodando
with open(winner_path, "rb") as f:
    winner = pickle.load(f)

config_neat = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_file,
)

worker = Worker(
    genome = winner,
    config_neat = config_neat,
    config_file_path = config_file
)

env = retro.make(game_name, phase_name)

fitness = worker.eval_genome(env = env, render = True)
print("Winner fitness: ", fitness)