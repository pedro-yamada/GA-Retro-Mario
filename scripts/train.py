from classes.evolution import Evolution

evolution_algorithm = Evolution(
    game_name = 'SuperMarioWorld-Snes',
    phase_name = 'YoshiIsland2.state',
    config_file= 'config-feedfoward.txt',
    checkpoint_path= 'checkpoints',
    best_path = 'best_models',
    best_name = 'best_individual_RUN2',
    render = True
)

evolution_algorithm.run(
    num_generations = 25,
    checkpoint_interval = 5,
    restore_checkpoint = 'checkpoints/neat-checkpoint-28' ## Caso queira usar um checkpoint, passar o caminho do checkpoint salvo
)