from classes.evolution import Evolution

evolution_algorithm = Evolution(
    game_name = 'SuperMarioWorld-Snes',
    phase_name = 'YoshiIsland2.state',
    config_file= 'config-feedfoward.txt',
    checkpoint_path= 'checkpoints',
    best_path = 'best_models',
    best_name = 'best_individual',
    render = True
)

evolution_algorithm.run()