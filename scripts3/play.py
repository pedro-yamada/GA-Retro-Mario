import retro

# Nome do arquivo de gravação gerado durante o treinamento
movie_file = 'SuperMarioWorld-Snes-YoshiIsland2-000000.bk2'

# Carrega o arquivo de gravação
movie = retro.Movie(movie_file)
movie.step()

# Cria o ambiente com base nas informações do arquivo de gravação
env = retro.make(
    game=movie.get_game(),
    state=None,  # Não carrega um estado específico
    use_restricted_actions=retro.Actions.ALL,  # Permite todas as ações
    players=movie.players,
)

# Inicializa o ambiente com o estado salvo no arquivo de gravação
env.initial_state = movie.get_state()
env.reset()

# Reproduz a gravação
while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    # Executa o próximo passo no ambiente com as entradas capturadas
    env.step(keys)
    env.render()  # Renderiza o ambiente para visualizar o jogo
