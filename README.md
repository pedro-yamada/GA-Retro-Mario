# GA-Retro-Mario
Repositório dedicado ao desenvolvimento de um agente inteligente para resolver a fase YoshiIsland do Super Mario World 2.

Projeto dedicado à disciplina _Inteligência Artificial_, da _Universidade Federal do ABC_.
## Modelo utilizado: Neat

# Instalação
## Linux

Primeiramente, estamos utilizando o Python na versão `3.7.16`

### Utilizando Conda
```sh
conda create -n <ENV_NAME> python=3.7.16
conda activate <ENV_NAME>
```

```sh
pip install -r requirements.txt
```

### Copie o arquivo do jogo para o diretório
Execute a seguinte célula após o download e cópia do jogo (no local do diretório):

```sh
cp rom.sfc ~/anaconda3/envs/<ENV_NAME>/lib/python3.7/site-packages/retro/data/stable/SuperMarioWorld-Snes/
```

### Copie o arquivo de dados do jogo para o diretório
Execute a seguinte celula no terminal (no local do diretório):

```sh
cp data.json ~/anaconda3/envs/<ENV_NAME>/lib/python3.7/site-packages/retro/data/stable/SuperMarioWorld-Snes/
```

### Execute o script de teste
```sh
python scripts/random_agent.py
```

# Rodando o GA
script modelo em: `scripts/train.py`

## Passo 1: Modifique os parâmetros no arquivo de configurações
Arquivo modelo: `scripts/config-feedfoward.txt`

## Passo 2: Instancie a classe de treino
```py
evolution = Evolution(
    game_name = 'SuperMarioWorld-Snes',
    phase_name = 'YoshiIsland2.state',
    config_file= 'config-feedfoward.txt',
    checkpoint_path= 'checkpoints',
    best_path = 'best_models',
    best_name = 'best_individual',
    render = True
)
```
sendo: 
* `game_name`: nome do jogo (padrão `retro`)
* `phase_name`: nome da fase jogada
* `config_file`: caminho para txt de configuracoes do neat
* `checkpoint_path`: caminho para pasta para salvar os checkpoints
* `best_path`: caminho para salvar melhor individuo
* `best_name`: nome do melhor individuo
* `render = True`: Caso queira acompanhar a evolução (apenas run single thread)

## Passo 3: Execute a classe de treino
```py
evolution.run(
    num_generations = 20,
    checkpoint_interval = 10,
    restore_checkpoint = None
)
```

sendo:
* `num_generations`: total de gerações até salvar o melhor individuo
* `checkpoint interval`: intervalo entre gerações para salvar o checkpoint
* `restore checkpont`: Caminho do checkpoint utilizado para continuar o treino, caso `None` treinamos do zero

# Carregando modelo para jogar
script modelo em: `scripts/run_winner.py`

## Passo 1: Carregue o genoma com o pickle:
```py
import pickle
import neat

with open(winner_path, "rb") as f:
    winner = pickle.load(f)
```

## Passo 2: Crie as configuracoes do NEAT:
```py
config_neat = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_file,
)
```

## Passo 3: Instancie o worker para rodar o jogo
```py
from scripts.classes.worker import Worker

worker = Worker(
    genome = winner,
    config_neat = config_neat,
    config_file_path = config_file
)
```
Sendo:
* `genome`: Configuracao de genoma do NEAT
* `config_file_path`: Caminho do arquivo de configuracoes
* `config_neat`: Configuracao do NEAT

## Passo 4: Instancie ambiente retro e execute o jogo
```py
import retro

env = retro.make(game_name, phase_name)
fitness = worker.eval_genome(env = env, render = True)
```
Sendo:
* `game_name`: Nome do jogo no `retro`
* `phase_name`: Nome da fase dentro do jogo
* `render = True` utilizado para verificar

Ao final do script, a variável `fitness` irá conter o valor da função de avaliação determinada.