# GA-Retro-Mario
Repositório dedicado ao desenvolvimento de um agente inteligente para resolver a fase YoshiIsland do Super Mario World 2.

Projeto dedicado à disciplina _Inteligência Artificial_, da _Universidade Federal do ABC_.
# Modelo utilizado: Neat
## Instalação

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
Execute a seguinte célula após o download e cópia do jogo

```sh
cp rom.sfc ~/anaconda3/envs/<ENV_NAME>/lib/python3.7/site-packages/retro/data/stable/SuperMarioWorld-Snes/
```

### Execute o script de teste
```sh
python scripts/random_agent.py
```
