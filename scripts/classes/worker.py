import numpy as np
import retro
import neat
import cv2 

class Worker:
    genome : neat.genome.DefaultGenome
    config_file : str
    
    def __init__(self,
                 game_name : str,
                 phase_name : str,
                 genome : neat.genome.DefaultGenome,
                 config_file_path : str
                 ) -> None:
        self.game_name = game_name
        self.phase_name = phase_name
        self.genome = genome
        self.config_file = config_file_path
    
    def get_neural_network(self):
        return neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config_file)
    
    def eval_genome(self, render = True):
        self.env = retro.make(self.game_name, self.phase_name)
        
        self.env.reset()

        ob = self.env.reset()
        ac = self.env.action_space.sample()

        iny, inx, inc = self.env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        # Comando de criação da rede neural.
        net = self.get_neural_network()

        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        scoreTracker = 0
        coins = 0
        coinsTracker = 0
        yoshiCoins = 0
        yoshiCoinsTracker = 0
        xPosPrevious = 0
        yPosPrevious = 0
        checkpoint = False
        checkpointValue = 0
        endOfLevel = 0
        powerUps = 0
        powerUpsLast = 0
        jump = 0

        done = False

        while not done:
            if render:
                self.env.render()
                
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = ob.flatten()
            nnOutput = net.activate(imgarray)   
            
            ob, rew, done, info = self.env.step(nnOutput)        

            score = info['score']
            coins = info['coins']
            yoshiCoins = info['yoshiCoins']
            dead = info['dead']
            xPos = info['x']
            yPos = info['y']
            jump = info['jump']
            checkpointValue = info['checkpoint']
            endOfLevel = info['endOfLevel']
            powerUps = info['powerups']

            # Adicione à pontuação de fitness_current se Mario ganhar pontos.
            if score > 0:
                if score > scoreTracker:
                    fitness_current = (score * 10)
                    scoreTracker = score
            
            # Adicione à pontuação de fitness_current se Mario conseguir mais moedas.
            if coins > 0:
                if coins > coinsTracker:
                    fitness_current += (coins - coinsTracker)
                    coinsTracker = coins
        
            # Adicione à pontuação de fitness_current se Mario conseguir mais moedas Yoshi.
            if yoshiCoins > 0:
                if yoshiCoins > yoshiCoinsTracker:
                    fitness_current += (yoshiCoins - yoshiCoinsTracker) * 10
                    yoshiCoinsTracker = yoshiCoins

            # À medida que Mario se move para a direita, recompense-o ligeiramente.
            if xPos > xPosPrevious:
                if jump > 0:
                    fitness_current += 10
                fitness_current += (xPos / 100)
                xPosPrevious = xPos
                counter = 0

            # Se Mario estiver parado ou andando para trás, penalize-o levemente.
            else: 
                counter += 1
                fitness_current -= 0.1                     
            
            # Dê um leve prêmio a Mario por subir mais alto na posição y.
            if yPos < yPosPrevious:
                fitness_current += 10
                yPosPrevious = yPos
            elif yPos < yPosPrevious:
                yPosPrevious = yPos

            # Se Mario perder um power-up, puna-o com 1000 pontos.
            if powerUps == 0:
                if powerUpsLast == 1:
                    fitness_current -= 500
                    print("Lost Upgrade")

            # Se os powerups forem 1, Mario ganhou um cogumelo... recompense-o por ficar com ele.
            elif powerUps == 1:
                if powerUpsLast == 1 or powerUpsLast == 0:
                    fitness_current += 0.025       
                elif powerUpsLast == 2: 
                    fitness_current -= 500
                    print("Lost Upgrade")
                                
            # Se Mario chegar ao checkpoint, dê a ele um bônus.         
            if checkpointValue == 1 and checkpoint == False:
                fitness_current += 20000
                checkpoint = True
        
            # Se Mario chegar ao final do nível, conceda-lhe a maior pontuação.
            if endOfLevel == 1:
                fitness_current += 1000000
                done = True

            # Se Mario estiver parado ou retrocedendo por 450 quadros, encerre sua tentativa.
            if counter == 450:
                fitness_current -= 125
                done = True                

            # Se Mario morrer, penalize-o e siga em frente.
            if dead == 0:
                fitness_current -= 100
                done = True 

        return fitness_current