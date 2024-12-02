'''
Script de teste default para verificar funcionamento do enviroment e do jogo
'''
import retro
def main():
    env = retro.make(game='SuperMarioWorld2-Snes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()
if __name__ == "__main__":
    main()