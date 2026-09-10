from dino import BasicEnv
env = BasicEnv()

observation, info = env.reset() 

def heuristic_action(obs):
    dino_y = obs[0]
    obstacle_distance = obs[1]
    obstacle_length = obs[2]
    obstacle_y = obs[3]

    if obstacle_distance == 1 and obstacle_y != 1:
        return 1

    return 0

total_reward = 0
reward = 0
done = False
episode = 0
while episode < 200 :
    action = heuristic_action(observation)
    print("Observation",observation)
    print("Action chosen",action)
    observation, reward, done, truncated, _info = env.step(action)
    # print(observation,reward,done)
    total_reward += reward
    # print(observation,total_reward,done)
    episode += 1
print("Total reward",total_reward)