# the Gym environment class
import gymnasium as gym
from gymnasium import spaces
# used to randomize starting positions
import random
# used for integer datatypes
import numpy as np
# used for clearing the display
import os

#
# Global Constants
#
GAME_SIZE = 20
GAME_RATE = 1
SNAPSHOT_DURATION = 1
JUMP_DURATION = 5
MAX_OBSTACLE_LENGTH = GAME_RATE*JUMP_DURATION-1

# Actions
NOTHING = 0
JUMP = 1

class BasicEnv(gym.Env):
  def __init__(self):
    self.cumulative_reward = 0
    self.dino_position_y = 0

    # Get an obstacle
    self.nearest_obstacle_length = random.randrange(1,MAX_OBSTACLE_LENGTH+1)
    # self.nearest_obstacle_length = 1 # TESTING
    self.nearest_obstacle_distance = random.randrange(1,GAME_SIZE-self.nearest_obstacle_length-1) # x axis of the start of the obstacle
    # only bird can be in the sky, not cactus. Bird length is 1
    if self.nearest_obstacle_length == 1:
      self.nearest_obstacle_y = random.choice([0,1])
    else:
      self.nearest_obstacle_y = 0

    self.nearest_obstacle_end_x = self.nearest_obstacle_distance + self.nearest_obstacle_length

    # First consider only one obstacle
    self.next_obstacle_length = -1
    self.next_obstacle_distance = -1
    self.next_obstacle_y = -1
    self.next_obstacle_end_x = -1


    self.action_space = spaces.Discrete(2)

    self.state = np.array([
    self.dino_position_y,
    self.nearest_obstacle_distance,
    self.nearest_obstacle_length,
    self.nearest_obstacle_y,
    self.next_obstacle_distance,
    self.next_obstacle_length,
    self.next_obstacle_y
], dtype=np.float32)

    # convert the self.state python array into a numpy array
    # (This is needed since Gym expects the state to be this way)
    self.state = np.array(self.state, dtype=np.float32)

#     observation = np.array([
#     self.dino_position[1],
#     self.obstacle_start_x,
#     self.obstacle_length,
#     self.obstacle_start_y
# ], dtype=np.float32)

    self.observation_space = spaces.Box(
    low=np.array([0, 0, 1, 0, -1, -1, -1], dtype=np.float32),
    high=np.array([1, GAME_SIZE, MAX_OBSTACLE_LENGTH, 1, GAME_SIZE, MAX_OBSTACLE_LENGTH, 1], dtype=np.float32),
    dtype=np.float32
)

  def progress_game(self):
    self.nearest_obstacle_distance -= GAME_RATE
    self.nearest_obstacle_end_x -= GAME_RATE
    if(self.next_obstacle_length != -1):
      self.next_obstacle_distance -= GAME_RATE
      self.next_obstacle_end_x -= GAME_RATE

    self.first_obstacle_cleared()

  def first_obstacle_cleared(self):
     # this is for the case where the nearest obstacle is cleared and the next obstacle is not null
    if(self.nearest_obstacle_end_x < 0 and self.next_obstacle_distance <=0 and self.next_obstacle_length != -1):
      self.swap_obstacles()
      self.clear_next_obstacle()

  def swap_obstacles(self):
    self.nearest_obstacle_length = self.next_obstacle_length
    self.nearest_obstacle_distance = self.next_obstacle_distance
    self.nearest_obstacle_y = self.next_obstacle_y
    self.nearest_obstacle_end_x = self.next_obstacle_end_x

  def clear_next_obstacle(self):
      self.next_obstacle_length = -1
      self.next_obstacle_distance = -1
      self.next_obstacle_y = -1
      self.next_obstacle_end_x = -1

  def step(self, action):
    # placeholder for debugging information
    info = {}
    truncated = False
    # set default values for done, reward, and the player position
    #before taking the action
    done = False

    # If action is to do nothing, progress the obstacles
    if action == NOTHING:
      self.progress_game()
      collided = self.check_for_collision()
      if collided:
        done = True
        reward = -50
        self.cumulative_reward += reward
        print(f'Cumulative Reward: {self.cumulative_reward}')
        print('You Lost :(')
      else:
        reward = 1
        self.cumulative_reward += reward

        # check if obstacle cleared, if the obstacle was bird in the sky
        if self.nearest_obstacle_end_x < 0:
          self.swap_obstacles()
          self.clear_next_obstacle()

    elif action == JUMP:
          for i in range(JUMP_DURATION):
            self.dino_position_y = 1
            self.progress_game()
            if (i!=JUMP_DURATION-1):
              collided = self.check_for_collision()
              if collided:
                done = True
                reward = -50
                self.cumulative_reward += reward
                print(f'Cumulative Reward: {self.cumulative_reward}')
                print('You Lost :(')
                return self.state, reward, done, truncated, info
            # else:
            #   reward = 1
            #   self.cumulative_reward += reward
            if(i==JUMP_DURATION-1):
              self.dino_position_y = 0 # Dino lands in the last second
              collided = self.check_for_collision()
              if collided:
                done = True
                reward = -50
                self.cumulative_reward += reward
                print(f'Cumulative Reward: {self.cumulative_reward}')
                print('You Lost :(')
                return self.state, reward, done, truncated, info
              else:
                reward = 1
                self.cumulative_reward += reward

          # final coordinates of the dino after the jump
          self.dino_position_y = 0
          # since this is when the collision was avoided and we can assume
          # that here the nearest obstacle has been cleared, so we need to swap
          # the nearest and the next obstacle
        #   self.check_for_collision()

          # special case: one jump clears two obstacles
          if(self.nearest_obstacle_end_x<0 and self.next_obstacle_end_x<0):
            self.generate_next_obstacle()
            self.swap_obstacles()
            # clear the next obstacle
            self.clear_next_obstacle()

          elif(self.nearest_obstacle_end_x < 0):
            self.swap_obstacles()
            self.clear_next_obstacle()
          # else just a normal jump
              

    # case where the nearest obstacle is null


    if(self.next_obstacle_length == -1):
      self.generate_next_obstacle()

    # case where the nearest obstacle is null after swapping with none obstacle
    if(self.nearest_obstacle_length == -1):
      if self.next_obstacle_length != -1: # next obstacle is not null
        self.swap_obstacles()
        
      while (self.next_obstacle_length == -1):
        self.generate_next_obstacle()
        self.swap_obstacles()
      self.generate_next_obstacle() # after swapping non-null next obstacle, generate the next obstacle



    self._get_observation()
    
    return self.state, reward, done, truncated, info

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)

    info = {}
    self.cumulative_reward = 0
    self.dino_position_y = 0

    # Get an obstacle
    self.nearest_obstacle_length = self.np_random.integers(1,MAX_OBSTACLE_LENGTH+1)
    # self.nearest_obstacle_length = 1 # TESTING
    self.nearest_obstacle_distance = self.np_random.integers(1,GAME_SIZE-self.nearest_obstacle_length-1)
    # only bird can be in the sky, not cactus. Bird length is 1
    if self.nearest_obstacle_length == 1:
      self.nearest_obstacle_y = self.np_random.integers(0,2)
    else:
      self.nearest_obstacle_y = 0

    self.nearest_obstacle_end_x = self.nearest_obstacle_distance + self.nearest_obstacle_length

    # First consider only one obstacle
    self.next_obstacle_length = -1
    self.next_obstacle_distance = -1
    self.next_obstacle_y = -1
    self.next_obstacle_end_x = -1

    self._get_observation()

    print(f"Variables: {self.state},{info}")

    return self.state, info

  def render(self):
    """Render the current game state in the terminal."""

    # Clear terminal
    os.system("cls" if os.name == "nt" else "clear")

    # Create empty game board
    board = [[" " for _ in range(GAME_SIZE + 1)] for _ in range(2)]

    # Dino
    dino_x = 0
    board[self.dino_position_y][dino_x] = "D"

    # Nearest obstacle
    if self.nearest_obstacle_length != -1:
        start = max(0, int(self.nearest_obstacle_distance))
        end = min(
            GAME_SIZE,
            int(self.nearest_obstacle_distance +
                self.nearest_obstacle_length)
        )

        for x in range(start, end + 1):
            board[self.nearest_obstacle_y][x] = "O"

    # Next obstacle
    if self.next_obstacle_length != -1:
        start = max(0, int(self.next_obstacle_distance))
        end = min(
            GAME_SIZE,
            int(self.next_obstacle_distance +
                self.next_obstacle_length)
        )

        for x in range(start, end + 1):
            board[self.next_obstacle_y][x] = "N"

    # Print sky
    print("".join(board[1]))

    # Print ground
    print("".join(board[0]))

    # Debug information
    print()
    print(f"Dino Y: {self.dino_position_y}")

    print(
        f"Nearest obstacle: "
        f"x={self.nearest_obstacle_distance}, "
        f"length={self.nearest_obstacle_length}, "
        f"y={self.nearest_obstacle_y}"
    )

    print(
        f"Next obstacle: "
        f"x={self.next_obstacle_distance}, "
        f"length={self.next_obstacle_length}, "
        f"y={self.next_obstacle_y}"
    )

    print(f"Cumulative reward: {self.cumulative_reward}")

  def generate_next_obstacle(self):
    allowed_values = [-1] + list(range(1,MAX_OBSTACLE_LENGTH+1))
    # allowed_values = [1,-1,1,1,1,1,1,1,1] # TESTING
    self.next_obstacle_length = self.np_random.choice(allowed_values)
    # Obstacle was randomly chosen to not be generated if length is -1
    if self.next_obstacle_length == -1:
      self.next_obstacle_distance = -1
      self.next_obstacle_y = -1
      self.next_obstacle_end_x = -1
      return
    self.next_obstacle_distance = self.np_random.integers(self.nearest_obstacle_end_x+GAME_RATE, int(GAME_SIZE*1.5))
    self.next_obstacle_y = self.np_random.integers(0,2)
    if self.next_obstacle_length == 1:
      self.next_obstacle_y = self.np_random.integers(0,2)
    else:
      self.next_obstacle_y = 0

    self.next_obstacle_end_x = self.next_obstacle_distance + self.next_obstacle_length

  def check_for_collision(self):
  # Check collision between y-coordinates and then after the last jump the resultant obstacle distance must be negative
    if self.dino_position_y == self.nearest_obstacle_y and self.nearest_obstacle_distance<=0 and self.nearest_obstacle_end_x>=0:
      return True
    return False

  def _get_observation(self):
    self.state = np.array([
      self.dino_position_y,
      self.nearest_obstacle_distance,
      self.nearest_obstacle_length,
      self.nearest_obstacle_y,
      self.next_obstacle_distance,
      self.next_obstacle_length,
      self.next_obstacle_y
  ], dtype=np.float32)

def main():
    env = BasicEnv()

    state, info = env.reset(seed=288081653304652007915090246664360248990)

    print("Chrome Dino Debug Environment")
    print("-----------------------------")
    print("Controls:")
    print("  j       = jump")
    print("  n       = advance 1 step")
    print("  number  = advance that many NOTHING steps")
    print("  q       = quit")

    input("\nPress ENTER to start...")

    env.render()

    while True:

        key = input("\nAction: ").strip().lower()

        # Quit
        if key == "q":
            print("Exiting...")
            break

        # Jump
        elif key == "j":
            state, reward, terminated, truncated, info = env.step(JUMP)

            env.render()

            print(f"Action: JUMP")
            print(f"Reward: {reward}")

        # One NOTHING step
        elif key == "n":
            state, reward, terminated, truncated, info = env.step(NOTHING)

            env.render()

            print(f"Action: NOTHING")
            print(f"Reward: {reward}")

        # Multiple NOTHING steps
        elif key.isdigit():

            steps = int(key)

            if steps <= 0:
                print("Enter a positive number.")
                continue

            print(f"Advancing {steps} steps...")

            terminated = False
            truncated = False

            for _ in range(steps):

                state, reward, terminated, truncated, info = env.step(NOTHING)

                if terminated or truncated:
                    break

            env.render()

            print(f"Advanced: {_ + 1} steps")

            if terminated:
                print("GAME OVER!")
                print(f"Final reward: {env.cumulative_reward}")

        else:
            print("Invalid input.")
            print("Use j, n, a number, or q.")

        # Handle game over
        if terminated or truncated:

            print("\nGAME OVER!")
            print(f"Final cumulative reward: {env.cumulative_reward}")

            again = input("Play again? (y/n): ").strip().lower()

            if again == "y":
                state, info = env.reset()
                env.render()
            else:
                break


if __name__ == "__main__":
    main()