import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import numpy as np

from grid_world import GridWorld
from dqn_agent import DQNLearningAgent

hyperparameters = {
    'NUM_EPISODES': 6000,
    'LR': 0.001,
    'GAMMA': 0.95,
    'BATCH_SIZE': 64,
    'TARGET_UPDATE_FREQ': 10,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 30000
}

def draw_grid(env, ax, Q):
    """Render the GridWorld with Q-values and policy arrows."""
    ax.clear()
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=2)
    ax.set_xlim(-0.5, env.width-0.5)
    # Invert y-axis to match grid (0,0) at top-left
    ax.set_ylim(env.height-0.5, -0.5)
    
    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state == env.block_state:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                               color='gray'))
                ax.text(j, i, 'X', ha='center', va='center', 
                        fontsize=16, color='white')
            elif state in env.terminal_states:
                val = env.rewards[state]
                color = 'lightgreen' if val > 0 else 'red'
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                               color=color))
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        fontsize=16)
            else:
                # This is a normal state, show Q-values and policy
                qvals = Q[i, j]
                # Format Q-values text (N, E, S, W)
                txt = (f'N:{qvals[0]:.2f}\nE:{qvals[1]:.2f}\n'
                       f'S:{qvals[2]:.2f}\nW:{qvals[3]:.2f}')
                ax.text(j, i, txt, ha='center', va='center', fontsize=8)
                
                # Show best action arrow
                best_a = np.argmax(qvals)
                dx, dy = 0, 0
                if best_a == 0: dy = -0.3 # N
                elif best_a == 1: dx = 0.3 # E
                elif best_a == 2: dy = 0.3 # S
                elif best_a == 3: dx = -0.3 # W
                ax.arrow(j, i, dx, dy, head_width=0.15, 
                         head_length=0.15, fc='blue', ec='blue')
                
    ax.set_title('GridPilot — Q-values and policy')

def main():
    print("GridPilot — initializing environment and agent...")
    
    env = GridWorld(trap_penalty=-1.0)
    agent = DQNLearningAgent(env, hyperparameters)
    
    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    
    plt.ion()
    fig_grid, ax_grid = plt.subplots(figsize=(env.width * 2, env.height * 2))
    print("Showing initial grid. Training will start in 2 seconds...")
    draw_grid(env, ax_grid, agent.get_q_table())
    plt.pause(2.0)

    print(f"Starting training for {hyperparameters['NUM_EPISODES']} episodes...")

    # --- Main Training Loop ---
    for i_episode in range(hyperparameters['NUM_EPISODES']):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        
        # This shows the epsilon value as the episode begins
        episode_epsilons.append(agent.get_current_epsilon())

        while True:
            # 1. Agent chooses an action
            action_tensor = agent.choose_action(state)
            action = action_tensor.item()
            
            # 2. Environment processes the action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 3. Store this "memory" in the replay buffer
            agent.memory.push(state, action_tensor, next_state, reward, done)
            
            # 4. Move to the next state
            state = next_state
            
            # 5. Learn from a batch of memories
            loss = agent.learn_from_memory()
            if loss is not None:
                total_loss += loss
                steps += 1
            
            # Episode finished
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_loss = (total_loss / steps) if steps > 0 else 0
        episode_losses.append(avg_loss)
        
        if i_episode % hyperparameters['TARGET_UPDATE_FREQ'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode+1}/{hyperparameters['NUM_EPISODES']} | "
                  f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.3f} | "
                  f"Avg Loss (last 100): {np.mean(episode_losses[-100:]):.5f}")
            
            current_q_table = agent.get_q_table()
            draw_grid(env, ax_grid, current_q_table)
            plt.pause(0.1)

    print("Training complete.")

    print("Displaying final policy grid. Close plot to continue.")
    plt.ioff()
    draw_grid(env, ax_grid, agent.get_q_table())
    plt.show()

    # --- Print Final Policy ---
    print("\n" + "="*30)
    print("Final Learned Policy:")
    final_policy = agent.get_policy()
    print(final_policy)
    print("="*30 + "\n")

    # --- Plot Learning Curve ---
    print("Generating learning curve plot...")
    
    moving_avg = np.convolve(episode_rewards, 
                           np.ones(100)/100, mode='valid')
    
    fig_curve, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward per Episode', color=color)
     # Get the line handles so we can make a combined legend
    l1 = ax1.plot(episode_rewards, color=color, alpha=0.3, 
             label='Episode Reward')[0]
    l2 = ax1.plot(np.arange(99, len(episode_rewards)), moving_avg, 
             color='red', linewidth=2, label='Moving Avg (100 episodes)')[0]
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3) # Add grid for the primary axis
    
    # ax2 (Right Axis): Epsilon Decay
    ax2 = ax1.twinx() # Create a second y-axis
    color = 'tab:green'
    ax2.set_ylabel('Epsilon (Exploration Rate)', color=color)
    # Plot the epsilon history we tracked
    l3 = ax2.plot(episode_epsilons, color=color, alpha=0.7, linestyle='--',
             label='Epsilon Decay')[0]
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('GridPilot — reward and exploration decay')
    fig_curve.tight_layout() # Use fig_curve
    
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    plt.savefig("dqn_learning_curve.png")
    print("Saved learning curve to 'dqn_learning_curve.png'")
    plt.show() # Show the learning curve plot

if __name__ == "__main__":
    main()