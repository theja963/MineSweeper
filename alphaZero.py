from collections import deque
import random
import torch 
import numpy as np
from minesweeperGenerator import userAction 

class cnn(torch.nn.Module):
    def __init__(self, board_size):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),  # 3 chatorch.nnels: revealed numbers, flags, hidden
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU()
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(board_size * board_size * 64, board_size * board_size),
            torch.nn.Softmax(dim=-1)
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(board_size * board_size * 64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh()
        )
        self.board_size = board_size

    def forward(self, x):
        x = self.conv(x)
        return self.policy_head(x), self.value_head(x)
    

def run_mcts(action:userAction):
    # valid_actions = valid_actions*action.count_map(action.visible_map())
    action_probs = action.count_map(action.visible_map().fillna(1))
    valid_actions = [
        action_probs.iat[i,j] if not action.revealed_map.iat[i, j] and not action.flagged_map.iat[i, j] and action_probs.iat[i,j] != 9 else 0.2*(not action.revealed_map.iat[i, j])
        for i in range(action.size) for j in range(action.size)]
    
    print("Action probables",action_probs,"before filter",action.revealed_map,"invisible",action.generated_map)

    valid_sum = sum([x for x in valid_actions])+1
    
    if valid_sum==1: ## debug sequence for analysis
        action.terminate_flag = True
        action.win_status()
        return None
        # print("Action probables",action_probs,"before filter",action.revealed_map,"invisible",action.generated_map)
        # if input(print("Continue?")): pass
    
    print("Action bug",valid_actions,valid_sum)
    valid_actions = [x / valid_sum for x in valid_actions]
    # print("Action probables",valid_actions)
    action_probs = torch.tensor(valid_actions)
    
    # print("Action probables",action_probs,"converted")
    value = valid_sum
    return action_probs,value


class trainer:
    def __init__(self, model, env, lr=1e-3):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0)
        self.replay = deque(maxlen=10000)
        self.wins = 0

    def self_play(self, n_games=10):
        for _ in range(n_games):


            game = userAction(map_size=self.model.board_size)
            obs = game.training_action("start")[1]

            # predict_loc = run_mcts(game)
            
            # obs = obs.to_numpy_tensor() ## starting all at centre need to improve for later

            done = False
            while not done:
                # Convert obs to tensor
                x = torch.tensor(obs).unsqueeze(0).float()
                with torch.no_grad():
                    pi, _ = self.model(x)
                    predict = run_mcts(game)
                    
                    # print("Returned",pi, "comb action?", predict)
                    if not predict:
                        done = True
                        break
                        break
                    segment_pi = torch.argmax(pi*torch.flatten(predict[0])).item()

                    # print("Returned",pi, "action?")
                    action = "r,"+str(segment_pi//self.model.board_size)+","+str(segment_pi%self.model.board_size)
                    # print("Returned",pi, "action?",action,"manipulated?",game.revealed_map,"visible",game.visible_map())
                    # if input(print("Continue?")):continue
                print(segment_pi,"action",action)
                reward,next_obs, done,win,map = game.training_action(action)
                print(action,segment_pi,map,"Update papameters",done, pi.squeeze().numpy(), reward,)
                self.wins+=win
                self.replay.append((obs, pi.squeeze().numpy(), reward+win))
                obs = next_obs
                if win:
                    game = userAction()
                


    def train(self, batch_size=32):
        batch = random.sample(self.replay, min(len(self.replay), batch_size))
        for state, target_pi, target_value in batch:
            x = torch.tensor(state).unsqueeze(0).float()
            pi_pred, v_pred = self.model(x)
            loss = (torch.nn.functional.mse_loss(v_pred, torch.tensor([[target_value]])) +
                    torch.nn.functional.cross_entropy(pi_pred, torch.tensor([np.argmax(target_pi)])))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        for param in self.model.parameters():
            if param.requires_grad:
                total = param.data.sum()
                if total != 0:
                    param.data /= total


def execute_training_cycles(trainer=trainer,cnn=cnn,epochs = 10):
    board_model = cnn(board_size=4)
    trainer = trainer(board_model, userAction)

    for epoch in range(epochs):
        trainer.self_play(n_games=5)
        trainer.train(batch_size=4)
        print(f"Epoch {epoch} complete",trainer.wins)

execute_training_cycles()