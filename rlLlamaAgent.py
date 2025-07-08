from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from alphaZero import run_mcts
from minesweeperGenerator import userAction


def load_tiny_llama_lora():
    base_model = AutoModelForCausalLM.from_pretrained("TinyLLaMA/TinyLLaMA-1.1B-Chat", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("TinyLLaMA/TinyLLaMA-1.1B-Chat")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)
    return model, tokenizer

llama_model, llama_tokenizer = load_tiny_llama_lora()

def llama_agent_step(obs):
    prompt = f"You are playing Minesweeper. Board state:\n{obs}\nWhich cell should be uncovered next (return the index)?"
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
    outputs = llama_model.generate(**inputs, max_new_tokens=10)
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        action = int([s for s in response.split() if s.isdigit()][-1])
    except:
        action = np.random.randint(run_mcts(userAction))
    return action

game = userAction()
game.training_action("start")
done = False
while not done:
    # obs = obs.stringify_visible_map()
    action = llama_agent_step(obs)
    reward, done, win, obs = game.llm_training_action(action)

print("TinyLLaMA agent finished with reward:", reward+win*10 )