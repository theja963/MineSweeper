# 🧠 Minesweeper AI with AlphaZero & TinyLLaMA

This project implements a Reinforcement Learning (RL) agent that plays Minesweeper using a hybrid of AlphaZero-style Monte Carlo Tree Search (MCTS), convolutional neural networks (CNNs), and a TinyLLaMA LLM agent with LoRA fine-tuning.

---

## 📁 Project Structure

```
.
├── alphaZero.py             # AlphaZero-style CNN trainer & MCTS logic
├── minesweeperGenerator.py # Game environment for Minesweeper with training hooks
├── rlLlamaAgent.py          # TinyLLaMA agent for language-based Minesweeper decision-making
```

---

## ⚙️ Features

- ✅ Fully playable Minesweeper environment using NumPy and Pandas.
- 🧠 CNN-based policy and value network for self-play.
- 🌲 Monte Carlo Tree Search (restricted) for action sampling.
- 🔁 Replay buffer and training loop for improving agent over epochs.
- 🤖 TinyLLaMA-based agent that plays Minesweeper using stringified visible maps.
- 💬 Instruction-style prompt formatting for LLM-based decision making.

---

## 🧩 Components

### `minesweeperGenerator.py`

- Core game logic (reveal, flag, unflag).
- Ensures first-click safety.
- Converts board state to tensor for training.
- Used by both RL and LLM agents.

### `alphaZero.py`

- Defines `cnn` model: shared convolution + separate policy and value heads.
- `run_mcts`: Generates action probabilities from current visible map.
- `trainer`: Performs self-play, maintains replay memory, and trains model on sampled experiences.
- `execute_training_cycles`: Main training loop (can be extended for checkpoints, metrics, etc.)

### `rlLlamaAgent.py`

- Loads TinyLLaMA with LoRA (`peft`).
- Prompts LLM with current board state.
- Extracts next move prediction via token decoding.
- Runs in a game loop with environment's `llm_training_action`.

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch transformers peft pandas numpy scipy
```

### Run AlphaZero Training

```bash
python alphaZero.py
```

This will train a CNN policy/value network through 10 epochs of self-play.

### Run TinyLLaMA Agent

```bash
python rlLlamaAgent.py
```

This will play a game of Minesweeper using TinyLLaMA predictions on the visible map.

---

## 📌 Notes

- LoRA and TinyLLaMA fine-tuning setup is included but assumes the base model is downloaded and accessible.
- MCTS here is a placeholder; can be expanded with search tree and backpropagation.
- Tensor formatting assumes 3 channels: `[revealed_numbers, revealed_mask, flagged_mask]`.

---

## 🧠 TODO

- Improve MCTS to real tree-based rollout.
- Add GUI or CLI interface for interactive play.
- Save and load model checkpoints.
- Integrate reward shaping for better training signals.
- Integrate text-to-action parsing via LLM outputs robustly.

---

## 📜 License

MIT License
