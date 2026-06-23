# go-rl-agent

A reinforcement learning library in Go featuring **SARSA**(λ) with **linear function approximation**, **tile coding**, and **eligibility traces**. Designed for continuous state spaces and discrete action spaces — originally built for algorithmic trading environments.

## Features

- **SARSA(λ) with Tile Coding** — On-policy TD learning with eligibility traces for efficient credit assignment across state-action pairs.
- **Tile Coding (CMAC)** — Multi-resolution feature representation via multiple overlapping tilings. Uses the UNH hashing algorithm for memory-efficient index computation.
- **Eligibility Traces (Accumulating)** — Sparse trace representation that only tracks non-zero entries with automatic tolerance adjustment.
- **Multiple Action Selection Policies**
  - **Epsilon-Greedy** with configurable annealing schedule
  - **Boltzmann (Softmax)** with temperature annealing
  - **Greedy** with random tie-breaking
  - **Random**
- **Hogwild! Multithreaded Training** — Lock-free parallel SARSA(λ) training. Each worker gets its own Traces, Rand, and Policy while sharing the weight vector. Configure via `n_threads` in YAML.
- **Checkpointing** — Save and load model weights via Go's `gob` encoding. Automatically resumes from the latest checkpoint.
- **Configurable via YAML** — All hyperparameters, environment settings, and training configuration are defined in a structured YAML config.
- **Environment Interface** — Abstract `RLEnvironment` interface makes it easy to plug in custom environments (trading, games, etc.).

## Architecture

```
pkg/
├── environment/
│   └── types.go          # RLEnvironment interface definition
├── rl/
│   ├── agent.go          # Core Agent: SARSA, Q-value computation, weight updates, checkpoint save/load
│   ├── config.go         # Full YAML configuration structs (training, policy, market, state, etc.)
│   ├── ct.go             # CollisionTable for hash-based tile coding with collision resolution
│   ├── policy.go         # Action selection: Random, Greedy, Epsilon-Greedy, Boltzmann
│   ├── state.go          # State representation with tile-coded feature extraction
│   ├── tiles.go          # Tile coding algorithm (CMAC) with UNH hashing
│   ├── traces.go         # Accumulating eligibility traces with sparse representation
│   └── types.go          # Serializable SavedAgent struct for checkpointing
├── runner/
│   ├── base.go           # Common RL step loop (state refresh → learn → act → execute)
│   ├── learner.go         # Training episode runner with terminal handling
│   ├── tester.go          # Evaluation episode runner (no training updates)
│   └── types.go           # Runner interface definition
└── trainer/
    └── trainer.go         # High-level trainer with Hogwild! multi-worker orchestration
```

## Getting Started

### Prerequisites

- Go 1.23.1+

### Installation

```bash
go get github.com/algoboy-kevin/go-rl-agent
```

### Minimal Example

```go
package main

import (
    "github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

func main() {
    cfg := &rl.RLTrainingConfig{
        Learning: rl.LearningConfig{
            MemorySize:   4096,
            NTilings:     8,
            NActions:     3,
            AlphaStart:   0.1,
            AlphaFloor:   0.01,
            Omega:        0.995,
            Gamma:        0.99,
            Lambda:       0.9,
            RandomInit:   true,
            GroupWeights: []float64{0.4, 0.3, 0.3},
        },
        Policy: rl.PolicyConfig{
            Type:     "epsilon_greedy",
            EpsInit:  0.5,
            EpsFloor: 0.01,
            EpsT:     1000,
        },
        Model: rl.ModelConfig{
            Name: "my_agent",
        },
        OutputDir: "./output",
    }

    agent, err := rl.NewSARSA(cfg, 42)
    if err != nil {
        panic(err)
    }

    // Use with any environment implementing RLEnvironment
    // for each step:
    //   action := agent.Action(state)
    //   delta := agent.HandleTransition(fromState, action, reward, toState)
}
```

## Core Concepts

### Tile Coding

States are represented via **tile coding** (CMAC) — continuous variables are quantized into overlapping tilings, producing a sparse binary feature vector. This gives the linear function approximator the ability to generalize across similar states while discriminating between different regions of the state space.

The `State` struct computes **three feature groups** for each action:
1. First 3 state variables
2. Remaining state variables
3. All state variables combined

Each group is weighted by configurable `GroupWeights` (default: `[⅓, ⅓, ⅓]`).

### SARSA(λ) Updates

The agent performs on-policy TD learning:

$$\delta = r + \gamma \cdot V(s') - V(s) + \gamma \cdot \Phi(s') - \Phi(s)$$

Where $\Phi(s)$ is the **potential** of a state (used for potential-based reward shaping).

Weights are updated using eligibility traces:

$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot e$$

where $e$ is the accumulated eligibility trace.

### Learning Rate Annealing

$$\alpha_t = \max(\alpha_{\text{floor}}, \alpha_{\text{start}} \cdot \omega^{\text{episode}})$$

## Configuration

Full YAML configuration structure:

```yaml
training:
  n_threads: 4
  n_samples: 1000
  n_episodes: 500

learning:
  memory_size: 4096        # Number of tile indices in memory
  n_tilings: 8             # Number of overlapping tilings
  n_actions: 3
  algorithm: "sarsa"
  group_weights: [0.4, 0.3, 0.3]
  gamma: 0.99              # Discount factor
  lambda: 0.9              # Trace decay rate
  omega: 0.995             # Alpha decay factor
  alpha_start: 0.1         # Initial learning rate
  alpha_floor: 0.01        # Minimum learning rate
  random_init: true        # Randomly initialize weights

policy:
  type: "epsilon_greedy"
  eps_init: 0.5
  eps_floor: 0.01
  eps_T: 1000

model:
  name: "my_agent"
  load_last_checkpoint: true
  save_every_nstep: 10000
```

## Implementing a Custom Environment

To use the agent with your own environment, implement the `RLEnvironment` interface:

```go
import "github.com/algoboy-kevin/go-rl-agent/pkg/environment"

type MyEnv struct { /* ... */ }

func (e *MyEnv) IsTerminal() bool                         { /* ... */ }
func (e *MyEnv) Initialize() error                        { /* ... */ }
func (e *MyEnv) PerformAction(action int) error           { /* ... */ }
func (e *MyEnv) GetReward() float64                       { /* ... */ }
func (e *MyEnv) GetState() ([]float64, error)             { /* ... */ }
func (e *MyEnv) GetGUIUpdate() any                        { /* ... */ }
func (e *MyEnv) InitializeTrainingLogger(outputDir string) error { /* ... */ }
func (e *MyEnv) InitializeTestLogger(outputDir string) error     { /* ... */ }
func (e *MyEnv) WriteEpisodeLog(...) error                { /* ... */ }
func (e *MyEnv) GetStats(...) []string                    { /* ... */ }
func (e *MyEnv) GetEpisodeStats(...) []string             { /* ... */ }
func (e *MyEnv) UpdateEpisodeCount(eps int)               { /* ... */ }
func (e *MyEnv) Clear() error                             { /* ... */ }
```

## API Overview

| Function / Method | Description |
|---|---|
| `NewSARSA(config, seed)` | Creates a new SARSA(λ) agent |
| `agent.NewWorkerView(config, seed)` | Creates a worker agent sharing Theta but with independent Traces/Rand/Policy (Hogwild!) |
| `agent.Action(state)` | Selects an action using the current policy |
| `agent.HandleTransition(from, action, reward, to)` | Processes a transition: updates traces, weights, and optionally saves a checkpoint |
| `agent.GetQ(state, action)` | Computes Q-value for a state-action pair |
| `agent.GoGreedy()` | Switches to a greedy policy (for evaluation) |
| `agent.HandleTerminal(episode)` | Resets traces and decays learning rate at episode end |
| `agent.SaveModel(dir, filename)` | Serializes model weights to a binary file |
| `LoadModel(path)` | Loads a serialized model checkpoint |
| `NewStateFromConfig(config)` | Creates a state instance from config |
| `state.NewStateFromEnv(env)` | Updates state from environment observations |

## License

MIT
