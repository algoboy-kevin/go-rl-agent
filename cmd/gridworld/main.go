package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/algoboy-kevin/go-rl-agent/internal/envs"
	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

func main() {
	configPath := flag.String("config", "configs/gridworld.yaml", "path to YAML config")
	seed := flag.Int64("seed", 42, "random seed for reproducibility")
	gridW := flag.Int("width", 5, "grid width")
	gridH := flag.Int("height", 5, "grid height")
	maxSteps := flag.Int("max-steps", 1000, "max steps per episode")
	flag.Parse()

	// ── Load YAML config ──────────────────────────────────────────────
	data, err := os.ReadFile(*configPath)
	if err != nil {
		log.Fatalf("failed to read config %q: %v", *configPath, err)
	}

	var cfg rl.RLTrainingConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		log.Fatalf("failed to parse config: %v", err)
	}

	// Override for GridWorld — always 4 actions regardless of config
	cfg.Learning.NActions = 4

	// ── Create agent ───────────────────────────────────────────────────
	agent, err := rl.NewSARSA(cfg, *seed)
	if err != nil {
		log.Fatalf("failed to create SARSA agent: %v", err)
	}

	// ── Create environment ─────────────────────────────────────────────
	env := envs.NewGridWorldSized(*gridW, *gridH, *maxSteps)

	// ── Training header ────────────────────────────────────────────────
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  GridWorld — SARSA(λ) with Tile Coding")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("  Grid:       %d×%d\n", env.Width(), env.Height())
	fmt.Printf("  Optimal:    %d steps (Manhattan)\n", env.ShortestPathLength())
	fmt.Printf("  Episodes:   %d\n", cfg.Training.NEpisodes)
	fmt.Printf("  Tilings:    %d\n", cfg.Learning.NTilings)
	fmt.Printf("  Memory:     %d\n", cfg.Learning.MemorySize)
	fmt.Printf("  Seed:       %d\n", *seed)
	fmt.Println("───────────────────────────────────────────────────────")
	fmt.Printf("%-8s %-8s %-12s %-8s %-8s\n", "Episode", "Steps", "Goal?", "Alpha", "Eps")
	fmt.Println("───────────────────────────────────────────────────────")

	// ── Training loop ──────────────────────────────────────────────────
	bestSteps := int(^uint(0) >> 1) // max int
	successCount := 0

	for ep := 0; ep < cfg.Training.NEpisodes; ep++ {
		_ = env.Initialize()

		state := rl.NewStateInstance(
			cfg.Learning.MemorySize,
			cfg.Learning.NActions,
			cfg.Learning.NTilings,
		)
		_ = state.NewStateFromEnv(env)

		steps := 0
		for !env.IsTerminal() {
			action := agent.Action(state)
			_ = env.PerformAction(int(action))

			nextState := rl.NewStateInstance(
				cfg.Learning.MemorySize,
				cfg.Learning.NActions,
				cfg.Learning.NTilings,
			)
			_ = nextState.NewStateFromEnv(env)

			agent.HandleTransition(state, int(action), env.GetReward(), nextState)
			state = nextState
			steps++
		}

		agent.HandleTerminal(ep)

		reachedGoal := env.ReachedGoal()
		if reachedGoal {
			successCount++
			if steps < bestSteps {
				bestSteps = steps
			}
		}

		// Log every 25 episodes, and always log first/last/success
		shouldLog := ep == 0 ||
			ep == cfg.Training.NEpisodes-1 ||
			ep%25 == 0 ||
			(env.ReachedGoal() && steps <= 20)

		if shouldLog {
			fmt.Printf("%-8d %-8d %-12v %-8.4f %-8.4f\n",
				ep, steps, reachedGoal, agent.Alpha, agent.Policy.Descr())
		}
	}

	// ── Summary ────────────────────────────────────────────────────────
	fmt.Println("───────────────────────────────────────────────────────")
	fmt.Printf("  Success rate:  %d / %d (%.1f%%)\n",
		successCount, cfg.Training.NEpisodes,
		float64(successCount)/float64(cfg.Training.NEpisodes)*100)
	if bestSteps < int(^uint(0)>>1) {
		fmt.Printf("  Best steps:    %d (optimal: %d)\n", bestSteps, env.ShortestPathLength())
	}
	fmt.Println("═══════════════════════════════════════════════════════")
}
