package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math"
	"os"
	"time"

	"github.com/algoboy-kevin/go-rl-agent/internal/envs"
	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
	"github.com/algoboy-kevin/go-rl-agent/pkg/trainer"
)

type gridParams struct {
	width, height, maxSteps int
}

var (
	grid10 = gridParams{10, 10, 5000}
	grid20 = gridParams{20, 20, 20000}
)

func makeConfig(nThreads, nEpisodes int) rl.RLTrainingConfig {
	return rl.RLTrainingConfig{
		Training: rl.TrainingConfig{
			NThreads:  nThreads,
			NEpisodes: nEpisodes,
		},
		Learning: rl.LearningConfig{
			MemorySize:   16384,
			NTilings:     16,
			NActions:     4,
			GroupWeights: []float64{0.5, 0.0, 0.5},
			Gamma:        0.99,
			Lambda:       0.9,
			Omega:        0.995,
			AlphaStart:   0.1,
			AlphaFloor:   0.01,
			RandomInit:   true,
		},
		Policy: rl.PolicyConfig{
			Type:     "epsilon_greedy",
			EpsInit:  0.5,
			EpsFloor: 0.01,
			EpsT:     500,
		},
		Model: rl.ModelConfig{
			Name:           "bench_gridworld",
			SaveEverySteps: 0,
		},
		OutputDir: "./output/bench",
	}
}

// evaluate runs nEval greedy episodes and returns success rate and best steps.
func evaluate(agent *rl.Agent, g *envs.GridWorld, nEval int) (successPct float64, bestSteps int) {
	agent.GoGreedy()
	bestSteps = math.MaxInt32
	successes := 0

	for ep := 0; ep < nEval; ep++ {
		_ = g.Initialize()
		state := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings)
		_ = state.NewStateFromEnv(g)

		steps := 0
		for !g.IsTerminal() {
			action := agent.Action(state)
			_ = g.PerformAction(int(action))
			nextState := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings)
			_ = nextState.NewStateFromEnv(g)
			agent.HandleTransition(state, int(action), g.GetReward(), nextState)
			state = nextState
			steps++
			if steps > g.MaxSteps()*2 {
				break
			}
		}

		if g.ReachedGoal() {
			successes++
			if steps < bestSteps {
				bestSteps = steps
			}
		}
	}

	return float64(successes) / float64(nEval) * 100, bestSteps
}

// trainDirect trains an agent in a simple sequential loop (like gridworld main.go).
func trainDirect(ctx context.Context, agent *rl.Agent, g *envs.GridWorld, nEpisodes int) {
	for ep := 0; ep < nEpisodes; ep++ {
		_ = g.Initialize()
		state := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings)
		_ = state.NewStateFromEnv(g)

		for !g.IsTerminal() {
			action := agent.Action(state)
			_ = g.PerformAction(int(action))
			nextState := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings)
			_ = nextState.NewStateFromEnv(g)
			agent.HandleTransition(state, int(action), g.GetReward(), nextState)
			state = nextState
		}
		agent.HandleTerminal(ep)
	}
}

// ── Baseline: train sequentially and check convergence ──────────────

func checkBaseline(gp gridParams, nEpisodes, nEval int) {
	cfg := makeConfig(1, nEpisodes)
	agent, _ := rl.NewSARSA(cfg, 42)
	env := envs.NewGridWorldSized(gp.width, gp.height, gp.maxSteps)

	start := time.Now()
	trainDirect(context.Background(), agent, env, nEpisodes)
	elapsed := time.Since(start)

	success, best := evaluate(agent, envs.NewGridWorldSized(gp.width, gp.height, gp.maxSteps), nEval)

	fmt.Printf("  Time:      %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  Success:   %.0f%% (%d/%d eval eps)\n", success, int(success*float64(nEval)/100), nEval)
	if best < math.MaxInt32 {
		fmt.Printf("  Best path: %d steps (optimal: %d)\n", best, (gp.width-1)+(gp.height-1))
	}
}

// ── Hogwild! timing via trainer ─────────────────────────────────────

func runBenchmark(ctx context.Context, nThreads int, gp gridParams, nEpisodes, nEval int) {
	cfg := makeConfig(nThreads, nEpisodes)
	spawn := func() (rl.Environment, error) {
		return envs.NewGridWorldSized(gp.width, gp.height, gp.maxSteps), nil
	}

	// Warm-up.
	warmCfg := makeConfig(nThreads, 50)
	warmup, _ := trainer.NewTrainer(warmCfg, spawn)
	_ = warmup.StartTraining(ctx)

	// Benchmark: 3 runs.
	cfg = makeConfig(nThreads, nEpisodes)
	var total time.Duration
	nRuns := 3
	var lastAgent *rl.Agent

	for i := 0; i < nRuns; i++ {
		bt, _ := trainer.NewTrainer(cfg, spawn)
		start := time.Now()
		_ = bt.StartTraining(ctx)
		total += time.Since(start)
		lastAgent = bt.Agent()
	}

	avg := total / time.Duration(nRuns)
	eps := float64(nEpisodes) / avg.Seconds()

	// Evaluate convergence using last trained agent.
	evalEnv := envs.NewGridWorldSized(gp.width, gp.height, gp.maxSteps)
	success, best := evaluate(lastAgent, evalEnv, nEval)

	fmt.Printf("%-6s  %s avg  (%.0f ep/s)  success=%.0f%% best=%d\n",
		fmt.Sprintf("%d thread(s)", nThreads),
		avg.Round(time.Millisecond), eps,
		success, best)
}

func main() {
	ctx := context.Background()
	if os.Getenv("GO_RL_BENCH_VERBOSE") == "" {
		log.SetOutput(io.Discard)
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	}

	// ── 10×10 Grid ──────────────────────────────────────────────────
	fmt.Println("═══ 10×10 Grid ═══")
	fmt.Println()
	fmt.Println("── Baseline (sequential training) ──")
	checkBaseline(grid10, 500, 100)
	fmt.Println()
	fmt.Println("── Hogwild! trainer ──")
	for _, n := range []int{1, 2, 4, 8} {
		runBenchmark(ctx, n, grid10, 500, 100)
	}

	// ── 20×20 Grid ──────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("═══ 20×20 Grid ═══")
	fmt.Println()
	fmt.Println("── Baseline (sequential training) ──")
	checkBaseline(grid20, 2000, 100)
	fmt.Println()
	fmt.Println("── Hogwild! trainer ──")
	for _, n := range []int{1, 2, 4, 8} {
		runBenchmark(ctx, n, grid20, 2000, 100)
	}
}
