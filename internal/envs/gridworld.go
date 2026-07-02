package envs

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

const (
	ActionUp    = 0
	ActionDown  = 1
	ActionLeft  = 2
	ActionRight = 3

	DefaultGridWidth  = 5
	DefaultGridHeight = 5
	DefaultMaxSteps   = 1000
)

// GridWorld is a simple grid navigation environment with continuous state
// representation. The agent starts at (0,0) and must reach (width-1, height-1).
type GridWorld struct {
	width     int
	height    int
	maxSteps  int
	stepCount int

	// Agent position (discrete grid coordinates)
	px int
	py int

	// Reward for the most recent transition
	lastReward float64

	// Episode tracking
	episodeCount int
	episodeSteps int
	reachedGoal  bool

	// Logging state (minimal stubs)
	outputDir string
}

// NewGridWorld creates a new GridWorld environment.
func NewGridWorld() *GridWorld {
	return &GridWorld{
		width:    DefaultGridWidth,
		height:   DefaultGridHeight,
		maxSteps: DefaultMaxSteps,
	}
}

// NewGridWorldSized creates a GridWorld with a custom grid size.
func NewGridWorldSized(width, height, maxSteps int) *GridWorld {
	return &GridWorld{
		width:    width,
		height:   height,
		maxSteps: maxSteps,
	}
}

// IsTerminal returns true if the agent reached the goal or exceeded max steps.
func (g *GridWorld) IsTerminal() bool {
	if g.stepCount >= g.maxSteps {
		return true
	}
	return g.px == g.width-1 && g.py == g.height-1
}

// Initialize resets the environment to the start state.
func (g *GridWorld) Initialize() error {
	g.px = 0
	g.py = 0
	g.stepCount = 0
	g.lastReward = 0.0
	g.episodeSteps = 0
	g.reachedGoal = false
	return nil
}

// PerformAction moves the agent in the specified direction.
func (g *GridWorld) PerformAction(action int) error {
	g.stepCount++

	prevX, prevY := g.px, g.py

	switch action {
	case ActionUp:
		g.py--
	case ActionDown:
		g.py++
	case ActionLeft:
		g.px--
	case ActionRight:
		g.px++
	default:
		return fmt.Errorf("gridworld: unknown action %d", action)
	}

	// Bounce off walls (agent stays in place)
	if g.px < 0 || g.px >= g.width || g.py < 0 || g.py >= g.height {
		g.px, g.py = prevX, prevY

		if g.IsTerminal() {
			g.lastReward = 10.0
		} else {
			g.lastReward = -5.0 // wall penalty
		}
		return nil
	}

	// Reached goal
	if g.px == g.width-1 && g.py == g.height-1 {
		g.lastReward = 10.0
		g.reachedGoal = true
		return nil
	}

	// Default step penalty
	g.lastReward = -1.0
	return nil
}

// GetReward returns the reward from the most recent action.
func (g *GridWorld) GetReward() float64 {
	return g.lastReward
}

// GetState returns the normalized state representation [x, y, 0].
// The third dimension is a padding variable to match the tile coding's
// expectation of at least 3 state variables.
func (g *GridWorld) GetState() []float64 {
	return []float64{
		float64(g.px) / float64(g.width-1),
		float64(g.py) / float64(g.height-1),
		0.0,
	}
}

// InitializeTrainingLogger is a no-op for benchmark environments.
func (g *GridWorld) InitializeTrainingLogger(outputDir string) error {
	g.outputDir = outputDir
	return nil
}

// InitializeTestLogger is a no-op for benchmark environments.
func (g *GridWorld) InitializeTestLogger(outputDir string) error {
	g.outputDir = outputDir
	return nil
}

// WriteEpisodeLog is a no-op for benchmark environments.
func (g *GridWorld) WriteEpisodeLog() error {
	return nil
}

// GetEpisodeStats returns basic episode statistics.
func (g *GridWorld) GetEpisodeStats() (*rl.EpisodeData, error) {
	return &rl.EpisodeData{
		EpisodeIndex: g.episodeCount,
		Timestamp:    time.Now(),
		NStep:        g.stepCount,
		Data: map[string]interface{}{
			"steps":        g.stepCount,
			"reached_goal": g.reachedGoal,
		},
	}, nil
}

// UpdateEpisodeCount increments the episode counter.
func (g *GridWorld) UpdateEpisodeCount(eps int) {
	g.episodeCount = eps
}

// RunEpisode runs a full episode using the given agent.
func (g *GridWorld) RunEpisode(ctx context.Context, agent *rl.Agent, cb func() int) error {
	if err := g.Initialize(); err != nil {
		return err
	}

	state := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings, agent.GroupSplits)
	state.NewStateFromEnv(g)

	for !g.IsTerminal() {
		action := agent.Action(state)
		if err := g.PerformAction(int(action)); err != nil {
			return err
		}

		nextState := rl.NewStateInstance(agent.MemorySize, agent.NActions, agent.NTilings, agent.GroupSplits)
		nextState.NewStateFromEnv(g)

		agent.HandleTransition(state, int(action), g.GetReward(), nextState)
		state = nextState
	}

	return nil
}

// NumActions returns the number of discrete actions (4).
func (g *GridWorld) NumActions() int {
	return 4
}

// Steps returns the current step count for this episode.
func (g *GridWorld) Steps() int {
	return g.stepCount
}

// ReachedGoal returns whether the agent reached the goal in the current episode.
func (g *GridWorld) ReachedGoal() bool {
	return g.reachedGoal
}

// ShortestPathLength returns the Manhattan distance from start to goal.
// Useful as a reference for optimal performance.
func (g *GridWorld) ShortestPathLength() int {
	return (g.width - 1) + (g.height - 1)
}

// Position returns the agent's current grid coordinates.
func (g *GridWorld) Position() (int, int) {
	return g.px, g.py
}

// String returns a human-readable representation of the grid.
func (g *GridWorld) String() string {
	s := ""
	for y := 0; y < g.height; y++ {
		for x := 0; x < g.width; x++ {
			switch {
			case x == g.px && y == g.py:
				s += "A "
			case x == g.width-1 && y == g.height-1:
				s += "G "
			default:
				s += ". "
			}
		}
		s += "\n"
	}
	return s
}

// OptimalSteps returns the minimum number of steps to reach the goal
// from the current position (Manhattan distance).
func (g *GridWorld) OptimalSteps() int {
	return int(math.Abs(float64(g.width-1-g.px)) + math.Abs(float64(g.height-1-g.py)))
}

// Width returns the grid width.
func (g *GridWorld) Width() int {
	return g.width
}

// Height returns the grid height.
func (g *GridWorld) Height() int {
	return g.height
}

// MaxSteps returns the maximum steps before truncation.
func (g *GridWorld) MaxSteps() int {
	return g.maxSteps
}
