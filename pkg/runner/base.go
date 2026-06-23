package runner

import (
	"context"
	"log/slog"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

// Base holds common fields and logic shared by all runner types.
//
// State buffer lifecycle for Step():
//
//	Tick N:   state = refresh from env → learn(prevState, prevAction, reward, state)
//	          → act on state → swap(prevState, state) → PerformAction
//	Tick N+1: state = refresh from env → learn(prevState, prevAction, reward, state)
//	          → act → swap → PerformAction ...
type Base struct {
	env       rl.Environment
	state     *rl.State // refreshed from env each tick, used for action + transition "next state"
	prevState *rl.State // pre-action state from previous tick, used for transition "state"
	agent     *rl.Agent

	prevAction int  // action taken at the previous tick
	hasPrev    bool // false on first tick — no transition to learn from

	stepCount int // total steps across all episodes (for periodic logging)
}

// NewBase creates a Base with two state buffers.
func NewBase(config rl.RLTrainingConfig, env rl.Environment, agent *rl.Agent) *Base {
	state1 := rl.NewStateFromConfig(config)
	state2 := rl.NewStateFromConfig(config)
	return &Base{
		env:       env,
		agent:     agent,
		state:     state1,
		prevState: state2,
	}
}

// EpisodeInit prepares the environment and captures the initial state.
func (b *Base) EpisodeInit() error {
	if err := b.env.Initialize(); err != nil {
		return err
	}
	return b.state.NewStateFromEnv(b.env)
}

// Reset clears the transition buffer so the runner starts a fresh episode.
// Call on market rotation to prevent cross-market transition contamination.
func (b *Base) Reset() {
	b.hasPrev = false
	b.stepCount = 0
	b.prevAction = 0
}

// Step runs one RL step and returns the selected action.
// The transition is recorded with a one-tick delay so that reward reflects
// market events (fills, price changes) that occurred between ticks.
// Pass b.Step directly as a callback to SetActionCallback or env.RunEpisode.
func (b *Base) GetAction() int {
	if b.env.IsTerminal() {
		return -1
	}

	b.stepCount++

	// 1. Refresh state — captures market events since last tick.
	b.state.NewStateFromEnv(b.env)

	// 2. Learn from previous action: (prev_s, prev_a, reward, current_s)
	if b.hasPrev {
		reward := b.env.GetReward()
		b.agent.HandleTransition(b.prevState, b.prevAction, reward, b.state)
	}

	// 3. Compute action from fresh state.
	action := b.agent.Action(b.state)
	b.prevAction = action

	// 4. Save pre-action state via buffer swap.
	b.prevState, b.state = b.state, b.prevState
	b.hasPrev = true

	// 5. Execute on environment.
	if err := b.env.PerformAction(action); err != nil {
		slog.Error("rl perform action", "err", err)
		return -1
	}

	return action
}

// runEpisode is the common episode loop shared by Learner and Tester.
// It builds the action callback and passes it to the environment so the
// strategy's tick routine drives RL steps.
func (b *Base) runEpisode(ctx context.Context, m *rl.Agent) (bool, error) {
	if err := b.EpisodeInit(); err != nil {
		return true, err
	}

	err := b.env.RunEpisode(ctx, m, b.GetAction)
	if err != nil {
		return false, err
	}

	// Check if the environment signals episode completion (e.g. DES loop
	// reached EOF).  Used by Learner to finalize checkpoints and logs.
	return b.env.IsTerminal(), nil
}
