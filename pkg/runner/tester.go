package runner

import (
	"context"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

// Tester embeds Base and evaluates a fixed policy without training updates.
type Tester struct {
	*Base
}

// NewTester creates a Tester and initialises the test logger.
func NewTester(config rl.RLTrainingConfig, env rl.Environment, agent *rl.Agent) (*Tester, error) {
	t := &Tester{Base: NewBase(config, env, agent)}

	if err := env.InitializeTestLogger(config.OutputDir); err != nil {
		return nil, err
	}
	return t, nil
}

// RunEpisode runs one evaluation episode.
func (t *Tester) RunEpisode(ctx context.Context, m *rl.Agent) (bool, error) {
	return t.runEpisode(ctx, m)
}

// WriteEpisodeResult is a no-op for the tester.
func (t *Tester) WriteEpisodeResult() error {
	return nil
}
