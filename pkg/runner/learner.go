package runner

import (
	"context"
	"io"
	"log/slog"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

// Learner embeds Base and adds training-specific fields and logic.
type Learner struct {
	*Base
	stepCounter    int
	episodeCounter int
}

// NewLearner creates a Learner and initialises the training logger.
func NewLearner(config rl.RLTrainingConfig, env rl.Environment, agent *rl.Agent) (*Learner, error) {
	l := &Learner{Base: NewBase(config, env, agent)}

	if err := env.InitializeTrainingLogger(config.OutputDir); err != nil {
		return nil, err
	}
	return l, nil
}

// RunEpisode runs one training episode with terminal handling and logging.
func (l *Learner) RunEpisode(ctx context.Context, m *rl.Agent) (bool, error) {
	l.stepCounter = 0

	done, err := l.runEpisode(ctx, m)
	if err != nil {
		if err != io.EOF {
			slog.Error("Found error", "err", err)
			return true, err
		}
		done = true
		err = nil
	}

	if done {
		m.HandleTerminal(l.episodeCounter)
		if err := l.WriteEpisodeResult(); err != nil {
			return true, err
		}
		l.episodeCounter++
	}

	return done, nil
}

// WriteEpisodeResult writes the episode log via the environment.
func (l *Learner) WriteEpisodeResult() error {
	return l.env.WriteEpisodeLog()
}
