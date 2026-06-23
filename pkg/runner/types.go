package runner

import (
	"context"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
)

type RunnerType int

const (
	TypeBase RunnerType = iota
	TypeLearner
	TypeBacktester
)

// Runner defines the contract for RL episode runners.
type Runner interface {
	EpisodeInit() error
	RunEpisode(ctx context.Context, m *rl.Agent) (bool, error)
	WriteEpisodeResult() error
	GetAction() int
}
