package rl

import "time"

type EpisodeData struct {
	EpisodeID    string    `yaml:"id"`
	EpisodeIndex int       `yaml:"episode_index"`
	StrategyID   string    `yaml:"strategy_id"`
	AgentID      string    `yaml:"agent_id"`
	Timestamp    time.Time `yaml:"timestamp"`
	NStep        int       `yaml:"n_step"`

	Data any `yaml:"data"`
}
