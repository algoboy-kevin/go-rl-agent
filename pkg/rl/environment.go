package rl

import "context"

type EnvironmentType string

type Environment interface {
	// To check if the episode ended or limit has reached
	IsTerminal() bool

	// Reset function
	Initialize() error

	// This include a step in environment
	PerformAction(action int) error
	GetReward() float64
	GetState() []float64

	// Add logging
	InitializeTrainingLogger(outputDir string) error
	InitializeTestLogger(outputDir string) error 

	// Write log after episode done
	WriteEpisodeLog() error

	// Get final result of episode
	GetEpisodeStats() (*EpisodeData, error)

	// Update Episode
	UpdateEpisodeCount(eps int)

	// Optional: for injected step that run inside the environment
	RunEpisode(ctx context.Context, agent *Agent, cb func() int) error
}