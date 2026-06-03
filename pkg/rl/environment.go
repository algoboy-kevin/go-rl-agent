package rl

type EnvironmentType string

type Environment interface {
	// To check if the episode ended or limit has reached
	IsTerminal() bool

	// Reset function
	Initialize() error

	// This include a step in environment
	PerformAction(action int) error
	GetReward() float64
	GetState() ([]float64, error)
	GetGUIUpdate() any

	// Add logging
	InitializeTrainingLogger(outputDir string) error
	InitializeTestLogger(outputDir string) error 

	// Write log after episode done
	WriteEpisodeLog(agentName string, policyValue float64, nStep, episodeCounter int64) error

	// Get row values for stats logger
	GetStats(agentName string, stepCounter int64) []string

	// Get final result of episode
	GetEpisodeStats(policyVal float64, episodeCounter int64) []string

	// Update Episode
	UpdateEpisodeCount(eps int)

	// Optional: for example closing opened orders without reset the final state
	Clear() error

	// Optional: for injected step that run inside the environment
	RunEpisode(agent *Agent) error
}