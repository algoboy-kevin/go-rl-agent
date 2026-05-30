package rl

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// RewardMeasure defines the type of reward calculation
type AgentKind int

const (
	AgentNone AgentKind = iota
	Sarsa
)

// Agent base struct
type Agent struct {
	Kind AgentKind

	MemorySize int
	NTilings   int
	NActions   int

	GroupWeights []float64

	Traces *Traces

	AlphaStart float64
	AlphaFloor float64
	Omega      float64
	Alpha      float64

	Gamma  float64
	Lambda float64

	Rand   *rand.Rand
	Policy *Policy

	Theta []float64

	updateCounter int
	aggDelta      float64

	//For checkpoint
	NStep              int
	NEpisode           int
	SaveInterval       int
	SaveDirectory      string
	Name               string
	LoadLastCheckpoint bool
}

// NewAgent creates a new Agent instance
func NewAgent(config *RLTrainingConfig, seed int64) (*Agent, error) {
	var theta []float64
	var nstep int
	var neps int
	var memorySize int
	var nTilings int
	var nActions int
	var groupWeights []float64

	// Try loading checkpoint if configured
	if config.Model.LoadLastCheckpoint {
		directory := fmt.Sprintf("%s/checkpoints", config.OutputDir)
		path := findLatestCheckpoint(directory, config.Model.Name)
		if path != "" {
			saved, err := LoadModel(path)
			if err == nil {
				theta = saved.Theta
				nstep = saved.NStep
				neps = saved.NEpisode

				// Restore other saved values
				memorySize = saved.MemorySize
				nTilings = saved.NTilings
				nActions = saved.NActions
				groupWeights = saved.GroupWeights
			}
		}
	}

	// Fall back to default theta if not loaded
	if theta == nil {
		// fallback if no checkpoint loaded
		memorySize = config.Learning.MemorySize
		nTilings = config.Learning.NTilings
		nActions = config.Learning.NActions

		theta = make([]float64, memorySize)
		if config.Learning.RandomInit {
			r := rand.New(rand.NewSource(seed))
			for i := range theta {
				theta[i] = 2*r.Float64() - 1
			}
		}

		groupWeights = config.Learning.GroupWeights
		arrWeight := []float64{0, 0, 0}
		if equalSlices(groupWeights, arrWeight) {
			groupWeights = []float64{1.0 / 3, 1.0 / 3, 1.0 / 3}
		}
	}

	agent := &Agent{
		// Loadable params
		MemorySize:   memorySize,
		NTilings:     nTilings,
		NActions:     nActions,
		GroupWeights: groupWeights,
		Theta:        theta,

		Traces: NewTraces(
			memorySize,
			nTilings,
			nActions,
		),

		AlphaStart:         config.Learning.AlphaStart,
		AlphaFloor:         config.Learning.AlphaFloor,
		Omega:              config.Learning.Omega,
		Alpha:              config.Learning.AlphaStart,
		Gamma:              config.Learning.Gamma,
		Lambda:             config.Learning.Lambda,
		Rand:               rand.New(rand.NewSource(seed)),
		Policy:             NewBasePolicy(nActions, seed),
		NStep:              nstep,
		NEpisode:           neps,
		SaveInterval:       config.Model.SaveEverySteps,
		SaveDirectory:      config.OutputDir,
		Name:               config.Model.Name,
		LoadLastCheckpoint: config.Model.LoadLastCheckpoint,
	}
	return agent, nil
}

func (a *Agent) UpdateWeights(fromState *State, action int, reward float64, toState *State) float64 {
	switch a.Kind {
	case Sarsa:
		return a.updateWeightsSarsa(fromState, action, reward, toState)

	default:
		return 0.0
	}
}

// Action selects an action for the given state
func (a *Agent) Action(s *State) int {
	qs := make([]float64, a.NActions)
	for action := 0; action < a.NActions; action++ {
		qs[action] = a.GetQ(s, action)
	}
	return int(a.Policy.Sample(qs))
}

// GoGreedy switches the policy to a greedy policy
func (a *Agent) GoGreedy() {
	seed := rand.Int63()
	a.Policy = NewGreedy(a.NActions, seed)
}

// SetPolicy updates the agent's policy
func (a *Agent) SetPolicy(policy *Policy) {
	a.Policy = policy
}

// HandleTransition processes a state transition
func (a *Agent) HandleTransition(fromState *State, action int, reward float64, toState *State) {
	a.UpdateTraces(fromState, action)
	delta := a.UpdateWeights(fromState, action, reward, toState)
	a.aggDelta += math.Abs(delta)
	a.updateCounter++
	a.NStep++

	if a.updateCounter%1000 == 0 {
		// Logging omitted (originally used spdlog)
		a.aggDelta = 0.0
		a.updateCounter = 0
	}

	// Save model if needed. TODO: disable during prod
	if a.SaveInterval > 0 && a.NStep%a.SaveInterval == 0 {
		filename := fmt.Sprintf("%s_%d.bin", a.Name, a.NStep)
		directory := fmt.Sprintf("%s/checkpoints", a.SaveDirectory)
		if err := os.MkdirAll(directory, os.ModePerm); err != nil {
			panic(err)
		}

		err := a.SaveModel(directory, filename)
		if err != nil {
			panic(err)
		}
	}
}

func (a *Agent) SaveModel(directory, filename string) error {
	fullPath := filepath.Join(directory, filename)
	file, err := os.Create(fullPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	snapshot := SavedAgent{
		Theta:        a.Theta,
		MemorySize:   a.MemorySize,
		NTilings:     a.NTilings,
		NActions:     a.NActions,
		GroupWeights: a.GroupWeights,
		NStep:        a.NStep,
		NEpisode:     a.NEpisode,
	}
	return encoder.Encode(snapshot)
}

// HandleTerminal resets traces and adjusts learning rate at episode end
func (a *Agent) HandleTerminal(episode int) {
	a.Traces.Decay(0.0)
	a.Alpha = math.Max(a.AlphaFloor, a.AlphaStart*math.Pow(a.Omega, float64(episode)))
	a.Policy.HandleTerminal(uint(episode))
}

// UpdateTraces updates eligibility traces
func (a *Agent) UpdateTraces(fromState *State, action int) {
	a.Traces.Decay(a.Gamma * a.Lambda)
	a.Traces.Update(fromState, action)
}

// GetQ computes Q-value for a state-action pair
func (a *Agent) GetQ(state *State, action int) float64 {
	features := state.GetFeatures(action)
	q := 0.0

	w0 := a.GroupWeights[0]
	for i := range a.NTilings {
		idx := features[i]
		q += w0 * a.Theta[idx]
	}

	w1 := a.GroupWeights[1]
	for i := a.NTilings; i < 2*a.NTilings; i++ {
		idx := features[i]
		q += w1 * a.Theta[idx]
	}

	w2 := a.GroupWeights[2]
	for i := 2 * a.NTilings; i < 3*a.NTilings; i++ {
		idx := features[i]
		q += w2 * a.Theta[idx]
	}

	return q
}

// UpdateQ updates weights using eligibility traces
func (a *Agent) UpdateQ(update float64) {
	scaledUpdate := update / float64(a.NTilings)
	for _, idx := range a.Traces.Indices() {
		trace := a.Traces.Get(idx)
		a.Theta[idx] += scaledUpdate * trace
	}
}

// NewSARSA creates a new SARSA agent
func NewSARSA(config *RLTrainingConfig, seed int64) (*Agent, error) {
	baseAgent, err := NewAgent(config, seed)
	if err != nil {
		return nil, err
	}
	baseAgent.Kind = Sarsa
	return baseAgent, nil
}

// UpdateWeights computes SARSA weight updates
func (a *Agent) updateWeightsSarsa(fromState *State, action int, reward float64, toState *State) float64 {
	q1 := a.GetQ(fromState, action)
	nextAction := a.Action(toState)
	q2 := a.GetQ(toState, nextAction)
	F := a.Gamma*toState.GetPotential() - fromState.GetPotential()
	delta := reward + F + a.Gamma*q2 - q1

	a.UpdateQ(a.Alpha * delta)
	return delta
}

func LoadModel(path string) (*SavedAgent, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model SavedAgent
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return nil, err
	}
	return &model, nil
}

// findLatestCheckpoint returns the path of the most recently saved checkpoint
// for the given model name, or empty string if none found.
func findLatestCheckpoint(directory, modelName string) string {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return ""
	}

	var latest string
	var latestStep int = -1
	prefix := modelName + "_"

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".bin") {
			continue
		}
		stepStr := strings.TrimSuffix(strings.TrimPrefix(name, prefix), ".bin")
		var step int
		if _, err := fmt.Sscanf(stepStr, "%d", &step); err != nil {
			continue
		}
		if step > latestStep {
			latestStep = step
			latest = filepath.Join(directory, name)
		}
	}

	// Sort by modification time as tiebreaker
	if latestStep == -1 {
		return ""
	}
	return latest
}

// equalSlices returns true if two float64 slices have the same length and
// all elements are equal.
func equalSlices(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
