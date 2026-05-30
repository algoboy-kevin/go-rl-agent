package rl

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/algoboy-kevin/go-rl-agent/pkg/environment"
)

// State represents the environment state with tile-coded features
type State struct {
	// Psudo-constant
	MemorySize int
	NTilings   int
	NActions   int

	StateVars []float64
	Features  [][]int // Features[action] contains 3*NTilings tile indices

	Potential float64
}

// NewStateInstance creates a new State instance
func NewStateInstance(memorySize, nActions, nTilings int) *State {
	// Initialize features matrix: nActions rows, 3*nTilings columns
	features := make([][]int, nActions)
	for i := range features {
		features[i] = make([]int, 3*nTilings)
	}

	return &State{
		MemorySize: memorySize,
		NTilings:   nTilings,
		NActions:   nActions,
		Features:   features,
		Potential:  0.0,
	}
}

func NewStateFromConfig(config *RLTrainingConfig) *State {
	return NewStateInstance(config.Learning.MemorySize, config.Learning.NActions, config.Learning.NTilings)
}

// Initialise resets the state
func (s *State) Initialise() {
	s.StateVars = nil
	s.Potential = 0.0
	for a := 0; a < s.NActions; a++ {
		s.Features[a] = make([]int, 3*s.NTilings)
	}
}

// NewState sets state variables and calculates features
func (s *State) NewStateFromEnv(env environment.RLEnvironment) error {
	s.StateVars = nil
	stateVars, err := env.GetState()
	if err != nil {
		return err
	}

	s.StateVars = stateVars

	s.populateFeatures()
	s.Potential = 0.0
	return nil
}

// NewState sets state variables and calculates features
func (s *State) NewState(stateVars []float64, potential float64) {
	s.StateVars = stateVars
	s.Potential = potential
	s.populateFeatures()
}

func (s *State) populateFeatures() {
	for action := 0; action < s.NActions; action++ {
		// First feature group: first 3 variables
		line := fmt.Sprintf("[%d] [ ", action)

		for _, a := range s.StateVars {
			line += fmt.Sprintf("%.2f ", a)
		}
		line += ("]")
		slog.Debug(line)

		Tiles(
			s.Features[action][:s.NTilings],
			s.NTilings,
			s.MemorySize,
			s.StateVars[:3],
			3,
			[]int{action},
			1,
		)

		// Second feature group: variables starting from index 3
		Tiles(
			s.Features[action][s.NTilings:2*s.NTilings],
			s.NTilings,
			s.MemorySize,
			s.StateVars[3:],
			len(s.StateVars)-3,
			[]int{s.NActions + action},
			1,
		)

		// Third feature group: all variables
		Tiles(
			s.Features[action][2*s.NTilings:],
			s.NTilings,
			s.MemorySize,
			s.StateVars,
			len(s.StateVars),
			[]int{2*s.NActions + action},
			1,
		)
	}
	slog.Debug("\n\n")
}

// GetFeatures returns tile indices for a specific action
func (s *State) GetFeatures(action int) []int {
	return s.Features[action]
}

// GetPotential returns the state's potential
func (s *State) GetPotential() float64 {
	return s.Potential
}

// ToVector returns the raw state variables
func (s *State) ToVector() []float64 {
	return s.StateVars
}

// PrintState displays state information (for debugging)
func (s *State) PrintState() {
	for i, v := range s.StateVars {
		scaled := math.Floor(v * float64(s.NTilings))
		fmt.Printf("[%d] %.4f \t\t-> %.0f\n", i, v, scaled)
	}
	fmt.Println()
}
