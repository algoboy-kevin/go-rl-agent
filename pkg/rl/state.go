package rl

import (
	"fmt"
	"log/slog"
	"math"
)

// State represents the environment state with tile-coded features
type State struct {
	// Psudo-constant
	MemorySize  int
	NTilings    int
	NActions    int
	GroupSplits []int // variable boundaries for groups 0 and 1; e.g. [4,2] = group0 uses first 4 vars, group1 uses next 2 vars, group2 uses all

	StateVars []float64
	Features  [][]int // Features[action] contains 3*NTilings tile indices

	Potential float64
}

// NewStateInstance creates a new State instance
func NewStateInstance(memorySize, nActions, nTilings int, groupSplits []int) *State {
	// Initialize features matrix: nActions rows, 3*nTilings columns
	features := make([][]int, nActions)
	for i := range features {
		features[i] = make([]int, 3*nTilings)
	}

	return &State{
		MemorySize:  memorySize,
		NTilings:    nTilings,
		NActions:    nActions,
		GroupSplits: groupSplits,
		Features:    features,
		Potential:   0.0,
	}
}

func NewStateFromConfig(config RLTrainingConfig) *State {
	return NewStateInstance(config.Learning.MemorySize, config.Learning.NActions, config.Learning.NTilings, config.Learning.GroupSplits)
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
func (s *State) NewStateFromEnv(env Environment) error {
	s.StateVars = nil
	stateVars := env.GetState()

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

// groupSizes returns the number of variables for groups 0 and 1.
// Group 2 always uses all variables.
func (s *State) groupSizes() (group0Count, group1Start, group1Count int) {
	switch {
	case len(s.GroupSplits) >= 2:
		group0Count = s.GroupSplits[0]
		group1Start = s.GroupSplits[0]
		group1Count = s.GroupSplits[1]
	case len(s.GroupSplits) == 1:
		group0Count = s.GroupSplits[0]
		group1Start = s.GroupSplits[0]
		group1Count = len(s.StateVars) - s.GroupSplits[0]
	default:
		// Default: first 3 vars for group 0, rest for group 1
		group0Count = 3
		group1Start = 3
		group1Count = len(s.StateVars) - 3
	}
	return
}

func (s *State) populateFeatures() {
	g0Count, g1Start, g1Count := s.groupSizes()

	for action := 0; action < s.NActions; action++ {
		line := fmt.Sprintf("[%d] [ ", action)
		for _, a := range s.StateVars {
			line += fmt.Sprintf("%.2f ", a)
		}
		line += ("]")
		slog.Debug(line)

		// First feature group: first g0Count variables (e.g. agent-state)
		Tiles(
			s.Features[action][:s.NTilings],
			s.NTilings,
			s.MemorySize,
			s.StateVars[:g0Count],
			g0Count,
			[]int{action},
			1,
		)

		// Second feature group: g1Count variables starting at g1Start (e.g. market-state)
		Tiles(
			s.Features[action][s.NTilings:2*s.NTilings],
			s.NTilings,
			s.MemorySize,
			s.StateVars[g1Start:g1Start+g1Count],
			g1Count,
			[]int{s.NActions + action},
			1,
		)

		// Third feature group: all variables (full-state)
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
