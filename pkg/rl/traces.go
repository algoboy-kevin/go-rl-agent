package rl

import (
	"math"
)

// Traces manages eligibility traces for reinforcement learning
type Traces struct {
	memorySize int
	nTilings   int
	nActions   int

	tolerance      float64
	nonZeroCount   int
	eligibility    []float64
	nonZeroTraces  []int
	inverseIndices []int
}

const (
	initialTolerance   = 0.01
	MAX_NONZERO_TRACES = 100000
)

// NewTraces creates a new Traces instance
func NewTraces(memorySize, nTilings, nActions int) *Traces {
	t := &Traces{
		memorySize:     memorySize,
		nTilings:       nTilings,
		nActions:       nActions,
		tolerance:      initialTolerance,
		eligibility:    make([]float64, memorySize),
		nonZeroTraces:  make([]int, MAX_NONZERO_TRACES),
		inverseIndices: make([]int, memorySize),
	}

	// Initialize inverse indices to -1 (not present)
	for i := range t.inverseIndices {
		t.inverseIndices[i] = -1
	}

	return t
}

// Decay reduces all traces by the given rate
func (t *Traces) Decay(rate float64) {
	for i := t.nonZeroCount - 1; i >= 0; i-- {
		feature := t.nonZeroTraces[i]
		t.eligibility[feature] *= rate

		if math.Abs(t.eligibility[feature]) < t.tolerance {
			t.clearExisting(feature, i)
		}
	}
}

// Update updates traces for a state-action pair
func (t *Traces) Update(state *State, action int) {
	for a := 0; a < t.nActions; a++ {
		features := state.GetFeatures(a)

		if a != action {
			// Clear traces for other actions (first group only)
			for til := range t.nTilings {
				t.Clear(features[til])
			}

		} else {
			// Set traces for current action (first group only)
			for til := range t.nTilings {
				t.Set(features[til], 1.0)
			}

		}
	}
}

// Set sets the trace value for a feature
func (t *Traces) Set(feature int, value float64) {
	if t.inverseIndices[feature] != -1 {
		// Feature already has a trace - just update value
		t.eligibility[feature] = value
		return
	}

	if math.Abs(value) < t.tolerance {
		// Value below tolerance - skip
		return
	}

	// Increase tolerance until we have space for new trace
	for t.nonZeroCount >= MAX_NONZERO_TRACES {
		t.increaseTolerance()
	}

	// Add new feature to non-zero traces
	t.eligibility[feature] = value
	t.nonZeroTraces[t.nonZeroCount] = feature
	t.inverseIndices[feature] = t.nonZeroCount
	t.nonZeroCount++
}

// Clear removes the trace for a feature
func (t *Traces) Clear(feature int) {
	if t.eligibility[feature] == 0.0 {
		return
	}

	if index := t.inverseIndices[feature]; index != -1 {
		t.clearExisting(feature, index)
	}
	t.eligibility[feature] = 0.0
}

// Get returns the trace value for a feature
func (t *Traces) Get(feature int) float64 {
	return t.eligibility[feature]
}

// Indices returns the list of features with non-zero traces
func (t *Traces) Indices() []int {
	return t.nonZeroTraces[:t.nonZeroCount]
}

// clearExisting removes a feature from the non-zero traces
func (t *Traces) clearExisting(feature, index int) {
	// Swap with last element
	lastFeature := t.nonZeroTraces[t.nonZeroCount-1]
	t.nonZeroTraces[index] = lastFeature
	t.inverseIndices[lastFeature] = index

	// Clear the removed feature
	t.inverseIndices[feature] = -1
	t.nonZeroCount--
}

// increaseTolerance removes more traces by increasing tolerance
func (t *Traces) increaseTolerance() {
	t.tolerance *= 1.1
	for i := t.nonZeroCount - 1; i >= 0; i-- {
		feature := t.nonZeroTraces[i]
		if math.Abs(t.eligibility[feature]) < t.tolerance {
			t.clearExisting(feature, i)
		}
	}
}
