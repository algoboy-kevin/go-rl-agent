package rl

import (
	"math"
	"math/rand"

)

// RewardMeasure defines the type of reward calculation
type PolicyType int

const (
	None PolicyType = iota
	Random
	Greedy
	EpsilonGreedy
	Boltzmann
)

// Policy defines the interface for reinforcement learning policies.
type Policy struct {
	Kind PolicyType

	nActions int
	rng      *rand.Rand

	//Epsilon
	eps      float64 // exploration rate
	epsT     uint
	epsInit  float64
	epsFloor float64

	//Boltzman
	tau        float64 // degree of randomness
	tauT       uint
	tauInit    float64
	tauFloor   float64
	probs      []float64
	cumulative []float64
}

func NewPolicyByType(config *RLTrainingConfig, seed int64) *Policy {
	switch config.Policy.Type {
	case "random":
		return NewRandom(config.Learning.NActions, seed)
	case "epsilon_greedy":
		return NewEpsilonGreedy(config.Learning.NActions, config.Policy.EpsInit, config.Policy.EpsFloor, uint(config.Policy.EpsT), seed)
	case "greedy":
		return NewGreedy(config.Learning.NActions, seed)
	case "boltzman":
		return NewBoltzmann(config.Learning.NActions, config.Policy.EpsInit, config.Policy.EpsFloor, uint(config.Policy.EpsT), seed)
	default:
		return NewBasePolicy(config.Learning.NActions, seed)
	}
}

// newBasePolicy initializes a new basePolicy.
func NewBasePolicy(nActions int, seed int64) *Policy {
	src := rand.NewSource(seed)
	return &Policy{
		nActions: nActions,
		rng:      rand.New(src),
	}
}

func NewRandom(nActions int, seed int64) *Policy {
	policy := NewBasePolicy(nActions, seed)
	policy.Kind = Random

	return policy
}

func NewGreedy(nActions int, seed int64) *Policy {
	policy := NewBasePolicy(nActions, seed)
	policy.Kind = Greedy
	return policy
}

func NewEpsilonGreedy(nActions int, eps, epsFloor float64, epsT uint, seed int64) *Policy {
	policy := NewGreedy(nActions, seed)
	policy.Kind = EpsilonGreedy
	policy.eps = eps
	policy.epsT = epsT
	policy.epsInit = eps
	policy.epsFloor = epsFloor

	return policy
}

func NewBoltzmann(nActions int, tau, tauFloor float64, tauT uint, seed int64) *Policy {
	policy := NewBasePolicy(nActions, seed)
	policy.Kind = Boltzmann
	policy.tau = tau
	policy.tauT = tauT
	policy.tauInit = tau
	policy.tauFloor = tauFloor
	policy.probs = make([]float64, nActions)
	policy.cumulative = make([]float64, nActions)

	return policy
}

func (r *Policy) GreedySample(qs []float64) uint {
	argmax := 0
	nTies := 1

	for a := 1; a < r.nActions; a++ {
		if qs[a] > qs[argmax] {
			argmax = a
			nTies = 1
		} else if qs[a] == qs[argmax] {
			nTies++
			if r.rng.Intn(nTies) == 0 {
				argmax = a
			}
		}
	}
	return uint(argmax) // Convert to uint
}

func (r *Policy) Sample(qs []float64) uint {
	switch r.Kind {
	case Random:
		return uint(r.rng.Intn(r.nActions)) // Convert to uint
	case Greedy:
		return r.GreedySample(qs)
		
	case EpsilonGreedy:
		if r.rng.Float64() < r.eps {
			return uint(r.rng.Intn(r.nActions)) // Fixed: convert to uint
		}

		return r.GreedySample(qs)

	case Boltzmann:
		// Compute unnormalized probabilities and sum (Z)
		Z := 0.0
		for a := 0; a < r.nActions; a++ {
			r.probs[a] = math.Exp(qs[a] / r.tau)
			Z += r.probs[a]
		}

		// Build cumulative distribution
		acc := 0.0
		for a := 0; a < r.nActions; a++ {
			acc += r.probs[a] / Z
			r.cumulative[a] = acc
		}

		// Sample using cumulative distribution
		rng := r.rng.Float64()
		for a := 0; a < r.nActions; a++ {
			if rng < r.cumulative[a] {
				return uint(a) // Convert to uint
			}
		}
		return uint(r.nActions - 1) // Convert to uint
	}

	return 0// Convert to uint
}

func (r *Policy) Descr() float64 { 
	switch r.Kind {
	case Random:
		return 0.0

	case Greedy:
		return 0.0

	case EpsilonGreedy:
		return r.eps

	case Boltzmann:
		return r.tau
	}

	return 0.0
}

func (r *Policy) HandleTerminal(episode uint) {
	switch r.Kind {
	case Random:
		return 

	case Greedy:
		return 

	case EpsilonGreedy:
		if r.epsT == 0 {
			return
		}
		ratio := float64(episode) / float64(r.epsT)
		r.eps = r.epsInit * math.Pow(r.epsFloor/r.epsInit, ratio)

	case Boltzmann:
		if r.tauT == 0 {
			return
		}
		ratio := float64(episode) / float64(r.tauT)
		r.tau = r.tauInit * math.Pow(r.tauFloor/r.tauInit, ratio)
	}
}
