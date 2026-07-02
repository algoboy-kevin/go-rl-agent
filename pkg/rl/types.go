package rl

// SavedAgent is the serializable snapshot of an Agent's learned parameters.
// Used by gob encoding for checkpoint save/load.
type SavedAgent struct {
	Theta        []float64
	MemorySize   int
	NTilings     int
	NActions     int
	GroupWeights []float64
	GroupSplits  []int
	NStep        int
	NEpisode     int
}
