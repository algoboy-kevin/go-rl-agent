package rl

import "math"

// CollisionTable handles collision tracking (simplified version)
type CollisionTable struct {
	data   []int64
	size   int
	safe   bool
	calls  int64
	clears int64
	colls  int64
}

// NewCollisionTable creates a new collision table
func NewCollisionTable(size int, safe bool) *CollisionTable {
	// Validate size is power of 2
	tmp := size
	for tmp > 2 {
		if tmp%2 != 0 {
			panic("Size of collision table must be power of 2")
		}
		tmp /= 2
	}

	ct := &CollisionTable{
		data: make([]int64, size),
		size: size,
		safe: safe,
	}
	ct.Reset()
	return ct
}

// Reset clears the collision table
func (ct *CollisionTable) Reset() {
	for i := range ct.data {
		ct.data[i] = -1
	}
	ct.calls = 0
	ct.clears = 0
	ct.colls = 0
}

// TilesWithCollisionTable computes tiles using collision table
func (ct *CollisionTable) TilesWithCollisionTable(
	tiles []int, // Output tile indices
	numTilings int, // Number of tilings
	floats []float64, // Float variables
	numFloats int, // Number of float variables
	ints []int, // Int variables
	numInts int, // Number of int variables
) {
	coordinates := make([]int, numFloats+numInts+1)
	qState := make([]int, numFloats)
	base := make([]int, numFloats)

	// Copy integer variables to coordinates
	for i := range numInts {
		coordinates[numFloats+1+i] = ints[i]
	}

	// Quantize state to integers
	for i := range numFloats {
		qState[i] = int(math.Floor(floats[i] * float64(numTilings)))
		base[i] = 0
	}

	// Compute the tile numbers
	for j := range numTilings {
		// Loop over each relevant dimension
		for i := range numFloats {
			// Find coordinates of activated tile in tiling space
			if qState[i] >= base[i] {
				coordinates[i] = qState[i] - ((qState[i] - base[i]) % numTilings)
			} else {
				coordinates[i] = qState[i] + 1 + ((base[i] - qState[i] - 1) % numTilings) - numTilings
			}
			// Compute displacement of next tiling in quantized space
			base[i] += 1 + (2 * i)
		}
		// Add additional index for tiling
		coordinates[numFloats] = j

		tiles[j] = ct.hash(coordinates, numFloats+numInts+1)
	}
}

// hash computes hash with collision handling
func (ct *CollisionTable) hash(ints []int, numInts int) int {
	ct.calls++
	j := hashUNH(ints, numInts, ct.size, 449)
	ccheck := hashUNH(ints, numInts, MaxLongInt, 457)

	if int64(ccheck) == ct.data[j] {
		ct.clears++
	} else if ct.data[j] == -1 {
		ct.clears++
		ct.data[j] = int64(ccheck)
	} else if !ct.safe {
		ct.colls++
	} else {
		h2 := 1 + 2*hashUNH(ints, numInts, MaxLongInt/4, 449)
		i := 0
		for {
			ct.colls++
			j = (j + h2) % ct.size
			i++
			if i > ct.size {
				panic("Collision table out of memory")
			}
			if int64(ccheck) == ct.data[j] {
				break
			}
			if ct.data[j] == -1 {
				ct.data[j] = int64(ccheck)
				break
			}
		}
	}
	return j
}
