package trainer

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"sync"

	"github.com/algoboy-kevin/go-rl-agent/pkg/rl"
	"github.com/algoboy-kevin/go-rl-agent/pkg/runner"
)

// EpisodeInfo is passed to OnProgress after each completed episode.
type EpisodeInfo struct {
	CurrentEpisode int
	TotalEpisodes  int
	WorkerID       int
	Reward         float64 // 0 if env doesn't provide it (caller can enrich)
	Epsilon        float64 // from agent.Policy.Descr()
	Alpha          float64 // from agent.Alpha
}

type Trainer struct {
	nThreads       int
	nEvalEpisodes  int
	nTrainEpisodes int
	currentEpisode int
	episodeMutex   sync.Mutex
	wg             sync.WaitGroup
	isTesting      bool
	useVisual      bool

	agent  *rl.Agent
	config rl.RLTrainingConfig

	onSpawnEnv func() (rl.Environment, error)

	OnProgress func(info EpisodeInfo)
}

func NewTrainer(
	config rl.RLTrainingConfig,
	onSpawnEnv func() (rl.Environment, error),
) (*Trainer, error) {
	return &Trainer{
		onSpawnEnv: onSpawnEnv,
		config:     config,
		nThreads:   1, // Default to single thread
	}, nil
}

func (t *Trainer) StartTraining(ctx context.Context) error {
	if err := t.initialize(); err != nil {
		return fmt.Errorf("initialization failed: %w", err)
	}

	slog.Info("trainer: initialized",
		"train_episodes", t.nTrainEpisodes,
		"eval_episodes", t.nEvalEpisodes,
		"threads", t.nThreads)

	// Start training threads.
	t.wg.Add(t.nThreads)
	for i := 0; i < t.nThreads; i++ {
		go func(id int) {
			slog.Info("trainer: start worker", "id", id)
			if err := t.trainWorker(ctx, id); err != nil {
				slog.Error("failed to run worker", "err", err)
			}
		}(i)
	}

	// Wait for workers or cancellation.
	done := make(chan struct{})
	go func() {
		t.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *Trainer) initialize() error {
	// Seed random number generator
	seed := rand.Int63()

	// Load configuration
	t.nThreads = t.config.Training.NThreads
	t.nTrainEpisodes = t.config.Training.NEpisodes
	t.nEvalEpisodes = t.config.Evaluation.NSamples

	// Initialize agent
	agent, err := rl.NewSARSA(t.config, seed)
	if err != nil {
		return fmt.Errorf("failed to spawn agent: %w", err)
	}

	t.agent = agent

	// Set agent's policy
	policy := rl.NewPolicyByType(t.config, rand.Int63())
	t.agent.SetPolicy(policy)

	return nil
}

func (t *Trainer) trainWorker(ctx context.Context, id int) error {
	defer t.wg.Done()

	if t.onSpawnEnv == nil {
		return fmt.Errorf("spawn env handler is missing")
	}

	env, err := t.onSpawnEnv()
	if err != nil {
		return err
	}

	// Each worker gets its own agent view sharing Theta (Hogwild!).
	// Traces, Rand, and Policy are independent per worker so there is
	// no mutex contention on the hot path — only Theta writes may race,
	// which the Hogwild! paper proves converges for sparse updates.
	workerAgent := t.agent.NewWorkerView(t.config, rand.Int63())

	experiment, err := runner.NewLearner(t.config, env, workerAgent)
	if err != nil {
		return err
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		episode := t.getNextEpisode()
		env.UpdateEpisodeCount(episode)

		if episode > t.nTrainEpisodes {
			break
		}

		if _, err = experiment.RunEpisode(ctx, workerAgent); err != nil {
			return err
		}

		if t.OnProgress != nil {
			t.OnProgress(EpisodeInfo{
				CurrentEpisode: episode,
				TotalEpisodes:  t.nTrainEpisodes,
				WorkerID:       id,
				Reward:         0, // library doesn't know about EpisodeStatistic
				Epsilon:        t.agent.Policy.Descr(),
				Alpha:          t.agent.Alpha,
			})
		}
	}

	return nil
}

func (t *Trainer) getNextEpisode() int {
	t.episodeMutex.Lock()
	defer t.episodeMutex.Unlock()

	t.currentEpisode++
	t.agent.NEpisode++
	return t.currentEpisode
}

func (t *Trainer) EvaluationInit() (runner.Runner, error) {
	err := t.initialize()
	if err != nil {
		return nil, err
	}

	t.agent.GoGreedy()
	t.currentEpisode = 0

	if t.onSpawnEnv == nil {
		return nil, fmt.Errorf("spawn env handler is missing")
	}

	env, err := t.onSpawnEnv()
	if err != nil {
		return nil, err
	}

	experiment, err := runner.NewTester(t.config, env, t.agent)
	if err != nil {
		return nil, err
	}

	t.currentEpisode++

	err = experiment.EpisodeInit()
	if err != nil {
		return nil, err
	}

	return experiment, nil
}

func (t *Trainer) RunEvaluation(ctx context.Context) error {
	err := t.initialize()
	if err != nil {
		return err
	}

	t.agent.GoGreedy()
	t.currentEpisode = 0

	env, err := t.onSpawnEnv()
	if err != nil {
		return err
	}

	// Training log
	experiment, err := runner.NewTester(t.config, env, t.agent)
	if err != nil {
		return err
	}

	for i := 0; i < 1; i++ {
		env.UpdateEpisodeCount(t.currentEpisode)
		_, err := experiment.RunEpisode(ctx, t.agent)
		if err != nil {
			return err
		}

		t.currentEpisode++
		// if done {
		// 	t.logEvaluationProgress(i, env)
		// }
	}

	return nil
}

// Agent returns the underlying trained agent. The shared Theta slice contains
// the accumulated weights from all Hogwild! workers.
func (t *Trainer) Agent() *rl.Agent {
	return t.agent
}

func (t *Trainer) SetTrainerToBacktester(useVisual bool) {
	t.isTesting = true
	t.useVisual = useVisual
}

func (t *Trainer) SetSpawnEnv(handler func() (rl.Environment, error)) {
	t.onSpawnEnv = handler
}
