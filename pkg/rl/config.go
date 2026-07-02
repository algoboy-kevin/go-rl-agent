package rl

type DebugConfig struct {
	InspectBooks bool `yaml:"inspect_books"`
	RandomSeed   *int `yaml:"random_seed"`
}

type TrainingConfig struct {
	NThreads  int `yaml:"n_threads"`
	NSamples  int `yaml:"n_samples"`
	NEpisodes int `yaml:"n_episodes"`
}

type EvaluationConfig struct {
	NSamples       int  `yaml:"n_samples"`
	UseTrainSample bool `yaml:"use_train_sample"`
	RandomAgent    bool `yaml:"random_agent"`
}

type ExperimentConfig struct {
	DatabaseSize   int64 `yaml:"database_size"`
	BatchSize      int64 `yaml:"batch_size"`
	TrajectorySize int64 `yaml:"trajectory_size"`
}

type LearningConfig struct {
	MemorySize   int       `yaml:"memory_size"`
	NTilings     int       `yaml:"n_tilings"`
	NActions     int       `yaml:"n_actions"`
	Algorithm    string    `yaml:"algorithm"`
	GroupWeights []float64 `yaml:"group_weights"`
	GroupSplits  []int     `yaml:"group_splits"`
	Gamma        float64   `yaml:"gamma"`
	Lambda       float64   `yaml:"lambda"`
	Omega        float64   `yaml:"omega"`
	AlphaStart   float64   `yaml:"alpha_start"`
	AlphaFloor   float64   `yaml:"alpha_floor"`
	Beta         float64   `yaml:"beta"`
	RandomInit   bool      `yaml:"random_init"`
}

type PolicyConfig struct {
	Type           string  `yaml:"type"`
	EpsInit        float64 `yaml:"eps_init"`
	EpsFloor       float64 `yaml:"eps_floor"`
	EpsT           int     `yaml:"eps_T"`
	SpreadLookback int     `yaml:"spread_lookback"`
}

type LoggingConfig struct {
	LogLearning bool `yaml:"log_learning"`
	LogBacktest bool `yaml:"log_backtest"`
	MaxSize     int  `yaml:"max_size"`
}

type ModelConfig struct {
	LoadLastCheckpoint bool   `yaml:"load_last_checkpoint"`
	SaveEverySteps     int    `yaml:"save_every_nstep"`
	Name               string `yaml:"name"`
}

type RLTrainingConfig struct {
	Training   TrainingConfig   `yaml:"training"`
	Evaluation EvaluationConfig `yaml:"evaluation"`
	Experiment ExperimentConfig `yaml:"experiment"`
	Learning   LearningConfig   `yaml:"learning"`
	Policy     PolicyConfig     `yaml:"policy"`
	Model      ModelConfig      `yaml:"model"`
	Debug      DebugConfig      `yaml:"debug"`
	Logging    LoggingConfig    `yaml:"logging"`
	OutputDir  string           `yaml:"output_dir"`
}
