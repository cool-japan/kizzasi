//! # Neural Architecture Search (NAS) for SSM Hyperparameters
//!
//! This module implements three complementary search strategies for discovering
//! optimal State Space Model (SSM) architectures without full training:
//!
//! - **[`RandomArchSearcher`]**: Samples `n_candidates` architectures uniformly at
//!   random from the search space and returns the one with the highest proxy score.
//! - **[`EvolutionarySearcher`]**: Runs a mu+lambda evolutionary algorithm with
//!   tournament selection, mutation, and crossover for `n_generations` generations.
//! - **[`GridSearcher`]**: Exhaustively enumerates every combination in the search
//!   space, capped at `max_candidates` to keep runtime bounded.
//!
//! All searchers use a training-free **proxy score**
//! (`log2(capacity) / log2(params + 1)`) that balances model capacity against
//! parameter count, so no GPU or dataset is required during search.
//!
//! ## Determinism
//!
//! Random and evolutionary searchers accept a `seed` via [`ArchSearchConfig`] and
//! drive all randomness through a built-in xorshift64 PRNG — no external `rand`
//! crate is needed.
//!
//! ## Example
//!
//! ```rust
//! use kizzasi_model::arch_search::{ArchSearchSpace, ArchSearchConfig, search_best_arch};
//!
//! let space = ArchSearchSpace::default();
//! let config = ArchSearchConfig::default();
//! let result = search_best_arch(space, config).expect("search failed");
//! println!("Best arch: {:?} score={:.4}", result.best_candidate, result.best_score);
//! ```

use crate::error::{ModelError, ModelResult};

// ─────────────────────────────────────────────────────────────────────────────
// Search-space definition
// ─────────────────────────────────────────────────────────────────────────────

/// The discrete hyperparameter space explored during architecture search.
///
/// Each field is a list of allowed values for the corresponding hyperparameter.
/// Searchers sample from these lists to construct [`ArchCandidate`]s.
#[derive(Debug, Clone)]
pub struct ArchSearchSpace {
    /// Hidden dimension options (e.g. `[64, 128, 256, 512]`)
    pub d_model_options: Vec<usize>,
    /// State dimension options for SSMs (e.g. `[8, 16, 32, 64]`)
    pub d_state_options: Vec<usize>,
    /// Number of stacked layers (e.g. `[2, 4, 6, 8, 12]`)
    pub n_layers_options: Vec<usize>,
    /// Architecture family strings (e.g. `["mamba", "rwkv", "s4", "transformer"]`)
    pub model_types: Vec<String>,
    /// Expansion factor inside each layer (e.g. `[1.0, 2.0, 4.0]`)
    pub expand_factor_options: Vec<f32>,
    /// Dropout probability options (e.g. `[0.0, 0.1, 0.2]`)
    pub dropout_options: Vec<f32>,
}

impl Default for ArchSearchSpace {
    fn default() -> Self {
        Self {
            d_model_options: vec![64, 128, 256, 512],
            d_state_options: vec![8, 16, 32, 64],
            n_layers_options: vec![2, 4, 6, 8, 12],
            model_types: vec![
                "mamba".to_string(),
                "rwkv".to_string(),
                "s4".to_string(),
                "transformer".to_string(),
            ],
            expand_factor_options: vec![1.0, 2.0, 4.0],
            dropout_options: vec![0.0, 0.1, 0.2],
        }
    }
}

impl ArchSearchSpace {
    /// Returns `true` when every option list has at least one entry.
    pub fn is_valid(&self) -> bool {
        !self.d_model_options.is_empty()
            && !self.d_state_options.is_empty()
            && !self.n_layers_options.is_empty()
            && !self.model_types.is_empty()
            && !self.expand_factor_options.is_empty()
            && !self.dropout_options.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture candidate
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete architecture specification drawn from an [`ArchSearchSpace`].
#[derive(Debug, Clone)]
pub struct ArchCandidate {
    /// Hidden dimension.
    pub d_model: usize,
    /// State dimension (SSM-specific; used differently for attention models).
    pub d_state: usize,
    /// Number of stacked model layers.
    pub n_layers: usize,
    /// Architecture family identifier (`"mamba"`, `"rwkv"`, `"s4"`, `"transformer"`, …).
    pub model_type: String,
    /// Per-layer inner expansion factor.
    pub expand_factor: f32,
    /// Dropout probability applied between layers.
    pub dropout: f32,
}

impl ArchCandidate {
    /// Estimates the total parameter count of this architecture.
    ///
    /// Each architecture family uses a family-specific formula:
    ///
    /// | Family        | Per-layer formula |
    /// |---------------|-------------------|
    /// | `mamba`       | `2·d_model·(d_model·E + d_state·2) + d_model` |
    /// | `transformer` | `4·d_model²·2` |
    /// | `rwkv`        | `5·d_model²` |
    /// | `s4`          | `2·d_model·d_state + d_model²` |
    /// | _(default)_   | `2·d_model²` |
    ///
    /// where `E = expand_factor as usize`.
    pub fn param_count(&self) -> usize {
        let expand = self.expand_factor as usize;
        let per_layer: usize = match self.model_type.as_str() {
            "mamba" => 2 * self.d_model * (self.d_model * expand + self.d_state * 2) + self.d_model,
            "transformer" => 4 * self.d_model * self.d_model + 4 * self.d_model * self.d_model,
            "rwkv" => 5 * self.d_model * self.d_model,
            "s4" => 2 * self.d_model * self.d_state + self.d_model * self.d_model,
            _ => 2 * self.d_model * self.d_model,
        };
        self.n_layers * per_layer
    }

    /// Training-free proxy score.
    ///
    /// `score = log2(d_model · d_state · n_layers) / log2(param_count + 1)`
    ///
    /// Higher is better: a high score indicates strong representational capacity
    /// relative to the number of parameters.
    pub fn proxy_score(&self) -> f64 {
        let capacity = (self.d_model as f64 * self.d_state as f64 * self.n_layers as f64).log2();
        let params = (self.param_count() as f64 + 1.0).log2();
        capacity / params.max(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters that govern how a searcher explores the [`ArchSearchSpace`].
#[derive(Debug, Clone)]
pub struct ArchSearchConfig {
    /// Total number of candidates sampled by [`RandomArchSearcher`].
    pub n_candidates: usize,
    /// Number of evolutionary generations performed by [`EvolutionarySearcher`].
    pub n_generations: usize,
    /// Tournament pool size used during parent selection.
    pub tournament_size: usize,
    /// Per-gene mutation probability (0 – 1).
    pub mutation_prob: f64,
    /// Initial population size for [`EvolutionarySearcher`].
    pub population_size: usize,
    /// PRNG seed — fix this for reproducible runs.
    pub seed: u64,
}

impl Default for ArchSearchConfig {
    fn default() -> Self {
        Self {
            n_candidates: 50,
            n_generations: 20,
            tournament_size: 5,
            mutation_prob: 0.2,
            population_size: 30,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search result
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of a completed architecture search.
#[derive(Debug)]
pub struct ArchSearchResult {
    /// The architecture candidate with the highest proxy score.
    pub best_candidate: ArchCandidate,
    /// Proxy score of `best_candidate`.
    pub best_score: f64,
    /// Every evaluated `(candidate, score)` pair in evaluation order.
    pub all_candidates: Vec<(ArchCandidate, f64)>,
    /// Total number of proxy-score evaluations performed.
    pub n_evaluations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic PRNG (xorshift64) — no external `rand` crate
// ─────────────────────────────────────────────────────────────────────────────

/// A fast, deterministic pseudo-random number generator based on the xorshift64
/// algorithm.  All randomness in this module is driven exclusively through this
/// type, making every search reproducible given the same seed.
struct Prng {
    state: u64,
}

impl Prng {
    /// Initialise the PRNG.  A seed of `0` is replaced by `1` to avoid the
    /// degenerate all-zeros state.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Advance the state and return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a pseudo-random `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for the mantissa (IEEE-754 double precision)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniformly choose one element from `options`.
    ///
    /// Returns `None` when `options` is empty.
    fn choice<T: Clone>(&mut self, options: &[T]) -> Option<T> {
        if options.is_empty() {
            return None;
        }
        let idx = (self.next_u64() as usize) % options.len();
        Some(options[idx].clone())
    }

    /// Fisher-Yates in-place shuffle.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            items.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: pick the best from a scored list
// ─────────────────────────────────────────────────────────────────────────────

fn best_in(scored: &[(ArchCandidate, f64)]) -> Option<(ArchCandidate, f64)> {
    scored
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, s)| (c.clone(), *s))
}

// ─────────────────────────────────────────────────────────────────────────────
// Random searcher
// ─────────────────────────────────────────────────────────────────────────────

/// Samples `config.n_candidates` architectures uniformly at random and
/// returns the one with the highest proxy score.
pub struct RandomArchSearcher {
    space: ArchSearchSpace,
    config: ArchSearchConfig,
}

impl RandomArchSearcher {
    /// Construct a new random searcher.
    pub fn new(space: ArchSearchSpace, config: ArchSearchConfig) -> Self {
        Self { space, config }
    }

    /// Run the random search and return the result.
    pub fn search(&self) -> ModelResult<ArchSearchResult> {
        if !self.space.is_valid() {
            return Err(ModelError::invalid_config(
                "ArchSearchSpace has empty option lists",
            ));
        }
        let mut prng = Prng::new(self.config.seed);
        let mut all_candidates: Vec<(ArchCandidate, f64)> =
            Vec::with_capacity(self.config.n_candidates);

        for _ in 0..self.config.n_candidates {
            let candidate = self.random_candidate(&mut prng)?;
            let score = candidate.proxy_score();
            all_candidates.push((candidate, score));
        }

        let (best_candidate, best_score) = best_in(&all_candidates).ok_or_else(|| {
            ModelError::invalid_config("no candidates were generated during random search")
        })?;

        Ok(ArchSearchResult {
            best_candidate,
            best_score,
            n_evaluations: all_candidates.len(),
            all_candidates,
        })
    }

    fn random_candidate(&self, prng: &mut Prng) -> ModelResult<ArchCandidate> {
        let d_model = prng
            .choice(&self.space.d_model_options)
            .ok_or_else(|| ModelError::invalid_config("d_model_options is empty"))?;
        let d_state = prng
            .choice(&self.space.d_state_options)
            .ok_or_else(|| ModelError::invalid_config("d_state_options is empty"))?;
        let n_layers = prng
            .choice(&self.space.n_layers_options)
            .ok_or_else(|| ModelError::invalid_config("n_layers_options is empty"))?;
        let model_type = prng
            .choice(&self.space.model_types)
            .ok_or_else(|| ModelError::invalid_config("model_types is empty"))?;
        let expand_factor = prng
            .choice(&self.space.expand_factor_options)
            .ok_or_else(|| ModelError::invalid_config("expand_factor_options is empty"))?;
        let dropout = prng
            .choice(&self.space.dropout_options)
            .ok_or_else(|| ModelError::invalid_config("dropout_options is empty"))?;

        Ok(ArchCandidate {
            d_model,
            d_state,
            n_layers,
            model_type,
            expand_factor,
            dropout,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evolutionary searcher
// ─────────────────────────────────────────────────────────────────────────────

/// Runs a mu+lambda evolutionary algorithm over the search space.
///
/// Each generation:
/// 1. Tournament-selects two parents from the population.
/// 2. Produces one offspring via single-point crossover.
/// 3. Applies per-gene mutation with probability `config.mutation_prob`.
/// 4. Evaluates the offspring's proxy score.
/// 5. Replaces the worst member of the population if the offspring is better.
pub struct EvolutionarySearcher {
    space: ArchSearchSpace,
    config: ArchSearchConfig,
}

impl EvolutionarySearcher {
    /// Construct a new evolutionary searcher.
    pub fn new(space: ArchSearchSpace, config: ArchSearchConfig) -> Self {
        Self { space, config }
    }

    /// Run the evolutionary search and return the result.
    pub fn search(&self) -> ModelResult<ArchSearchResult> {
        if !self.space.is_valid() {
            return Err(ModelError::invalid_config(
                "ArchSearchSpace has empty option lists",
            ));
        }

        let mut prng = Prng::new(self.config.seed);

        // ── 1. Initialise population ──────────────────────────────────────────
        let seeder = RandomArchSearcher::new(
            self.space.clone(),
            ArchSearchConfig {
                n_candidates: self.config.population_size,
                seed: prng.next_u64(),
                ..self.config.clone()
            },
        );
        let seed_result = seeder.search()?;
        let mut population: Vec<(ArchCandidate, f64)> = seed_result.all_candidates;
        let mut all_candidates: Vec<(ArchCandidate, f64)> = population.clone();

        // ── 2. Evolve ─────────────────────────────────────────────────────────
        for _gen in 0..self.config.n_generations {
            // Produce `population_size` offspring each generation
            let offspring_count = self.config.population_size;
            for _ in 0..offspring_count {
                let parent_a = self.tournament_select(&population, &mut prng);
                let parent_b = self.tournament_select(&population, &mut prng);
                let mut child = self.crossover(parent_a, parent_b, &mut prng)?;
                child = self.mutate(&child, &mut prng)?;
                let child_score = child.proxy_score();
                all_candidates.push((child.clone(), child_score));

                // Replace the worst individual in the population
                if let Some(worst_pos) = population
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        a.1 .1
                            .partial_cmp(&b.1 .1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    if child_score > population[worst_pos].1 {
                        population[worst_pos] = (child, child_score);
                    }
                }
            }
        }

        let (best_candidate, best_score) = best_in(&all_candidates).ok_or_else(|| {
            ModelError::invalid_config("evolutionary search produced no candidates")
        })?;

        Ok(ArchSearchResult {
            best_candidate,
            best_score,
            n_evaluations: all_candidates.len(),
            all_candidates,
        })
    }

    /// Tournament selection: pick `tournament_size` random contestants and
    /// return the one with the highest score.
    fn tournament_select<'a>(
        &self,
        pop: &'a [(ArchCandidate, f64)],
        prng: &mut Prng,
    ) -> &'a ArchCandidate {
        let size = self.config.tournament_size.min(pop.len()).max(1);
        let mut indices: Vec<usize> = (0..pop.len()).collect();
        prng.shuffle(&mut indices);
        let best_idx = indices[..size]
            .iter()
            .max_by(|&&a, &&b| {
                pop[a]
                    .1
                    .partial_cmp(&pop[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0);
        &pop[best_idx].0
    }

    /// Single-point crossover: each gene is drawn independently from either
    /// parent with equal probability.
    fn crossover(
        &self,
        a: &ArchCandidate,
        b: &ArchCandidate,
        prng: &mut Prng,
    ) -> ModelResult<ArchCandidate> {
        // Each gene comes from parent A or B based on a coin flip
        let d_model = if prng.next_f64() < 0.5 {
            a.d_model
        } else {
            b.d_model
        };
        let d_state = if prng.next_f64() < 0.5 {
            a.d_state
        } else {
            b.d_state
        };
        let n_layers = if prng.next_f64() < 0.5 {
            a.n_layers
        } else {
            b.n_layers
        };
        let model_type = if prng.next_f64() < 0.5 {
            a.model_type.clone()
        } else {
            b.model_type.clone()
        };
        let expand_factor = if prng.next_f64() < 0.5 {
            a.expand_factor
        } else {
            b.expand_factor
        };
        let dropout = if prng.next_f64() < 0.5 {
            a.dropout
        } else {
            b.dropout
        };

        Ok(ArchCandidate {
            d_model,
            d_state,
            n_layers,
            model_type,
            expand_factor,
            dropout,
        })
    }

    /// Per-gene mutation: with probability `mutation_prob` each gene is replaced
    /// by a fresh random draw from the corresponding option list.
    fn mutate(&self, candidate: &ArchCandidate, prng: &mut Prng) -> ModelResult<ArchCandidate> {
        let d_model = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.d_model_options)
                .ok_or_else(|| ModelError::invalid_config("d_model_options is empty"))?
        } else {
            candidate.d_model
        };

        let d_state = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.d_state_options)
                .ok_or_else(|| ModelError::invalid_config("d_state_options is empty"))?
        } else {
            candidate.d_state
        };

        let n_layers = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.n_layers_options)
                .ok_or_else(|| ModelError::invalid_config("n_layers_options is empty"))?
        } else {
            candidate.n_layers
        };

        let model_type = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.model_types)
                .ok_or_else(|| ModelError::invalid_config("model_types is empty"))?
        } else {
            candidate.model_type.clone()
        };

        let expand_factor = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.expand_factor_options)
                .ok_or_else(|| ModelError::invalid_config("expand_factor_options is empty"))?
        } else {
            candidate.expand_factor
        };

        let dropout = if prng.next_f64() < self.config.mutation_prob {
            prng.choice(&self.space.dropout_options)
                .ok_or_else(|| ModelError::invalid_config("dropout_options is empty"))?
        } else {
            candidate.dropout
        };

        Ok(ArchCandidate {
            d_model,
            d_state,
            n_layers,
            model_type,
            expand_factor,
            dropout,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid searcher
// ─────────────────────────────────────────────────────────────────────────────

/// Exhaustively enumerates every combination in the search space, capped at
/// `max_candidates` so that searches over large grids remain tractable.
pub struct GridSearcher {
    space: ArchSearchSpace,
    max_candidates: usize,
}

impl GridSearcher {
    /// Construct a new grid searcher.
    ///
    /// * `max_candidates` — hard cap on the number of combinations evaluated.
    pub fn new(space: ArchSearchSpace, max_candidates: usize) -> Self {
        Self {
            space,
            max_candidates,
        }
    }

    /// Run the exhaustive grid search and return the result.
    pub fn search(&self) -> ModelResult<ArchSearchResult> {
        if !self.space.is_valid() {
            return Err(ModelError::invalid_config(
                "ArchSearchSpace has empty option lists",
            ));
        }

        let mut all_candidates: Vec<(ArchCandidate, f64)> = Vec::new();

        'outer: for &d_model in &self.space.d_model_options {
            for &d_state in &self.space.d_state_options {
                for &n_layers in &self.space.n_layers_options {
                    for model_type in &self.space.model_types {
                        for &expand_factor in &self.space.expand_factor_options {
                            for &dropout in &self.space.dropout_options {
                                if all_candidates.len() >= self.max_candidates {
                                    break 'outer;
                                }
                                let candidate = ArchCandidate {
                                    d_model,
                                    d_state,
                                    n_layers,
                                    model_type: model_type.clone(),
                                    expand_factor,
                                    dropout,
                                };
                                let score = candidate.proxy_score();
                                all_candidates.push((candidate, score));
                            }
                        }
                    }
                }
            }
        }

        let (best_candidate, best_score) = best_in(&all_candidates)
            .ok_or_else(|| ModelError::invalid_config("grid search produced no candidates"))?;

        Ok(ArchSearchResult {
            best_candidate,
            best_score,
            n_evaluations: all_candidates.len(),
            all_candidates,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience function
// ─────────────────────────────────────────────────────────────────────────────

/// Run an evolutionary search over `space` with the provided `config` and
/// return the best-found architecture.
///
/// This is the recommended entry-point for most users.
pub fn search_best_arch(
    space: ArchSearchSpace,
    config: ArchSearchConfig,
) -> ModelResult<ArchSearchResult> {
    let searcher = EvolutionarySearcher::new(space, config);
    searcher.search()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn small_space() -> ArchSearchSpace {
        ArchSearchSpace {
            d_model_options: vec![64, 128],
            d_state_options: vec![8, 16],
            n_layers_options: vec![2, 4],
            model_types: vec!["mamba".to_string(), "s4".to_string()],
            expand_factor_options: vec![1.0, 2.0],
            dropout_options: vec![0.0],
        }
    }

    fn candidate(model_type: &str) -> ArchCandidate {
        ArchCandidate {
            d_model: 128,
            d_state: 16,
            n_layers: 4,
            model_type: model_type.to_string(),
            expand_factor: 2.0,
            dropout: 0.0,
        }
    }

    // ── 1. param_count returns > 0 for all four model types ──────────────────
    #[test]
    fn test_arch_candidate_param_count() {
        for mt in &["mamba", "transformer", "rwkv", "s4"] {
            let c = candidate(mt);
            assert!(
                c.param_count() > 0,
                "param_count should be > 0 for model_type='{mt}'"
            );
        }
    }

    // ── 2. proxy_score > 0 for a valid candidate ─────────────────────────────
    #[test]
    fn test_arch_candidate_proxy_score() {
        let c = candidate("mamba");
        let score = c.proxy_score();
        assert!(score > 0.0, "proxy_score should be positive, got {score}");
    }

    // ── 3. random searcher returns best_score > 0 ─────────────────────────────
    #[test]
    fn test_random_searcher_finds_best() {
        let space = ArchSearchSpace::default();
        let config = ArchSearchConfig {
            n_candidates: 20,
            seed: 7,
            ..Default::default()
        };
        let searcher = RandomArchSearcher::new(space, config);
        let result = searcher.search().expect("random search failed");
        assert!(
            result.best_score > 0.0,
            "best_score should be > 0, got {}",
            result.best_score
        );
        assert_eq!(result.n_evaluations, 20);
    }

    // ── 4. evolutionary searcher: n_evaluations >= population_size ────────────
    #[test]
    fn test_evolutionary_searcher_n_candidates() {
        let space = ArchSearchSpace::default();
        let config = ArchSearchConfig {
            population_size: 10,
            n_generations: 3,
            n_candidates: 10,
            seed: 99,
            ..Default::default()
        };
        let searcher = EvolutionarySearcher::new(space, config.clone());
        let result = searcher.search().expect("evolutionary search failed");
        assert!(
            result.n_evaluations >= config.population_size,
            "n_evaluations ({}) should be >= population_size ({})",
            result.n_evaluations,
            config.population_size
        );
    }

    // ── 5. grid searcher — small space, all combos returned (up to cap) ───────
    #[test]
    fn test_grid_searcher_exhaustive() {
        let space = small_space();
        // 2*2*2*2*2*1 = 32 combos total
        let expected_total = space.d_model_options.len()
            * space.d_state_options.len()
            * space.n_layers_options.len()
            * space.model_types.len()
            * space.expand_factor_options.len()
            * space.dropout_options.len();

        // With a large cap all combos should be returned
        let searcher = GridSearcher::new(space.clone(), 1000);
        let result = searcher.search().expect("grid search failed");
        assert_eq!(
            result.n_evaluations, expected_total,
            "grid search should enumerate all {expected_total} combinations"
        );

        // With a smaller cap the result is capped
        let capped = GridSearcher::new(space, 5);
        let capped_result = capped.search().expect("capped grid search failed");
        assert_eq!(
            capped_result.n_evaluations, 5,
            "capped grid search should return exactly 5 candidates"
        );
    }

    // ── 6. PRNG is deterministic: same seed → same sequence ──────────────────
    #[test]
    fn test_prng_deterministic() {
        let mut a = Prng::new(12345);
        let mut b = Prng::new(12345);

        let av: Vec<u64> = (0..3).map(|_| a.next_u64()).collect();
        let bv: Vec<u64> = (0..3).map(|_| b.next_u64()).collect();
        assert_eq!(av, bv, "same seed must yield identical sequences");
        // Sanity: different seeds should differ
        let mut c = Prng::new(99999);
        let cv: Vec<u64> = (0..3).map(|_| c.next_u64()).collect();
        assert_ne!(av, cv, "different seeds should produce different sequences");
    }

    // ── 7. convenience function returns Ok ────────────────────────────────────
    #[test]
    fn test_search_best_arch_convenience() {
        let space = ArchSearchSpace::default();
        let config = ArchSearchConfig {
            population_size: 5,
            n_generations: 2,
            n_candidates: 5,
            seed: 1,
            ..Default::default()
        };
        let result = search_best_arch(space, config);
        assert!(result.is_ok(), "search_best_arch should return Ok");
    }

    // ── 8. ArchSearchSpace::default() has non-empty option lists ─────────────
    #[test]
    fn test_arch_search_space_default() {
        let space = ArchSearchSpace::default();
        assert!(
            !space.d_model_options.is_empty(),
            "d_model_options is empty"
        );
        assert!(
            !space.d_state_options.is_empty(),
            "d_state_options is empty"
        );
        assert!(
            !space.n_layers_options.is_empty(),
            "n_layers_options is empty"
        );
        assert!(!space.model_types.is_empty(), "model_types is empty");
        assert!(
            !space.expand_factor_options.is_empty(),
            "expand_factor_options is empty"
        );
        assert!(
            !space.dropout_options.is_empty(),
            "dropout_options is empty"
        );
        assert!(space.is_valid());
    }

    // ── 9. mutated candidate's fields remain valid w.r.t. the search space ───
    #[test]
    fn test_mutation_preserves_validity() {
        let space = small_space();
        let config = ArchSearchConfig {
            mutation_prob: 1.0, // force all genes to mutate
            seed: 777,
            ..Default::default()
        };
        let searcher = EvolutionarySearcher::new(space.clone(), config);
        let base = candidate("mamba");
        let mut prng = Prng::new(42);

        for _ in 0..20 {
            let mutated = searcher.mutate(&base, &mut prng).expect("mutate failed");
            assert!(
                space.d_model_options.contains(&mutated.d_model),
                "mutated d_model {} not in search space",
                mutated.d_model
            );
            assert!(
                space.d_state_options.contains(&mutated.d_state),
                "mutated d_state {} not in search space",
                mutated.d_state
            );
            assert!(
                space.n_layers_options.contains(&mutated.n_layers),
                "mutated n_layers {} not in search space",
                mutated.n_layers
            );
            assert!(
                space.model_types.contains(&mutated.model_type),
                "mutated model_type '{}' not in search space",
                mutated.model_type
            );
            assert!(
                space.expand_factor_options.contains(&mutated.expand_factor),
                "mutated expand_factor {} not in search space",
                mutated.expand_factor
            );
            assert!(
                space.dropout_options.contains(&mutated.dropout),
                "mutated dropout {} not in search space",
                mutated.dropout
            );
        }
    }

    // ── 10. crossover result's fields come from one of the two parents ────────
    #[test]
    fn test_crossover_fields_from_parents() {
        let space = small_space();
        let config = ArchSearchConfig {
            seed: 55,
            ..Default::default()
        };
        let searcher = EvolutionarySearcher::new(space, config);
        let a = ArchCandidate {
            d_model: 64,
            d_state: 8,
            n_layers: 2,
            model_type: "mamba".to_string(),
            expand_factor: 1.0,
            dropout: 0.0,
        };
        let b = ArchCandidate {
            d_model: 128,
            d_state: 16,
            n_layers: 4,
            model_type: "s4".to_string(),
            expand_factor: 2.0,
            dropout: 0.0,
        };
        let mut prng = Prng::new(321);

        for _ in 0..20 {
            let child = searcher
                .crossover(&a, &b, &mut prng)
                .expect("crossover failed");

            assert!(
                child.d_model == a.d_model || child.d_model == b.d_model,
                "d_model {} must come from a parent",
                child.d_model
            );
            assert!(
                child.d_state == a.d_state || child.d_state == b.d_state,
                "d_state {} must come from a parent",
                child.d_state
            );
            assert!(
                child.n_layers == a.n_layers || child.n_layers == b.n_layers,
                "n_layers {} must come from a parent",
                child.n_layers
            );
            assert!(
                child.model_type == a.model_type || child.model_type == b.model_type,
                "model_type '{}' must come from a parent",
                child.model_type
            );
            assert!(
                (child.expand_factor - a.expand_factor).abs() < f32::EPSILON
                    || (child.expand_factor - b.expand_factor).abs() < f32::EPSILON,
                "expand_factor {} must come from a parent",
                child.expand_factor
            );
        }
    }
}
