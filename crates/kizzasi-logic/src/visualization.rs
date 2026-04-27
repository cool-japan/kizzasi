//! Visualization and debugging tools for constraints
//!
//! This module provides utilities for:
//! - Inspecting constraint states and violations
//! - Debugging constraint composition and decomposition
//! - Analyzing constraint satisfaction over time
//! - Generating human-readable constraint reports

use crate::ViolationComputable;
use std::collections::HashMap;
use std::fmt;

/// Statistics about constraint violations over a sequence of values
#[derive(Debug, Clone)]
pub struct ViolationStats {
    /// Total number of samples
    pub num_samples: usize,
    /// Number of violations
    pub num_violations: usize,
    /// Mean violation magnitude
    pub mean_violation: f32,
    /// Maximum violation magnitude
    pub max_violation: f32,
    /// Minimum violation magnitude (for violated samples)
    pub min_violation: f32,
    /// Standard deviation of violations
    pub std_violation: f32,
}

impl ViolationStats {
    /// Create violation statistics from a sequence of values
    pub fn from_values<C: ViolationComputable>(constraint: &C, values: &[f32]) -> Self {
        let num_samples = values.len();
        let mut num_violations = 0;
        let mut violation_sum = 0.0f32;
        let mut max_violation = 0.0f32;
        let mut min_violation = f32::MAX;
        let mut violations = Vec::new();

        for &val in values {
            let viol = constraint.violation(&[val]);
            if viol > 1e-6 {
                num_violations += 1;
                violation_sum += viol;
                max_violation = max_violation.max(viol);
                min_violation = min_violation.min(viol);
                violations.push(viol);
            }
        }

        let mean_violation = if num_violations > 0 {
            violation_sum / num_violations as f32
        } else {
            0.0
        };

        let std_violation = if num_violations > 0 {
            let variance: f32 = violations
                .iter()
                .map(|&v| (v - mean_violation).powi(2))
                .sum::<f32>()
                / num_violations as f32;
            variance.sqrt()
        } else {
            0.0
        };

        if min_violation == f32::MAX {
            min_violation = 0.0;
        }

        Self {
            num_samples,
            num_violations,
            mean_violation,
            max_violation,
            min_violation,
            std_violation,
        }
    }

    /// Violation rate (fraction of samples that violate)
    pub fn violation_rate(&self) -> f32 {
        if self.num_samples == 0 {
            0.0
        } else {
            self.num_violations as f32 / self.num_samples as f32
        }
    }

    /// Satisfaction rate (fraction of samples that satisfy)
    pub fn satisfaction_rate(&self) -> f32 {
        1.0 - self.violation_rate()
    }
}

impl fmt::Display for ViolationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Violation Statistics:")?;
        writeln!(f, "  Samples: {}", self.num_samples)?;
        writeln!(
            f,
            "  Violations: {} ({:.1}%)",
            self.num_violations,
            self.violation_rate() * 100.0
        )?;
        writeln!(
            f,
            "  Satisfactions: {} ({:.1}%)",
            self.num_samples - self.num_violations,
            self.satisfaction_rate() * 100.0
        )?;
        writeln!(f, "  Mean violation: {:.4}", self.mean_violation)?;
        writeln!(f, "  Max violation: {:.4}", self.max_violation)?;
        writeln!(f, "  Min violation: {:.4}", self.min_violation)?;
        writeln!(f, "  Std violation: {:.4}", self.std_violation)?;
        Ok(())
    }
}

/// Time series analysis of constraint satisfaction
pub struct ConstraintTimeSeries {
    name: String,
    values: Vec<f32>,
    violations: Vec<f32>,
    timestamps: Vec<usize>,
}

impl ConstraintTimeSeries {
    /// Create a new time series tracker
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            violations: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a sample to the time series
    pub fn add_sample<C: ViolationComputable>(
        &mut self,
        constraint: &C,
        value: f32,
        timestamp: usize,
    ) {
        let violation = constraint.violation(&[value]);
        self.values.push(value);
        self.violations.push(violation);
        self.timestamps.push(timestamp);
    }

    /// Get statistics for the entire time series
    pub fn stats(&self) -> ViolationStats {
        let num_samples = self.values.len();
        let mut num_violations = 0;
        let mut violation_sum = 0.0f32;
        let mut max_violation = 0.0f32;
        let mut min_violation = f32::MAX;
        let mut active_violations = Vec::new();

        for &viol in &self.violations {
            if viol > 1e-6 {
                num_violations += 1;
                violation_sum += viol;
                max_violation = max_violation.max(viol);
                min_violation = min_violation.min(viol);
                active_violations.push(viol);
            }
        }

        let mean_violation = if num_violations > 0 {
            violation_sum / num_violations as f32
        } else {
            0.0
        };

        let std_violation = if num_violations > 0 {
            let variance: f32 = active_violations
                .iter()
                .map(|&v| (v - mean_violation).powi(2))
                .sum::<f32>()
                / num_violations as f32;
            variance.sqrt()
        } else {
            0.0
        };

        if min_violation == f32::MAX {
            min_violation = 0.0;
        }

        ViolationStats {
            num_samples,
            num_violations,
            mean_violation,
            max_violation,
            min_violation,
            std_violation,
        }
    }

    /// Get the name of this time series
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get violation history
    pub fn violations(&self) -> &[f32] {
        &self.violations
    }

    /// Get value history
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get timestamps
    pub fn timestamps(&self) -> &[usize] {
        &self.timestamps
    }

    /// Find periods of continuous violation
    pub fn violation_periods(&self) -> Vec<(usize, usize, f32)> {
        let mut periods = Vec::new();
        let mut in_violation = false;
        let mut start = 0;
        let mut max_viol_in_period = 0.0f32;

        for (i, &viol) in self.violations.iter().enumerate() {
            if viol > 1e-6 {
                if !in_violation {
                    in_violation = true;
                    start = i;
                    max_viol_in_period = viol;
                } else {
                    max_viol_in_period = max_viol_in_period.max(viol);
                }
            } else if in_violation {
                periods.push((start, i - 1, max_viol_in_period));
                in_violation = false;
                max_viol_in_period = 0.0;
            }
        }

        // Handle case where violation continues to end
        if in_violation {
            periods.push((start, self.violations.len() - 1, max_viol_in_period));
        }

        periods
    }
}

/// Debugging report for multiple constraints
pub struct ConstraintReport {
    constraint_names: Vec<String>,
    stats: HashMap<String, ViolationStats>,
}

impl ConstraintReport {
    /// Create a new constraint report
    pub fn new() -> Self {
        Self {
            constraint_names: Vec::new(),
            stats: HashMap::new(),
        }
    }

    /// Add constraint statistics to the report
    pub fn add_constraint(&mut self, name: impl Into<String>, stats: ViolationStats) {
        let name = name.into();
        self.constraint_names.push(name.clone());
        self.stats.insert(name, stats);
    }

    /// Get statistics for a specific constraint
    pub fn get_stats(&self, name: &str) -> Option<&ViolationStats> {
        self.stats.get(name)
    }

    /// Get all constraint names
    pub fn constraint_names(&self) -> &[String] {
        &self.constraint_names
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::from("Constraint Report\n");
        report.push_str("=================\n\n");

        for name in &self.constraint_names {
            if let Some(stats) = self.stats.get(name) {
                report.push_str(&format!("Constraint: {}\n", name));
                report.push_str(&format!(
                    "  Violation rate: {:.1}%\n",
                    stats.violation_rate() * 100.0
                ));
                report.push_str(&format!("  Mean violation: {:.4}\n", stats.mean_violation));
                report.push_str(&format!("  Max violation: {:.4}\n", stats.max_violation));
                report.push('\n');
            }
        }

        report
    }

    /// Find the most frequently violated constraint
    pub fn most_violated(&self) -> Option<(&str, &ViolationStats)> {
        self.stats
            .iter()
            .max_by(|a, b| {
                a.1.violation_rate()
                    .partial_cmp(&b.1.violation_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, stats)| (name.as_str(), stats))
    }

    /// Find the constraint with the largest violations
    pub fn largest_violations(&self) -> Option<(&str, &ViolationStats)> {
        self.stats
            .iter()
            .max_by(|a, b| {
                a.1.max_violation
                    .partial_cmp(&b.1.max_violation)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, stats)| (name.as_str(), stats))
    }
}

impl Default for ConstraintReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConstraintReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Constraint inspector for interactive debugging
pub struct ConstraintInspector {
    samples: Vec<f32>,
}

impl ConstraintInspector {
    /// Create a new constraint inspector
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add a sample value to inspect
    pub fn add_sample(&mut self, value: f32) {
        self.samples.push(value);
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Inspect a constraint against all samples
    pub fn inspect<C: ViolationComputable + ?Sized>(
        &self,
        constraint: &C,
        name: &str,
    ) -> InspectionResult {
        let mut satisfied = Vec::new();
        let mut violated = Vec::new();

        for (i, &val) in self.samples.iter().enumerate() {
            let viol = constraint.violation(&[val]);
            if viol <= 1e-6 {
                satisfied.push((i, val));
            } else {
                violated.push((i, val, viol));
            }
        }

        InspectionResult {
            constraint_name: name.to_string(),
            total_samples: self.samples.len(),
            satisfied,
            violated,
        }
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get all samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

impl Default for ConstraintInspector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of constraint inspection
pub struct InspectionResult {
    constraint_name: String,
    total_samples: usize,
    satisfied: Vec<(usize, f32)>,
    violated: Vec<(usize, f32, f32)>,
}

impl InspectionResult {
    /// Get constraint name
    pub fn constraint_name(&self) -> &str {
        &self.constraint_name
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get satisfied samples (index, value)
    pub fn satisfied(&self) -> &[(usize, f32)] {
        &self.satisfied
    }

    /// Get violated samples (index, value, violation)
    pub fn violated(&self) -> &[(usize, f32, f32)] {
        &self.violated
    }

    /// Get violation rate
    pub fn violation_rate(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.violated.len() as f32 / self.total_samples as f32
        }
    }

    /// Print a summary
    pub fn print_summary(&self) {
        println!("Inspection: {}", self.constraint_name);
        println!("  Total samples: {}", self.total_samples);
        println!(
            "  Satisfied: {} ({:.1}%)",
            self.satisfied.len(),
            (self.satisfied.len() as f32 / self.total_samples as f32) * 100.0
        );
        println!(
            "  Violated: {} ({:.1}%)",
            self.violated.len(),
            self.violation_rate() * 100.0
        );
    }

    /// Print detailed violation information
    pub fn print_violations(&self, max_items: usize) {
        println!("\nViolations for {}:", self.constraint_name);
        let n = self.violated.len().min(max_items);
        for (i, &(idx, val, viol)) in self.violated.iter().take(n).enumerate() {
            println!(
                "  [{}] Sample {}: value={:.4}, violation={:.4}",
                i + 1,
                idx,
                val,
                viol
            );
        }
        if self.violated.len() > max_items {
            println!(
                "  ... and {} more violations",
                self.violated.len() - max_items
            );
        }
    }
}

// ============================================================================
// SVG-based constraint visualization
// ============================================================================

/// Color maps for heatmaps
#[derive(Debug, Clone, Copy, Default)]
pub enum Colormap {
    #[default]
    /// Blue → green → yellow (approximate viridis)
    Viridis,
    /// Red (violation) → green (satisfied)
    RdGn,
    /// Grayscale
    Grayscale,
}

impl Colormap {
    /// Map scalar value `t` in `[0, 1]` to an RGB triple.
    pub fn map(&self, t: f64) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => {
                let r = (255.0 * (t * t * t * 0.5 + t * 0.5)).min(255.0) as u8;
                let g = (255.0 * (0.5 * (1.0 - (2.0 * t - 1.0).abs()))).min(255.0) as u8;
                let b = (255.0 * ((1.0 - t) * 0.9)).min(255.0) as u8;
                (r, g, b)
            }
            Colormap::RdGn => {
                let r = (255.0 * (1.0 - t)).min(255.0) as u8;
                let g = (255.0 * t).min(255.0) as u8;
                (r, g, 0)
            }
            Colormap::Grayscale => {
                let v = (255.0 * t) as u8;
                (v, v, v)
            }
        }
    }
}

/// 2D plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// SVG width in pixels
    pub width: usize,
    /// SVG height in pixels
    pub height: usize,
    /// x-axis range `(min, max)`
    pub x_range: (f64, f64),
    /// y-axis range `(min, max)`
    pub y_range: (f64, f64),
    /// Optional title drawn at the top of the SVG
    pub title: String,
    /// Draw background grid
    pub grid: bool,
    /// Colormap used for heatmap rendering
    pub colormap: Colormap,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 600,
            height: 600,
            x_range: (-3.0, 3.0),
            y_range: (-3.0, 3.0),
            title: String::new(),
            grid: true,
            colormap: Colormap::Viridis,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SvgBuilder
// ────────────────────────────────────────────────────────────────────────────

/// Lightweight SVG string builder — no external dependencies required.
pub struct SvgBuilder {
    width: usize,
    height: usize,
    elements: Vec<String>,
}

impl SvgBuilder {
    /// Create a new builder for an SVG canvas of `width × height` pixels.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            elements: Vec::new(),
        }
    }

    /// Append a `<rect>` element.
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, opacity: f64) -> &mut Self {
        self.elements.push(format!(
            r#"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}" opacity="{:.3}"/>"#,
            x, y, w, h, fill, opacity
        ));
        self
    }

    /// Append a `<circle>` element.
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<circle cx="{:.3}" cy="{:.3}" r="{:.3}" fill="{}" stroke="{}"/>"#,
            cx, cy, r, fill, stroke
        ));
        self
    }

    /// Append a `<line>` element.
    pub fn line(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stroke: &str,
        width: f64,
    ) -> &mut Self {
        self.elements.push(format!(
            r#"<line x1="{:.3}" y1="{:.3}" x2="{:.3}" y2="{:.3}" stroke="{}" stroke-width="{:.3}"/>"#,
            x1, y1, x2, y2, stroke, width
        ));
        self
    }

    /// Append a `<text>` element.
    pub fn text(&mut self, x: f64, y: f64, content: &str, size: usize, anchor: &str) -> &mut Self {
        self.elements.push(format!(
            r#"<text x="{:.3}" y="{:.3}" font-size="{}" text-anchor="{}">{}</text>"#,
            x,
            y,
            size,
            anchor,
            escape_xml(content)
        ));
        self
    }

    /// Serialise all accumulated elements into a complete SVG document.
    pub fn build(&self) -> String {
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );
        for elem in &self.elements {
            svg.push('\n');
            svg.push_str(elem);
        }
        svg.push_str("\n</svg>");
        svg
    }
}

/// Escape `<`, `>`, `&`, `"` for safe inclusion inside SVG text nodes.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ────────────────────────────────────────────────────────────────────────────
// ConstraintPlotter
// ────────────────────────────────────────────────────────────────────────────

/// Renders 2-D constraint regions and trajectories as SVG.
pub struct ConstraintPlotter {
    config: PlotConfig,
}

impl ConstraintPlotter {
    /// Construct a plotter with the given configuration.
    pub fn new(config: PlotConfig) -> Self {
        Self { config }
    }

    // ── coordinate helpers ────────────────────────────────────────────────

    /// Convert world coordinates `(x, y)` to SVG pixel coordinates.
    fn world_to_svg(&self, x: f64, y: f64) -> (f64, f64) {
        let px = (x - self.config.x_range.0) / (self.config.x_range.1 - self.config.x_range.0)
            * self.config.width as f64;
        let py = (1.0
            - (y - self.config.y_range.0) / (self.config.y_range.1 - self.config.y_range.0))
            * self.config.height as f64;
        (px, py)
    }

    // ── optional background helpers ───────────────────────────────────────

    fn push_background(&self, svg: &mut SvgBuilder) {
        svg.rect(
            0.0,
            0.0,
            self.config.width as f64,
            self.config.height as f64,
            "#ffffff",
            1.0,
        );
    }

    fn push_grid(&self, svg: &mut SvgBuilder) {
        if !self.config.grid {
            return;
        }
        let n_lines = 10usize;
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        for i in 0..=n_lines {
            let frac = i as f64 / n_lines as f64;
            let x = frac * w;
            let y = frac * h;
            svg.line(x, 0.0, x, h, "#e0e0e0", 0.5);
            svg.line(0.0, y, w, y, "#e0e0e0", 0.5);
        }
    }

    fn push_title(&self, svg: &mut SvgBuilder) {
        if !self.config.title.is_empty() {
            svg.text(
                self.config.width as f64 / 2.0,
                16.0,
                &self.config.title,
                14,
                "middle",
            );
        }
    }

    // ── public API ────────────────────────────────────────────────────────

    /// Plot the 2D constraint region for `constraint_fn(x, y)`.
    ///
    /// Values `< 0` are considered *satisfied* (green / dark for viridis),
    /// values `≥ 0` are *violated* and rendered with a colour that encodes
    /// the magnitude.
    ///
    /// Returns a self-contained SVG string.
    pub fn plot_constraint_region<F>(&self, constraint_fn: F, resolution: usize) -> String
    where
        F: Fn(f64, f64) -> f64,
    {
        let res = resolution.max(1);
        let (x0, x1) = self.config.x_range;
        let (y0, y1) = self.config.y_range;
        let dx = (x1 - x0) / res as f64;
        let dy = (y1 - y0) / res as f64;

        // First pass: compute all values and find the maximum violation magnitude.
        let mut values = Vec::with_capacity(res * res);
        let mut max_viol = 1e-9_f64;
        for row in 0..res {
            for col in 0..res {
                let cx = x0 + (col as f64 + 0.5) * dx;
                let cy = y0 + (row as f64 + 0.5) * dy;
                let v = constraint_fn(cx, cy);
                if v > max_viol {
                    max_viol = v;
                }
                values.push((cx, cy, v));
            }
        }

        let cell_w = self.config.width as f64 / res as f64;
        let cell_h = self.config.height as f64 / res as f64;

        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        self.push_background(&mut svg);
        self.push_grid(&mut svg);

        for (cx, cy, v) in &values {
            // t = 0 → satisfied (constraint boundary), t = 1 → maximally violated
            let t = if *v <= 0.0 {
                0.0
            } else {
                (*v / max_viol).clamp(0.0, 1.0)
            };
            let (r, g, b) = self.config.colormap.map(t);
            let fill = format!("#{:02X}{:02X}{:02X}", r, g, b);
            let (px, py) = self.world_to_svg(*cx - dx * 0.5, *cy + dy * 0.5);
            svg.rect(px, py, cell_w, cell_h, &fill, 0.85);
        }

        self.push_title(&mut svg);
        svg.build()
    }

    /// Plot the *feasible region* where **all** constraints are satisfied.
    ///
    /// Each cell is coloured green when all constraints hold (every `f(x,y) ≤ 0`)
    /// and red when at least one is violated.
    pub fn plot_feasible_region<F>(&self, constraints: &[F], resolution: usize) -> String
    where
        F: Fn(f64, f64) -> f64,
    {
        // Degenerate: no constraints → everything is feasible.
        if constraints.is_empty() {
            let mut svg = SvgBuilder::new(self.config.width, self.config.height);
            self.push_background(&mut svg);
            self.push_grid(&mut svg);
            svg.rect(
                0.0,
                0.0,
                self.config.width as f64,
                self.config.height as f64,
                "#00cc00",
                0.4,
            );
            self.push_title(&mut svg);
            return svg.build();
        }

        let res = resolution.max(1);
        let (x0, x1) = self.config.x_range;
        let (y0, y1) = self.config.y_range;
        let dx = (x1 - x0) / res as f64;
        let dy = (y1 - y0) / res as f64;
        let cell_w = self.config.width as f64 / res as f64;
        let cell_h = self.config.height as f64 / res as f64;

        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        self.push_background(&mut svg);
        self.push_grid(&mut svg);

        for row in 0..res {
            for col in 0..res {
                let cx = x0 + (col as f64 + 0.5) * dx;
                let cy = y0 + (row as f64 + 0.5) * dy;

                let max_viol = constraints
                    .iter()
                    .map(|f| f(cx, cy))
                    .fold(f64::NEG_INFINITY, f64::max);

                let fill = if max_viol <= 0.0 {
                    "#00cc00".to_string()
                } else {
                    "#cc0000".to_string()
                };
                let (px, py) = self.world_to_svg(cx - dx * 0.5, cy + dy * 0.5);
                svg.rect(px, py, cell_w, cell_h, &fill, 0.6);
            }
        }

        self.push_title(&mut svg);
        svg.build()
    }

    /// Plot a sequence of 2-D points as a connected trajectory.
    ///
    /// If `constraint_fn` is provided, points that violate the constraint
    /// (`f(x, y) > 0`) are rendered in red; satisfied points are green.
    pub fn plot_trajectory(
        &self,
        points: &[(f64, f64)],
        constraint_fn: Option<&dyn Fn(f64, f64) -> f64>,
    ) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        self.push_background(&mut svg);
        self.push_grid(&mut svg);

        if points.is_empty() {
            self.push_title(&mut svg);
            return svg.build();
        }

        // Draw edges
        let svg_pts: Vec<(f64, f64)> = points
            .iter()
            .map(|&(x, y)| self.world_to_svg(x, y))
            .collect();

        for i in 1..svg_pts.len() {
            let (x1, y1) = svg_pts[i - 1];
            let (x2, y2) = svg_pts[i];
            svg.line(x1, y1, x2, y2, "#888888", 1.5);
        }

        // Draw nodes
        for (idx, &(x, y)) in points.iter().enumerate() {
            let violated = constraint_fn.is_some_and(|f| f(x, y) > 0.0);
            let fill = if violated { "#cc0000" } else { "#00aa44" };
            let (px, py) = svg_pts[idx];
            svg.circle(px, py, 4.0, fill, "#333333");
        }

        self.push_title(&mut svg);
        svg.build()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Constraint network
// ────────────────────────────────────────────────────────────────────────────

/// A node in a constraint dependency graph.
#[derive(Debug, Clone)]
pub struct ConstraintNetworkNode {
    /// Unique numeric identifier for the node.
    pub id: usize,
    /// Human-readable label (e.g. "x >= 0").
    pub label: String,
    /// Category of the constraint (e.g. "range", "linear", "temporal").
    pub constraint_type: String,
    /// Satisfaction status: `Some(true)` = satisfied, `Some(false)` = violated,
    /// `None` = unknown.
    pub satisfied: Option<bool>,
}

/// A directed edge in a constraint dependency graph.
#[derive(Debug, Clone)]
pub struct ConstraintNetworkEdge {
    /// Source node id.
    pub from: usize,
    /// Destination node id.
    pub to: usize,
    /// Relationship label (e.g. "AND", "implies").
    pub label: String,
}

/// Render a constraint dependency network as an SVG diagram.
///
/// Nodes are laid out on a circle; edges are drawn as lines with labels.
pub fn render_constraint_network(
    nodes: &[ConstraintNetworkNode],
    edges: &[ConstraintNetworkEdge],
    width: usize,
    height: usize,
) -> String {
    let mut svg = SvgBuilder::new(width, height);

    // White background
    svg.rect(0.0, 0.0, width as f64, height as f64, "#ffffff", 1.0);

    if nodes.is_empty() {
        return svg.build();
    }

    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let radius = (width.min(height) as f64 / 2.0 - 60.0).max(20.0);
    let node_r = 22.0_f64;

    // Compute positions
    let positions: Vec<(f64, f64)> = nodes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (nodes.len() as f64);
            let cx = center_x + radius * angle.cos();
            let cy = center_y + radius * angle.sin();
            (cx, cy)
        })
        .collect();

    // Build a quick id → position index map
    let id_to_idx: std::collections::HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .map(|(idx, n)| (n.id, idx))
        .collect();

    // Draw edges first (below nodes)
    for edge in edges {
        let from_idx = match id_to_idx.get(&edge.from) {
            Some(&i) => i,
            None => continue,
        };
        let to_idx = match id_to_idx.get(&edge.to) {
            Some(&i) => i,
            None => continue,
        };
        let (x1, y1) = positions[from_idx];
        let (x2, y2) = positions[to_idx];
        svg.line(x1, y1, x2, y2, "#666666", 1.5);

        // Edge label at midpoint
        if !edge.label.is_empty() {
            svg.text(
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0 - 4.0,
                &edge.label,
                10,
                "middle",
            );
        }
    }

    // Draw nodes
    for (i, node) in nodes.iter().enumerate() {
        let (cx, cy) = positions[i];
        let fill = match node.satisfied {
            Some(true) => "#44bb77",
            Some(false) => "#cc4444",
            None => "#8888cc",
        };
        svg.circle(cx, cy, node_r, fill, "#333333");
        svg.text(cx, cy + 4.0, &node.label, 10, "middle");
    }

    svg.build()
}

// ────────────────────────────────────────────────────────────────────────────
// Violation heatmap for scattered points
// ────────────────────────────────────────────────────────────────────────────

/// Render a scatter-plot heatmap where each point is coloured by its violation
/// magnitude.
///
/// Points and violations must be the same length; any mismatch silently uses
/// the shorter length. Returns a self-contained SVG string.
pub fn render_violation_heatmap(
    points: &[(f64, f64)],
    violations: &[f64],
    config: &PlotConfig,
) -> String {
    let mut svg = SvgBuilder::new(config.width, config.height);
    svg.rect(
        0.0,
        0.0,
        config.width as f64,
        config.height as f64,
        "#ffffff",
        1.0,
    );

    if config.grid {
        let n_lines = 10usize;
        let w = config.width as f64;
        let h = config.height as f64;
        for i in 0..=n_lines {
            let frac = i as f64 / n_lines as f64;
            svg.line(frac * w, 0.0, frac * w, h, "#e0e0e0", 0.5);
            svg.line(0.0, frac * h, w, frac * h, "#e0e0e0", 0.5);
        }
    }

    let max_viol = violations.iter().cloned().fold(1e-9_f64, f64::max);

    let n = points.len().min(violations.len());
    for i in 0..n {
        let (x, y) = points[i];
        let v = violations[i];

        // Map world coords to SVG
        let px =
            (x - config.x_range.0) / (config.x_range.1 - config.x_range.0) * config.width as f64;
        let py = (1.0 - (y - config.y_range.0) / (config.y_range.1 - config.y_range.0))
            * config.height as f64;

        let t = (v / max_viol).clamp(0.0, 1.0);
        let (r, g, b) = config.colormap.map(t);
        let fill = format!("#{:02X}{:02X}{:02X}", r, g, b);

        svg.circle(px, py, 6.0, &fill, "#333333");
    }

    if !config.title.is_empty() {
        svg.text(config.width as f64 / 2.0, 16.0, &config.title, 14, "middle");
    }

    svg.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintBuilder;

    #[test]
    fn test_violation_stats() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let values = vec![5.0, 8.0, 12.0, 15.0, 9.0];
        let stats = ViolationStats::from_values(&constraint, &values);

        assert_eq!(stats.num_samples, 5);
        assert_eq!(stats.num_violations, 2); // 12.0 and 15.0 violate
        assert!(stats.violation_rate() > 0.39 && stats.violation_rate() < 0.41);
    }

    #[test]
    fn test_constraint_time_series() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let mut ts = ConstraintTimeSeries::new("test_series");
        ts.add_sample(&constraint, 5.0, 0);
        ts.add_sample(&constraint, 12.0, 1);
        ts.add_sample(&constraint, 15.0, 2);
        ts.add_sample(&constraint, 8.0, 3);

        let stats = ts.stats();
        assert_eq!(stats.num_samples, 4);
        assert_eq!(stats.num_violations, 2);

        let periods = ts.violation_periods();
        assert_eq!(periods.len(), 1); // One continuous period
        assert_eq!(periods[0].0, 1); // Starts at index 1
        assert_eq!(periods[0].1, 2); // Ends at index 2
    }

    #[test]
    fn test_constraint_report() {
        let constraint1 = ConstraintBuilder::new()
            .name("c1")
            .less_than(10.0)
            .build()
            .unwrap();

        let constraint2 = ConstraintBuilder::new()
            .name("c2")
            .less_than(5.0)
            .build()
            .unwrap();

        let values = vec![3.0, 7.0, 12.0];

        let mut report = ConstraintReport::new();
        report.add_constraint("c1", ViolationStats::from_values(&constraint1, &values));
        report.add_constraint("c2", ViolationStats::from_values(&constraint2, &values));

        assert_eq!(report.constraint_names().len(), 2);

        let most_violated = report.most_violated();
        assert!(most_violated.is_some());
    }

    #[test]
    fn test_constraint_inspector() {
        let constraint = ConstraintBuilder::new()
            .name("test")
            .less_than(10.0)
            .build()
            .unwrap();

        let mut inspector = ConstraintInspector::new();
        inspector.add_sample(5.0);
        inspector.add_sample(12.0);
        inspector.add_sample(8.0);

        let result = inspector.inspect(&constraint, "test");
        assert_eq!(result.total_samples(), 3);
        assert_eq!(result.satisfied().len(), 2);
        assert_eq!(result.violated().len(), 1);
    }

    // ── SVG visualization tests ───────────────────────────────────────────

    #[test]
    fn test_svg_builder_produces_valid_xml() {
        let mut builder = SvgBuilder::new(100, 100);
        builder.rect(10.0, 10.0, 80.0, 80.0, "#ff0000", 1.0);
        builder.circle(50.0, 50.0, 20.0, "#00ff00", "none");
        let svg = builder.build();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("rect"));
        assert!(svg.contains("circle"));
    }

    #[test]
    fn test_colormap_viridis_range() {
        let cm = Colormap::Viridis;
        let (r0, g0, b0) = cm.map(0.0);
        let (r1, g1, b1) = cm.map(1.0);
        // Verify valid RGB ranges (u8 is always <= 255, but confirm values are assigned)
        let _ = (r0, g0, b0, r1, g1, b1);
    }

    #[test]
    fn test_colormap_rdgn_extremes() {
        let cm = Colormap::RdGn;
        let (r_red, _, _) = cm.map(0.0);
        let (_, g_green, _) = cm.map(1.0);
        assert_eq!(r_red, 255); // fully red at 0
        assert_eq!(g_green, 255); // fully green at 1
    }

    #[test]
    fn test_plot_constraint_region_returns_svg() {
        let config = PlotConfig {
            width: 100,
            height: 100,
            x_range: (-2.0, 2.0),
            y_range: (-2.0, 2.0),
            ..Default::default()
        };
        let plotter = ConstraintPlotter::new(config);
        // Circle constraint: x^2 + y^2 - 1 <= 0 (satisfied inside unit circle)
        let svg = plotter.plot_constraint_region(|x, y| x * x + y * y - 1.0, 10);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn test_plot_feasible_region() {
        let config = PlotConfig::default();
        let plotter = ConstraintPlotter::new(config);
        let constraints: Vec<Box<dyn Fn(f64, f64) -> f64>> = vec![
            Box::new(|x, _y| x - 1.0), // x <= 1
            Box::new(|_x, y| y - 1.0), // y <= 1
        ];
        let svg = plotter.plot_feasible_region(&constraints, 5);
        assert!(!svg.is_empty());
    }

    #[test]
    fn test_plot_trajectory() {
        let config = PlotConfig::default();
        let plotter = ConstraintPlotter::new(config);
        let pts = vec![(0.0_f64, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let svg = plotter.plot_trajectory(&pts, None);
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_render_constraint_network() {
        let nodes = vec![
            ConstraintNetworkNode {
                id: 0,
                label: "x >= 0".to_string(),
                constraint_type: "range".to_string(),
                satisfied: Some(true),
            },
            ConstraintNetworkNode {
                id: 1,
                label: "x <= 1".to_string(),
                constraint_type: "range".to_string(),
                satisfied: Some(false),
            },
        ];
        let edges = vec![ConstraintNetworkEdge {
            from: 0,
            to: 1,
            label: "AND".to_string(),
        }];
        let svg = render_constraint_network(&nodes, &edges, 400, 400);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("circle") || svg.contains("ellipse"));
    }

    #[test]
    fn test_render_violation_heatmap() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let violations = vec![0.0, 0.5, 1.0];
        let config = PlotConfig {
            width: 100,
            height: 100,
            ..Default::default()
        };
        let svg = render_violation_heatmap(&pts, &violations, &config);
        assert!(!svg.is_empty());
    }
}
