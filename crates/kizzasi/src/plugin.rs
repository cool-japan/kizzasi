//! Plugin system for extending Kizzasi functionality
//!
//! This module provides a trait-based plugin architecture that allows
//! users to hook into the prediction pipeline for custom preprocessing,
//! postprocessing, logging, metrics collection, and more.

use crate::error::{KizzasiError, KizzasiResult};
use scirs2_core::ndarray::Array1;
use std::any::Any;
use std::fmt;

/// Plugin execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginPhase {
    /// Before prediction (preprocessing)
    PreProcess,
    /// After prediction (postprocessing)
    PostProcess,
    /// On prediction error
    OnError,
    /// On state reset
    OnReset,
}

/// Context provided to plugins during execution
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Current prediction step number
    pub step: usize,
    /// Model input dimension
    pub input_dim: usize,
    /// Model output dimension
    pub output_dim: usize,
    /// Optional user data
    pub user_data: Option<String>,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new(step: usize, input_dim: usize, output_dim: usize) -> Self {
        Self {
            step,
            input_dim,
            output_dim,
            user_data: None,
        }
    }

    /// Set user data
    pub fn with_user_data(mut self, data: String) -> Self {
        self.user_data = Some(data);
        self
    }
}

/// Core plugin trait
///
/// Implement this trait to create custom plugins that hook into
/// the Kizzasi prediction pipeline.
///
/// # Example
///
/// ```rust,ignore
/// use kizzasi::plugin::{Plugin, PluginContext, PluginPhase};
///
/// struct LoggingPlugin {
///     name: String,
/// }
///
/// impl Plugin for LoggingPlugin {
///     fn name(&self) -> &str {
///         &self.name
///     }
///
///     fn on_pre_process(
///         &mut self,
///         input: &Array1<f32>,
///         ctx: &PluginContext,
///     ) -> KizzasiResult<()> {
///         println!("Step {}: Input = {:?}", ctx.step, input);
///         Ok(())
///     }
/// }
/// ```
pub trait Plugin: Send {
    /// Get the plugin name
    fn name(&self) -> &str;

    /// Get plugin description
    fn description(&self) -> &str {
        "No description"
    }

    /// Check if plugin is enabled
    fn is_enabled(&self) -> bool {
        true
    }

    /// Called before prediction (preprocessing)
    fn on_pre_process(&mut self, _input: &Array1<f32>, _ctx: &PluginContext) -> KizzasiResult<()> {
        Ok(())
    }

    /// Transform input before prediction (allows modification)
    fn transform_input(
        &mut self,
        input: Array1<f32>,
        _ctx: &PluginContext,
    ) -> KizzasiResult<Array1<f32>> {
        Ok(input)
    }

    /// Called after prediction (postprocessing)
    fn on_post_process(
        &mut self,
        _input: &Array1<f32>,
        _output: &Array1<f32>,
        _ctx: &PluginContext,
    ) -> KizzasiResult<()> {
        Ok(())
    }

    /// Transform output after prediction (allows modification)
    fn transform_output(
        &mut self,
        output: Array1<f32>,
        _ctx: &PluginContext,
    ) -> KizzasiResult<Array1<f32>> {
        Ok(output)
    }

    /// Called on prediction error
    fn on_error(&mut self, _error: &KizzasiError, _ctx: &PluginContext) -> KizzasiResult<()> {
        Ok(())
    }

    /// Called on state reset
    fn on_reset(&mut self, _ctx: &PluginContext) -> KizzasiResult<()> {
        Ok(())
    }

    /// Get plugin as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get mutable plugin as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Plugin manager for organizing and executing plugins
pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
    step_counter: usize,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            step_counter: 0,
        }
    }

    /// Add a plugin
    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    /// Remove a plugin by name
    pub fn remove_plugin(&mut self, name: &str) -> Option<Box<dyn Plugin>> {
        if let Some(pos) = self.plugins.iter().position(|p| p.name() == name) {
            Some(self.plugins.remove(pos))
        } else {
            None
        }
    }

    /// Get a plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&dyn Plugin> {
        self.plugins
            .iter()
            .find(|p| p.name() == name)
            .map(|p| p.as_ref())
    }

    /// Get a mutable plugin by name
    pub fn get_plugin_mut(&mut self, name: &str) -> Option<&mut Box<dyn Plugin>> {
        self.plugins.iter_mut().find(|p| p.name() == name)
    }

    /// Execute pre-process hooks
    pub fn execute_pre_process(
        &mut self,
        input: &Array1<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<()> {
        let ctx = PluginContext::new(self.step_counter, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                plugin.on_pre_process(input, &ctx)?;
            }
        }
        Ok(())
    }

    /// Transform input through all plugins
    pub fn transform_input(
        &mut self,
        mut input: Array1<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<Array1<f32>> {
        let ctx = PluginContext::new(self.step_counter, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                input = plugin.transform_input(input, &ctx)?;
            }
        }
        Ok(input)
    }

    /// Execute post-process hooks
    pub fn execute_post_process(
        &mut self,
        input: &Array1<f32>,
        output: &Array1<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<()> {
        let ctx = PluginContext::new(self.step_counter, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                plugin.on_post_process(input, output, &ctx)?;
            }
        }
        self.step_counter += 1;
        Ok(())
    }

    /// Transform output through all plugins
    pub fn transform_output(
        &mut self,
        mut output: Array1<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<Array1<f32>> {
        let ctx = PluginContext::new(self.step_counter, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                output = plugin.transform_output(output, &ctx)?;
            }
        }
        Ok(output)
    }

    /// Execute error hooks
    pub fn execute_on_error(
        &mut self,
        error: &KizzasiError,
        input_dim: usize,
        output_dim: usize,
    ) -> KizzasiResult<()> {
        let ctx = PluginContext::new(self.step_counter, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                plugin.on_error(error, &ctx)?;
            }
        }
        Ok(())
    }

    /// Execute reset hooks
    pub fn execute_on_reset(&mut self, input_dim: usize, output_dim: usize) -> KizzasiResult<()> {
        let ctx = PluginContext::new(0, input_dim, output_dim);
        for plugin in &mut self.plugins {
            if plugin.is_enabled() {
                plugin.on_reset(&ctx)?;
            }
        }
        self.step_counter = 0;
        Ok(())
    }

    /// Get number of plugins
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// List all plugin names
    pub fn plugin_names(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for PluginManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginManager")
            .field("plugin_count", &self.plugins.len())
            .field("step_counter", &self.step_counter)
            .field("plugins", &self.plugin_names())
            .finish()
    }
}

// ============================================================================
// Built-in Plugins
// ============================================================================

/// Logging plugin that prints prediction information
pub struct LoggingPlugin {
    name: String,
    enabled: bool,
}

impl LoggingPlugin {
    /// Create a new logging plugin
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
        }
    }

    /// Enable or disable the plugin
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Plugin for LoggingPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Logs prediction inputs and outputs"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn on_pre_process(&mut self, input: &Array1<f32>, ctx: &PluginContext) -> KizzasiResult<()> {
        println!("[{}] Step {}: Input = {:?}", self.name, ctx.step, input);
        Ok(())
    }

    fn on_post_process(
        &mut self,
        _input: &Array1<f32>,
        output: &Array1<f32>,
        ctx: &PluginContext,
    ) -> KizzasiResult<()> {
        println!("[{}] Step {}: Output = {:?}", self.name, ctx.step, output);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Statistics collection plugin
pub struct StatsPlugin {
    name: String,
    enabled: bool,
    prediction_count: usize,
    total_input_magnitude: f32,
    total_output_magnitude: f32,
}

impl StatsPlugin {
    /// Create a new statistics plugin
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            prediction_count: 0,
            total_input_magnitude: 0.0,
            total_output_magnitude: 0.0,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, f32, f32) {
        (
            self.prediction_count,
            if self.prediction_count > 0 {
                self.total_input_magnitude / self.prediction_count as f32
            } else {
                0.0
            },
            if self.prediction_count > 0 {
                self.total_output_magnitude / self.prediction_count as f32
            } else {
                0.0
            },
        )
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.prediction_count = 0;
        self.total_input_magnitude = 0.0;
        self.total_output_magnitude = 0.0;
    }
}

impl Plugin for StatsPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Collects prediction statistics"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn on_post_process(
        &mut self,
        input: &Array1<f32>,
        output: &Array1<f32>,
        _ctx: &PluginContext,
    ) -> KizzasiResult<()> {
        self.prediction_count += 1;
        self.total_input_magnitude += input.iter().map(|x| x.abs()).sum::<f32>();
        self.total_output_magnitude += output.iter().map(|x| x.abs()).sum::<f32>();
        Ok(())
    }

    fn on_reset(&mut self, _ctx: &PluginContext) -> KizzasiResult<()> {
        self.reset_stats();
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_add_remove_plugin() {
        let mut manager = PluginManager::new();

        let plugin = Box::new(LoggingPlugin::new("test_logger"));
        manager.add_plugin(plugin);

        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());

        let removed = manager.remove_plugin("test_logger");
        assert!(removed.is_some());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_logging_plugin() {
        let mut plugin = LoggingPlugin::new("test");
        assert_eq!(plugin.name(), "test");
        assert!(plugin.is_enabled());

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let ctx = PluginContext::new(0, 3, 3);

        plugin.on_pre_process(&input, &ctx).unwrap();
        plugin.on_post_process(&input, &input, &ctx).unwrap();
    }

    #[test]
    fn test_stats_plugin() {
        let mut plugin = StatsPlugin::new("stats");

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let ctx = PluginContext::new(0, 3, 3);

        plugin.on_post_process(&input, &output, &ctx).unwrap();

        let (count, avg_in, avg_out) = plugin.stats();
        assert_eq!(count, 1);
        assert_eq!(avg_in, 6.0); // 1 + 2 + 3
        assert_eq!(avg_out, 3.0); // 0.5 + 1.0 + 1.5
    }

    #[test]
    fn test_plugin_manager_execution() {
        let mut manager = PluginManager::new();
        manager.add_plugin(Box::new(StatsPlugin::new("stats")));

        let input = Array1::from_vec(vec![0.1, 0.2]);
        let output = Array1::from_vec(vec![0.3, 0.4]);

        manager.execute_pre_process(&input, 2, 2).unwrap();
        manager.execute_post_process(&input, &output, 2, 2).unwrap();

        let plugin = manager.get_plugin("stats").unwrap();
        let stats_plugin = plugin.as_any().downcast_ref::<StatsPlugin>().unwrap();
        let (count, _, _) = stats_plugin.stats();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_plugin_reset() {
        let mut manager = PluginManager::new();
        manager.add_plugin(Box::new(StatsPlugin::new("stats")));

        let input = Array1::from_vec(vec![0.1]);
        let output = Array1::from_vec(vec![0.2]);

        manager.execute_post_process(&input, &output, 1, 1).unwrap();
        manager.execute_on_reset(1, 1).unwrap();

        let plugin = manager.get_plugin("stats").unwrap();
        let stats_plugin = plugin.as_any().downcast_ref::<StatsPlugin>().unwrap();
        let (count, _, _) = stats_plugin.stats();
        assert_eq!(count, 0);
    }
}
