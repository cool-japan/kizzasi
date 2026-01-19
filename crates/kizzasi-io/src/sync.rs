//! Multi-stream synchronization for time-aligned processing
//!
//! This module provides utilities for synchronizing multiple data streams
//! based on timestamps, enabling precise multi-modal sensor fusion.

use crate::error::{IoError, IoResult};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Timestamp type (microseconds since epoch)
pub type Timestamp = u64;

/// Timestamped data sample
#[derive(Debug, Clone)]
pub struct TimestampedSample {
    pub timestamp: Timestamp,
    pub data: Array1<f32>,
    pub stream_id: String,
}

impl TimestampedSample {
    /// Create new timestamped sample
    pub fn new(timestamp: Timestamp, data: Array1<f32>, stream_id: String) -> Self {
        Self {
            timestamp,
            data,
            stream_id,
        }
    }

    /// Create with current time
    pub fn now(data: Array1<f32>, stream_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time must be after UNIX_EPOCH")
            .as_micros() as u64;
        Self::new(timestamp, data, stream_id)
    }

    /// Get age of sample
    pub fn age(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        Duration::from_micros(now.saturating_sub(self.timestamp))
    }
}

/// Configuration for stream synchronizer
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum time difference for alignment (microseconds)
    pub max_time_diff: u64,
    /// Buffer size per stream
    pub buffer_size: usize,
    /// Timeout for waiting for samples
    pub timeout: Duration,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Use nearest sample in time
    Nearest,
    /// Linear interpolation between samples
    Linear,
    /// Hold previous value
    Hold,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_time_diff: 10_000, // 10ms
            buffer_size: 100,
            timeout: Duration::from_millis(100),
            interpolation: InterpolationMethod::Linear,
        }
    }
}

/// Multi-stream synchronizer
pub struct StreamSynchronizer {
    config: SyncConfig,
    buffers: HashMap<String, VecDeque<TimestampedSample>>,
    last_sync_time: Option<Timestamp>,
}

impl StreamSynchronizer {
    /// Create new synchronizer with configuration
    pub fn new(config: SyncConfig) -> Self {
        Self {
            config,
            buffers: HashMap::new(),
            last_sync_time: None,
        }
    }

    /// Add a stream to synchronize
    pub fn add_stream(&mut self, stream_id: String) {
        self.buffers
            .entry(stream_id)
            .or_insert_with(|| VecDeque::with_capacity(self.config.buffer_size));
    }

    /// Push sample from a stream
    pub fn push(&mut self, sample: TimestampedSample) -> IoResult<()> {
        let buffer = self
            .buffers
            .entry(sample.stream_id.clone())
            .or_insert_with(|| VecDeque::with_capacity(self.config.buffer_size));

        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front(); // Remove oldest
        }

        buffer.push_back(sample);
        Ok(())
    }

    /// Try to get synchronized samples from all streams
    /// Returns HashMap of stream_id -> interpolated sample
    pub fn try_sync(&mut self) -> IoResult<HashMap<String, Array1<f32>>> {
        if self.buffers.is_empty() {
            return Err(IoError::InvalidConfig("No streams registered".to_string()));
        }

        // Find latest common timestamp where all streams have data
        let target_time = self.find_sync_time()?;

        // Interpolate samples for each stream at target time
        let mut synced = HashMap::new();
        for (stream_id, buffer) in &self.buffers {
            let sample = self.interpolate_at_time(buffer, target_time)?;
            synced.insert(stream_id.clone(), sample);
        }

        self.last_sync_time = Some(target_time);
        Ok(synced)
    }

    /// Find a timestamp where all streams have data
    fn find_sync_time(&self) -> IoResult<Timestamp> {
        // Get the latest minimum timestamp across all streams
        let mut min_max_time: Option<Timestamp> = None;

        for buffer in self.buffers.values() {
            if buffer.is_empty() {
                return Err(IoError::BufferEmpty);
            }

            let max_time = buffer
                .back()
                .expect("Buffer must be non-empty (checked above)")
                .timestamp;
            min_max_time = Some(match min_max_time {
                None => max_time,
                Some(current) => current.min(max_time),
            });
        }

        min_max_time.ok_or_else(|| IoError::BufferEmpty)
    }

    /// Interpolate sample at specific timestamp
    fn interpolate_at_time(
        &self,
        buffer: &VecDeque<TimestampedSample>,
        target_time: Timestamp,
    ) -> IoResult<Array1<f32>> {
        if buffer.is_empty() {
            return Err(IoError::BufferEmpty);
        }

        // Find samples before and after target time
        let mut before: Option<&TimestampedSample> = None;
        let mut after: Option<&TimestampedSample> = None;

        for sample in buffer {
            if sample.timestamp <= target_time {
                before = Some(sample);
            }
            if sample.timestamp >= target_time && after.is_none() {
                after = Some(sample);
                break;
            }
        }

        match self.config.interpolation {
            InterpolationMethod::Nearest => {
                // Use nearest sample
                let nearest = match (before, after) {
                    (Some(b), Some(a)) => {
                        let diff_before = target_time - b.timestamp;
                        let diff_after = a.timestamp - target_time;
                        if diff_before < diff_after {
                            b
                        } else {
                            a
                        }
                    }
                    (Some(b), None) => b,
                    (None, Some(a)) => a,
                    (None, None) => return Err(IoError::BufferEmpty),
                };
                Ok(nearest.data.clone())
            }
            InterpolationMethod::Linear => {
                match (before, after) {
                    (Some(b), Some(a)) if b.timestamp != a.timestamp => {
                        // Linear interpolation
                        let t =
                            (target_time - b.timestamp) as f32 / (a.timestamp - b.timestamp) as f32;
                        Ok(&b.data * (1.0 - t) + &a.data * t)
                    }
                    (Some(b), _) => Ok(b.data.clone()),
                    (None, Some(a)) => Ok(a.data.clone()),
                    _ => Err(IoError::BufferEmpty),
                }
            }
            InterpolationMethod::Hold => {
                // Use previous value
                match before {
                    Some(b) => Ok(b.data.clone()),
                    None => match after {
                        Some(a) => Ok(a.data.clone()),
                        None => Err(IoError::BufferEmpty),
                    },
                }
            }
        }
    }

    /// Get number of buffered samples for a stream
    pub fn buffer_len(&self, stream_id: &str) -> usize {
        self.buffers.get(stream_id).map(|b| b.len()).unwrap_or(0)
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.clear();
        }
        self.last_sync_time = None;
    }

    /// Get registered stream IDs
    pub fn stream_ids(&self) -> Vec<String> {
        self.buffers.keys().cloned().collect()
    }
}

impl Default for StreamSynchronizer {
    fn default() -> Self {
        Self::new(SyncConfig::default())
    }
}

/// Time synchronizer using NTP-like algorithm
pub struct TimeSynchronizer {
    /// Offset from system time (microseconds)
    offset: i64,
    /// Round-trip time (microseconds)
    rtt: u64,
    /// Synchronization samples
    samples: VecDeque<(i64, u64)>, // (offset, rtt)
    /// Maximum samples to keep
    max_samples: usize,
}

impl TimeSynchronizer {
    /// Create new time synchronizer
    pub fn new() -> Self {
        Self {
            offset: 0,
            rtt: 0,
            samples: VecDeque::with_capacity(10),
            max_samples: 10,
        }
    }

    /// Record synchronization sample
    pub fn add_sample(
        &mut self,
        send_time: Timestamp,
        recv_time: Timestamp,
        remote_time: Timestamp,
    ) {
        let t1 = send_time as i64;
        let t2 = remote_time as i64;
        let t3 = recv_time as i64;

        let offset = ((t2 - t1) + (t2 - t3)) / 2;
        let rtt = (t3 - t1) as u64;

        self.samples.push_back((offset, rtt));
        if self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }

        self.update_estimate();
    }

    /// Update offset and RTT estimate
    fn update_estimate(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        // Use median for robustness
        let mut offsets: Vec<i64> = self.samples.iter().map(|(o, _)| *o).collect();
        offsets.sort_unstable();
        self.offset = offsets[offsets.len() / 2];

        let mut rtts: Vec<u64> = self.samples.iter().map(|(_, r)| *r).collect();
        rtts.sort_unstable();
        self.rtt = rtts[rtts.len() / 2];
    }

    /// Get synchronized timestamp
    pub fn synchronized_time(&self) -> Timestamp {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time must be after UNIX_EPOCH")
            .as_micros() as u64;

        (now as i64 + self.offset) as u64
    }

    /// Get current offset
    pub fn offset(&self) -> i64 {
        self.offset
    }

    /// Get current RTT
    pub fn rtt(&self) -> u64 {
        self.rtt
    }
}

impl Default for TimeSynchronizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase-locked loop for maintaining stream synchronization
pub struct PhaseLockLoop {
    /// Target phase (timestamp difference)
    target_phase: i64,
    /// Current phase error integral
    integral: f64,
    /// Proportional gain
    kp: f64,
    /// Integral gain
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// Last error
    last_error: f64,
}

impl PhaseLockLoop {
    /// Create new PLL with gains
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            target_phase: 0,
            integral: 0.0,
            kp,
            ki,
            kd,
            last_error: 0.0,
        }
    }

    /// Update PLL with phase error and get correction
    pub fn update(&mut self, measured_phase: i64) -> f64 {
        let error = (self.target_phase - measured_phase) as f64;

        // PID control
        self.integral += error;
        let derivative = error - self.last_error;

        let correction = self.kp * error + self.ki * self.integral + self.kd * derivative;

        self.last_error = error;
        correction
    }

    /// Set target phase
    pub fn set_target_phase(&mut self, phase: i64) {
        self.target_phase = phase;
    }

    /// Reset integral term
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
    }
}

impl Default for PhaseLockLoop {
    fn default() -> Self {
        Self::new(1.0, 0.1, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamped_sample() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sample = TimestampedSample::now(data.clone(), "test".to_string());

        assert_eq!(sample.stream_id, "test");
        assert_eq!(sample.data, data);
        assert!(sample.timestamp > 0);
    }

    #[test]
    fn test_stream_synchronizer() {
        let mut sync = StreamSynchronizer::default();
        sync.add_stream("stream1".to_string());
        sync.add_stream("stream2".to_string());

        let base_time = 1000000u64;

        // Add samples to both streams
        for i in 0..5 {
            let time = base_time + i * 1000;
            let data1 = Array1::from_vec(vec![i as f32]);
            let data2 = Array1::from_vec(vec![(i * 2) as f32]);

            sync.push(TimestampedSample::new(time, data1, "stream1".to_string()))
                .unwrap();
            sync.push(TimestampedSample::new(time, data2, "stream2".to_string()))
                .unwrap();
        }

        // Should be able to synchronize
        let result = sync.try_sync();
        assert!(result.is_ok());

        let synced = result.unwrap();
        assert_eq!(synced.len(), 2);
        assert!(synced.contains_key("stream1"));
        assert!(synced.contains_key("stream2"));
    }

    #[test]
    fn test_time_synchronizer() {
        let mut sync = TimeSynchronizer::new();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Simulate time sync samples with realistic offset
        // Client sends at t1=now, server responds with t2=now+600 (server is ahead),
        // client receives at t3=now+1000
        sync.add_sample(now, now + 1000, now + 600);
        sync.add_sample(now + 10000, now + 11000, now + 10600);

        // Should have non-zero offset (server is ahead)
        assert!(sync.offset().abs() > 0);
        assert!(sync.rtt() > 0);
    }

    #[test]
    fn test_pll() {
        let mut pll = PhaseLockLoop::default();
        pll.set_target_phase(0);

        // Test convergence
        let correction1 = pll.update(100);
        let correction2 = pll.update(50);
        let correction3 = pll.update(10);

        // Corrections should decrease as we approach target
        assert!(correction1.abs() > correction2.abs());
        assert!(correction2.abs() > correction3.abs());
    }
}
