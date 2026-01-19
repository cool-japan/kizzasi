//! Stream multiplexing and demultiplexing utilities
//!
//! This module provides tools for combining multiple streams into one
//! and splitting one stream into multiple outputs.

use crate::error::{IoError, IoResult};
use crate::sync::{Timestamp, TimestampedSample};
use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc;

/// Strategy for multiplexing multiple streams
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiplexStrategy {
    /// Interleave samples round-robin
    RoundRobin,
    /// Merge by timestamp order
    TimeOrdered,
    /// Concatenate streams sequentially
    Sequential,
    /// Weighted round-robin
    Weighted,
}

/// Multiplexer configuration
#[derive(Debug, Clone)]
pub struct MultiplexConfig {
    /// Multiplexing strategy
    pub strategy: MultiplexStrategy,
    /// Buffer size per input stream
    pub buffer_size: usize,
    /// Weights for weighted round-robin (stream_id -> weight)
    pub weights: HashMap<String, u32>,
}

impl Default for MultiplexConfig {
    fn default() -> Self {
        Self {
            strategy: MultiplexStrategy::RoundRobin,
            buffer_size: 1024,
            weights: HashMap::new(),
        }
    }
}

/// Stream multiplexer - combines multiple streams into one
pub struct StreamMultiplexer {
    config: MultiplexConfig,
    buffers: HashMap<String, VecDeque<TimestampedSample>>,
    round_robin_index: usize,
    stream_ids: Vec<String>,
}

impl StreamMultiplexer {
    /// Create new multiplexer
    pub fn new(config: MultiplexConfig) -> Self {
        Self {
            config,
            buffers: HashMap::new(),
            round_robin_index: 0,
            stream_ids: Vec::new(),
        }
    }

    /// Add input stream
    pub fn add_stream(&mut self, stream_id: String) {
        if !self.buffers.contains_key(&stream_id) {
            self.buffers.insert(
                stream_id.clone(),
                VecDeque::with_capacity(self.config.buffer_size),
            );
            self.stream_ids.push(stream_id);
        }
    }

    /// Push sample from a stream
    pub fn push(&mut self, sample: TimestampedSample) -> IoResult<()> {
        let buffer = self.buffers.get_mut(&sample.stream_id).ok_or_else(|| {
            IoError::InvalidConfig(format!("Unknown stream: {}", sample.stream_id))
        })?;

        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front(); // Remove oldest
        }

        // For TimeOrdered strategy, insert in sorted order by timestamp
        if self.config.strategy == MultiplexStrategy::TimeOrdered {
            // Find insertion position to maintain ascending timestamp order
            let pos = buffer
                .iter()
                .position(|s| s.timestamp > sample.timestamp)
                .unwrap_or(buffer.len());
            buffer.insert(pos, sample);
        } else {
            buffer.push_back(sample);
        }
        Ok(())
    }

    /// Get next multiplexed sample
    pub fn next_sample(&mut self) -> Option<TimestampedSample> {
        match self.config.strategy {
            MultiplexStrategy::RoundRobin => self.next_round_robin(),
            MultiplexStrategy::TimeOrdered => self.next_time_ordered(),
            MultiplexStrategy::Sequential => self.next_sequential(),
            MultiplexStrategy::Weighted => self.next_weighted(),
        }
    }

    fn next_round_robin(&mut self) -> Option<TimestampedSample> {
        if self.stream_ids.is_empty() {
            return None;
        }

        let start_index = self.round_robin_index;
        loop {
            let stream_id = &self.stream_ids[self.round_robin_index];
            self.round_robin_index = (self.round_robin_index + 1) % self.stream_ids.len();

            if let Some(buffer) = self.buffers.get_mut(stream_id) {
                if let Some(sample) = buffer.pop_front() {
                    return Some(sample);
                }
            }

            // Check if we've looped back without finding anything
            if self.round_robin_index == start_index {
                break;
            }
        }

        None
    }

    fn next_time_ordered(&mut self) -> Option<TimestampedSample> {
        let mut earliest: Option<(String, Timestamp)> = None;

        // Find stream with earliest timestamp
        for (stream_id, buffer) in &self.buffers {
            if let Some(sample) = buffer.front() {
                match earliest {
                    None => earliest = Some((stream_id.clone(), sample.timestamp)),
                    Some((_, min_ts)) if sample.timestamp < min_ts => {
                        earliest = Some((stream_id.clone(), sample.timestamp));
                    }
                    _ => {}
                }
            }
        }

        // Pop from that stream
        earliest.and_then(|(stream_id, _)| self.buffers.get_mut(&stream_id)?.pop_front())
    }

    fn next_sequential(&mut self) -> Option<TimestampedSample> {
        // Drain first stream completely before moving to next
        for stream_id in &self.stream_ids {
            if let Some(buffer) = self.buffers.get_mut(stream_id) {
                if let Some(sample) = buffer.pop_front() {
                    return Some(sample);
                }
            }
        }
        None
    }

    fn next_weighted(&mut self) -> Option<TimestampedSample> {
        // Weighted round-robin based on configured weights
        if self.stream_ids.is_empty() {
            return None;
        }

        let total_weight: u32 = self
            .stream_ids
            .iter()
            .map(|id| self.config.weights.get(id).copied().unwrap_or(1))
            .sum();

        if total_weight == 0 {
            return self.next_round_robin();
        }

        // Try streams proportional to their weights
        for _ in 0..total_weight {
            let stream_id = &self.stream_ids[self.round_robin_index % self.stream_ids.len()];
            let weight = self.config.weights.get(stream_id).copied().unwrap_or(1);

            self.round_robin_index += 1;

            if let Some(buffer) = self.buffers.get_mut(stream_id) {
                if !buffer.is_empty() && weight > 0 {
                    return buffer.pop_front();
                }
            }
        }

        None
    }

    /// Get number of buffered samples for a stream
    pub fn buffered(&self, stream_id: &str) -> usize {
        self.buffers.get(stream_id).map(|b| b.len()).unwrap_or(0)
    }

    /// Get total buffered samples across all streams
    pub fn total_buffered(&self) -> usize {
        self.buffers.values().map(|b| b.len()).sum()
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.clear();
        }
    }
}

impl Default for StreamMultiplexer {
    fn default() -> Self {
        Self::new(MultiplexConfig::default())
    }
}

/// Stream demultiplexer - splits one stream into multiple outputs
pub struct StreamDemultiplexer<F>
where
    F: Fn(&TimestampedSample) -> String,
{
    router: F,
    buffers: HashMap<String, VecDeque<TimestampedSample>>,
    buffer_size: usize,
}

impl<F> StreamDemultiplexer<F>
where
    F: Fn(&TimestampedSample) -> String,
{
    /// Create new demultiplexer with routing function
    pub fn new(router: F, buffer_size: usize) -> Self {
        Self {
            router,
            buffers: HashMap::new(),
            buffer_size,
        }
    }

    /// Push sample (will be routed to appropriate output)
    pub fn push(&mut self, sample: TimestampedSample) {
        let output_id = (self.router)(&sample);

        let buffer = self
            .buffers
            .entry(output_id)
            .or_insert_with(|| VecDeque::with_capacity(self.buffer_size));

        if buffer.len() >= self.buffer_size {
            buffer.pop_front();
        }

        buffer.push_back(sample);
    }

    /// Get samples from specific output
    pub fn pop(&mut self, output_id: &str) -> Option<TimestampedSample> {
        self.buffers.get_mut(output_id)?.pop_front()
    }

    /// Get all samples from specific output
    pub fn drain(&mut self, output_id: &str) -> Vec<TimestampedSample> {
        self.buffers
            .get_mut(output_id)
            .map(|b| b.drain(..).collect())
            .unwrap_or_default()
    }

    /// Get number of buffered samples for output
    pub fn buffered(&self, output_id: &str) -> usize {
        self.buffers.get(output_id).map(|b| b.len()).unwrap_or(0)
    }

    /// Get all output IDs
    pub fn output_ids(&self) -> Vec<String> {
        self.buffers.keys().cloned().collect()
    }

    /// Clear all buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Async multiplexer using channels
pub struct AsyncMultiplexer {
    receivers: Vec<mpsc::Receiver<TimestampedSample>>,
    strategy: MultiplexStrategy,
}

impl AsyncMultiplexer {
    /// Create new async multiplexer
    pub fn new(strategy: MultiplexStrategy) -> Self {
        Self {
            receivers: Vec::new(),
            strategy,
        }
    }

    /// Add input channel
    pub fn add_receiver(&mut self, receiver: mpsc::Receiver<TimestampedSample>) {
        self.receivers.push(receiver);
    }

    /// Get next multiplexed sample (async)
    pub async fn next(&mut self) -> Option<TimestampedSample> {
        match self.strategy {
            MultiplexStrategy::RoundRobin => self.next_round_robin().await,
            MultiplexStrategy::TimeOrdered => {
                // Time-ordered requires buffering, not efficient for async
                self.next_round_robin().await
            }
            _ => self.next_round_robin().await,
        }
    }

    async fn next_round_robin(&mut self) -> Option<TimestampedSample> {
        for receiver in &mut self.receivers {
            if let Ok(sample) = receiver.try_recv() {
                return Some(sample);
            }
        }
        None
    }

    /// Get number of input streams
    pub fn num_streams(&self) -> usize {
        self.receivers.len()
    }
}

/// Channel-based stream splitter
pub struct ChannelSplitter {
    senders: HashMap<String, mpsc::Sender<TimestampedSample>>,
}

impl ChannelSplitter {
    /// Create new channel splitter
    pub fn new() -> Self {
        Self {
            senders: HashMap::new(),
        }
    }

    /// Add output channel
    pub fn add_output(&mut self, output_id: String, sender: mpsc::Sender<TimestampedSample>) {
        self.senders.insert(output_id, sender);
    }

    /// Send sample to specific output
    pub async fn send(&self, output_id: &str, sample: TimestampedSample) -> IoResult<()> {
        let sender = self
            .senders
            .get(output_id)
            .ok_or_else(|| IoError::InvalidConfig(format!("Unknown output: {}", output_id)))?;

        sender
            .send(sample)
            .await
            .map_err(|_| IoError::SendFailed("Channel send failed".to_string()))
    }

    /// Broadcast sample to all outputs
    pub async fn broadcast(&self, sample: TimestampedSample) -> IoResult<()> {
        for sender in self.senders.values() {
            sender
                .send(sample.clone())
                .await
                .map_err(|_| IoError::SendFailed("Broadcast failed".to_string()))?;
        }
        Ok(())
    }

    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.senders.len()
    }
}

impl Default for ChannelSplitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_multiplexer_round_robin() {
        let mut mux = StreamMultiplexer::default();

        mux.add_stream("stream1".to_string());
        mux.add_stream("stream2".to_string());

        // Add samples
        for i in 0..5 {
            let sample1 = TimestampedSample::new(
                i * 1000,
                Array1::from_vec(vec![i as f32]),
                "stream1".to_string(),
            );
            let sample2 = TimestampedSample::new(
                i * 1000 + 500,
                Array1::from_vec(vec![(i + 10) as f32]),
                "stream2".to_string(),
            );

            mux.push(sample1).unwrap();
            mux.push(sample2).unwrap();
        }

        // Should alternate between streams
        let s1 = mux.next_sample().unwrap();
        assert_eq!(s1.stream_id, "stream1");

        let s2 = mux.next_sample().unwrap();
        assert_eq!(s2.stream_id, "stream2");
    }

    #[test]
    fn test_demultiplexer() {
        let router = |sample: &TimestampedSample| {
            if sample.data[0] > 5.0 {
                "high".to_string()
            } else {
                "low".to_string()
            }
        };

        let mut demux = StreamDemultiplexer::new(router, 100);

        // Push samples
        demux.push(TimestampedSample::new(
            0,
            Array1::from_vec(vec![3.0]),
            "test".to_string(),
        ));
        demux.push(TimestampedSample::new(
            1000,
            Array1::from_vec(vec![7.0]),
            "test".to_string(),
        ));
        demux.push(TimestampedSample::new(
            2000,
            Array1::from_vec(vec![2.0]),
            "test".to_string(),
        ));

        assert_eq!(demux.buffered("low"), 2);
        assert_eq!(demux.buffered("high"), 1);

        let low_sample = demux.pop("low").unwrap();
        assert_eq!(low_sample.data[0], 3.0);
    }

    #[test]
    fn test_multiplexer_time_ordered() {
        let config = MultiplexConfig {
            strategy: MultiplexStrategy::TimeOrdered,
            ..Default::default()
        };

        let mut mux = StreamMultiplexer::new(config);

        mux.add_stream("s1".to_string());
        mux.add_stream("s2".to_string());

        // Add samples with different timestamps
        mux.push(TimestampedSample::new(
            3000,
            Array1::from_vec(vec![3.0]),
            "s1".to_string(),
        ))
        .unwrap();
        mux.push(TimestampedSample::new(
            1000,
            Array1::from_vec(vec![1.0]),
            "s2".to_string(),
        ))
        .unwrap();
        mux.push(TimestampedSample::new(
            2000,
            Array1::from_vec(vec![2.0]),
            "s1".to_string(),
        ))
        .unwrap();

        // Should come out in time order
        let first = mux.next_sample().unwrap();
        assert_eq!(first.timestamp, 1000);
        assert_eq!(first.data[0], 1.0);

        let second = mux.next_sample().unwrap();
        assert_eq!(second.timestamp, 2000);
        assert_eq!(second.data[0], 2.0);

        let third = mux.next_sample().unwrap();
        assert_eq!(third.timestamp, 3000);
        assert_eq!(third.data[0], 3.0);
    }

    #[tokio::test]
    async fn test_channel_splitter() {
        let mut splitter = ChannelSplitter::new();

        let (tx1, mut rx1) = mpsc::channel(10);
        let (tx2, mut rx2) = mpsc::channel(10);

        splitter.add_output("out1".to_string(), tx1);
        splitter.add_output("out2".to_string(), tx2);

        let sample = TimestampedSample::new(0, Array1::from_vec(vec![1.0]), "test".to_string());

        // Send to specific output
        splitter.send("out1", sample.clone()).await.unwrap();

        let received = rx1.recv().await.unwrap();
        assert_eq!(received.data[0], 1.0);

        // Broadcast
        let sample2 = TimestampedSample::new(1000, Array1::from_vec(vec![2.0]), "test".to_string());
        splitter.broadcast(sample2).await.unwrap();

        assert!(rx1.recv().await.is_some());
        assert!(rx2.recv().await.is_some());
    }
}
