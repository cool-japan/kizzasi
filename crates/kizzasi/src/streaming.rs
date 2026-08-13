//! Async streaming prediction support
//!
//! Provides async interfaces for streaming signal prediction.

use crate::error::KizzasiResult;
use crate::predictor::Kizzasi;
use futures::Stream;
use scirs2_core::ndarray::Array1;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// Async signal stream that yields predictions
pub struct PredictionStream {
    predictor: Kizzasi,
    input_rx: mpsc::Receiver<Array1<f32>>,
    pending: Option<KizzasiResult<Array1<f32>>>,
}

impl PredictionStream {
    /// Create a new prediction stream
    ///
    /// Returns the stream and a sender to push inputs.
    pub fn new(predictor: Kizzasi, buffer_size: usize) -> (Self, mpsc::Sender<Array1<f32>>) {
        let (tx, rx) = mpsc::channel(buffer_size);
        let stream = Self {
            predictor,
            input_rx: rx,
            pending: None,
        };
        (stream, tx)
    }

    /// Create a bounded prediction stream
    pub fn bounded(predictor: Kizzasi) -> (Self, mpsc::Sender<Array1<f32>>) {
        Self::new(predictor, 16)
    }

    /// Get mutable access to the predictor
    pub fn predictor_mut(&mut self) -> &mut Kizzasi {
        &mut self.predictor
    }

    /// Reset the predictor state
    pub fn reset(&mut self) {
        self.predictor.reset();
    }
}

impl Stream for PredictionStream {
    type Item = KizzasiResult<Array1<f32>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check for pending result first
        if let Some(result) = self.pending.take() {
            return Poll::Ready(Some(result));
        }

        // Try to receive next input
        match self.input_rx.poll_recv(cx) {
            Poll::Ready(Some(input)) => {
                let result = self.predictor.step(&input);
                Poll::Ready(Some(result))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Async predictor wrapper with channel-based I/O
pub struct AsyncPredictor {
    /// `None` once [`AsyncPredictor::close`] has dropped the sender.
    input_tx: Option<mpsc::Sender<Array1<f32>>>,
    output_rx: mpsc::Receiver<KizzasiResult<Array1<f32>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl AsyncPredictor {
    /// Create a new async predictor
    ///
    /// Spawns a background task to process predictions.
    pub fn spawn(mut predictor: Kizzasi, buffer_size: usize) -> Self {
        let (input_tx, mut input_rx) = mpsc::channel::<Array1<f32>>(buffer_size);
        let (output_tx, output_rx) = mpsc::channel::<KizzasiResult<Array1<f32>>>(buffer_size);

        let handle = tokio::spawn(async move {
            while let Some(input) = input_rx.recv().await {
                let result = predictor.step(&input);
                if output_tx.send(result).await.is_err() {
                    break;
                }
            }
        });

        Self {
            input_tx: Some(input_tx),
            output_rx,
            handle,
        }
    }

    /// Send an input for prediction
    ///
    /// Returns `SendError` once [`Self::close`] has been called.
    pub async fn send(
        &self,
        input: Array1<f32>,
    ) -> Result<(), mpsc::error::SendError<Array1<f32>>> {
        match self.input_tx.as_ref() {
            Some(tx) => tx.send(input).await,
            None => Err(mpsc::error::SendError(input)),
        }
    }

    /// Whether the input channel is still open.
    pub fn is_open(&self) -> bool {
        self.input_tx.is_some()
    }

    /// Receive the next prediction
    pub async fn recv(&mut self) -> Option<KizzasiResult<Array1<f32>>> {
        self.output_rx.recv().await
    }

    /// Predict a single input (send + receive)
    pub async fn predict(&mut self, input: Array1<f32>) -> KizzasiResult<Array1<f32>> {
        self.send(input)
            .await
            .map_err(|e| crate::error::KizzasiError::Inference(format!("Send failed: {}", e)))?;
        self.recv()
            .await
            .ok_or_else(|| crate::error::KizzasiError::Inference("Channel closed".into()))?
    }

    /// Close the input channel; the worker drains what is queued and exits.
    ///
    /// This drops the only sender, which is what actually terminates the
    /// background task. (It used to be an empty function body with a comment
    /// explaining why dropping a *reference* would not work — so calling it
    /// did nothing at all and leaked the task.) Subsequent [`Self::send`]
    /// calls return `SendError`; queued outputs can still be drained with
    /// [`Self::recv`].
    pub fn close(&mut self) {
        self.input_tx = None;
    }

    /// Close the input channel and wait for the worker task to finish.
    pub async fn shutdown(mut self) -> Result<(), tokio::task::JoinError> {
        self.close();
        // Drop the output receiver so a worker blocked on `send` can finish.
        drop(self.output_rx);
        self.handle.await
    }
}

/// Stream adapter for processing signal streams asynchronously
pub struct StreamProcessor<S> {
    predictor: Kizzasi,
    source: S,
}

impl<S> StreamProcessor<S>
where
    S: Stream<Item = Array1<f32>> + Unpin,
{
    /// Create a new stream processor
    pub fn new(predictor: Kizzasi, source: S) -> Self {
        Self { predictor, source }
    }

    /// Get mutable access to the predictor
    pub fn predictor_mut(&mut self) -> &mut Kizzasi {
        &mut self.predictor
    }
}

impl<S> Stream for StreamProcessor<S>
where
    S: Stream<Item = Array1<f32>> + Unpin,
{
    type Item = KizzasiResult<Array1<f32>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let source = Pin::new(&mut self.source);
        match source.poll_next(cx) {
            Poll::Ready(Some(input)) => {
                let result = self.predictor.step(&input);
                Poll::Ready(Some(result))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_close_terminates_the_worker_task() {
        // Regression: `close()` had an empty body, so the background task
        // outlived every call to it.
        let predictor = crate::KizzasiBuilder::lightweight_preset(2, 2)
            .build()
            .expect("build");
        let mut async_predictor = AsyncPredictor::spawn(predictor, 4);

        async_predictor
            .predict(Array1::from_vec(vec![0.1, 0.2]))
            .await
            .expect("predict");

        assert!(async_predictor.is_open());
        async_predictor.close();
        assert!(!async_predictor.is_open());

        // Sending after close is an error, not a silent drop.
        assert!(async_predictor
            .send(Array1::from_vec(vec![0.3, 0.4]))
            .await
            .is_err());

        // And the worker actually finishes.
        async_predictor.shutdown().await.expect("worker joined");
    }
    use crate::KizzasiBuilder;
    use futures::StreamExt;
    use kizzasi_core::ModelType;

    #[tokio::test]
    async fn test_prediction_stream() {
        let predictor = KizzasiBuilder::new()
            .model_type(ModelType::Mamba2)
            .input_dim(2)
            .output_dim(2)
            .hidden_dim(16)
            .state_dim(4)
            .num_layers(1)
            .build()
            .unwrap();

        let (mut stream, tx) = PredictionStream::new(predictor, 4);

        // Send some inputs
        tx.send(Array1::from_vec(vec![0.1, 0.2])).await.unwrap();
        tx.send(Array1::from_vec(vec![0.3, 0.4])).await.unwrap();
        drop(tx); // Close channel

        // Collect outputs
        let mut outputs = Vec::new();
        while let Some(result) = stream.next().await {
            outputs.push(result.unwrap());
        }

        assert_eq!(outputs.len(), 2);
        for output in &outputs {
            assert_eq!(output.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_async_predictor() {
        let predictor = KizzasiBuilder::lightweight_preset(2, 2).build().unwrap();

        let mut async_pred = AsyncPredictor::spawn(predictor, 4);

        let input = Array1::from_vec(vec![0.1, 0.2]);
        let output = async_pred.predict(input).await.unwrap();

        assert_eq!(output.len(), 2);
    }
}
