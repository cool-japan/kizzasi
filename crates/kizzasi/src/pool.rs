//! Connection pooling for efficient I/O resource management.
//!
//! This module provides a generic connection pool that can be used to manage
//! connections to external systems (MQTT brokers, databases, file handles, etc.)
//! efficiently.
//!
//! # Features
//!
//! - Generic connection type support
//! - Configurable pool size (min/max connections)
//! - Connection health checking
//! - Idle connection timeout
//! - Connection lifecycle hooks
//! - Pool statistics and metrics
//!
//! # Example
//!
//! ```rust,no_run
//! use kizzasi::pool::{ConnectionPool, PoolConfig, ConnectionFactory};
//! use std::sync::Arc;
//!
//! // Define your connection type
//! struct MyConnection {
//!     id: usize,
//! }
//!
//! // Implement the factory
//! struct MyFactory;
//!
//! #[async_trait::async_trait]
//! impl ConnectionFactory<MyConnection> for MyFactory {
//!     async fn create(&self) -> Result<MyConnection, Box<dyn std::error::Error + Send + Sync>> {
//!         Ok(MyConnection { id: 0 })
//!     }
//!
//!     async fn validate(&self, _conn: &MyConnection) -> bool {
//!         true
//!     }
//! }
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let config = PoolConfig::default()
//!     .with_min_connections(2)
//!     .with_max_connections(10);
//!
//! let pool = ConnectionPool::new(Arc::new(MyFactory), config).await?;
//! let conn = pool.acquire().await?;
//! // Use connection...
//! pool.release(conn).await;
//! # Ok(())
//! # }
//! ```

use crate::error::{KizzasiError, KizzasiResult};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

/// Factory trait for creating and validating connections.
#[async_trait::async_trait]
pub trait ConnectionFactory<T: Send + 'static>: Send + Sync {
    /// Create a new connection.
    async fn create(&self) -> Result<T, Box<dyn std::error::Error + Send + Sync>>;

    /// Validate that a connection is still healthy.
    async fn validate(&self, conn: &T) -> bool;

    /// Optional cleanup when a connection is destroyed.
    async fn destroy(&self, _conn: T) {
        // Default: no cleanup
    }
}

/// Configuration for connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections to maintain.
    pub min_connections: usize,

    /// Maximum number of connections allowed.
    pub max_connections: usize,

    /// Maximum time a connection can be idle before being closed.
    pub idle_timeout: Duration,

    /// Maximum time to wait for a connection to become available.
    pub acquire_timeout: Duration,

    /// Whether to validate connections before acquiring.
    pub validate_on_acquire: bool,

    /// Whether to validate connections before releasing back to pool.
    pub validate_on_release: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 10,
            idle_timeout: Duration::from_secs(300), // 5 minutes
            acquire_timeout: Duration::from_secs(30),
            validate_on_acquire: true,
            validate_on_release: false,
        }
    }
}

impl PoolConfig {
    /// Create a new pool configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum number of connections.
    pub fn with_min_connections(mut self, min: usize) -> Self {
        self.min_connections = min;
        self
    }

    /// Set maximum number of connections.
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Set idle timeout.
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Set acquire timeout.
    pub fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// Enable/disable validation on acquire.
    pub fn with_validate_on_acquire(mut self, validate: bool) -> Self {
        self.validate_on_acquire = validate;
        self
    }

    /// Enable/disable validation on release.
    pub fn with_validate_on_release(mut self, validate: bool) -> Self {
        self.validate_on_release = validate;
        self
    }
}

/// A pooled connection with metadata.
struct PooledConnection<T> {
    connection: Option<T>,
    created_at: Instant,
    last_used: Instant,
}

impl<T> PooledConnection<T> {
    fn new(connection: T) -> Self {
        let now = Instant::now();
        Self {
            connection: Some(connection),
            created_at: now,
            last_used: now,
        }
    }

    fn is_idle_expired(&self, timeout: Duration) -> bool {
        self.last_used.elapsed() > timeout
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    fn take(mut self) -> T {
        self.connection.take().expect("Connection already taken")
    }

    #[allow(dead_code)]
    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Statistics for the connection pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Total number of connections created.
    pub total_created: usize,

    /// Total number of connections destroyed.
    pub total_destroyed: usize,

    /// Number of currently active connections (in use).
    pub active_connections: usize,

    /// Number of idle connections available in pool.
    pub idle_connections: usize,

    /// Total number of successful acquires.
    pub total_acquires: usize,

    /// Total number of failed acquires (timeout).
    pub total_acquire_failures: usize,

    /// Total number of releases.
    pub total_releases: usize,
}

impl PoolStats {
    /// Total connections (active + idle).
    pub fn total_connections(&self) -> usize {
        self.active_connections + self.idle_connections
    }
}

/// Inner state of the connection pool.
struct PoolState<T> {
    idle: VecDeque<PooledConnection<T>>,
    stats: PoolStats,
}

/// A connection pool for managing reusable connections.
pub struct ConnectionPool<T> {
    factory: Arc<dyn ConnectionFactory<T>>,
    config: PoolConfig,
    state: Arc<Mutex<PoolState<T>>>,
    semaphore: Arc<Semaphore>,
}

impl<T: Send + 'static> ConnectionPool<T> {
    /// Create a new connection pool with the given factory and configuration.
    pub async fn new(
        factory: Arc<dyn ConnectionFactory<T>>,
        config: PoolConfig,
    ) -> KizzasiResult<Self> {
        if config.min_connections > config.max_connections {
            return Err(KizzasiError::invalid_state(
                "min_connections cannot exceed max_connections",
            ));
        }

        let state = Arc::new(Mutex::new(PoolState {
            idle: VecDeque::new(),
            stats: PoolStats::default(),
        }));

        let semaphore = Arc::new(Semaphore::new(config.max_connections));

        let pool = Self {
            factory,
            config,
            state,
            semaphore,
        };

        // Pre-fill with minimum connections
        pool.ensure_min_connections().await?;

        Ok(pool)
    }

    /// Acquire a connection from the pool.
    ///
    /// This will either return an idle connection from the pool or create a new one
    /// if the pool is not at capacity. If the pool is at capacity and no idle connections
    /// are available, this will wait up to `acquire_timeout` for a connection to become available.
    pub async fn acquire(&self) -> KizzasiResult<T> {
        let acquire_start = Instant::now();

        // Wait for semaphore permit (respects max_connections)
        let permit = tokio::time::timeout(
            self.config.acquire_timeout,
            self.semaphore.acquire(),
        )
        .await
        .map_err(|_| {
            KizzasiError::resource_exhausted(
                "connection pool",
                self.config.max_connections,
                self.config.max_connections,
                format!("Failed to acquire connection within {:?}. Try increasing max_connections or acquire_timeout", self.config.acquire_timeout),
            )
        })?
        .map_err(|e| KizzasiError::invalid_state(format!("Semaphore error: {}", e)))?;

        permit.forget(); // We'll manually release later

        // Try to get an idle connection
        let mut conn = self.try_acquire_idle().await;

        // If no idle connection, create a new one
        if conn.is_none() {
            match self.factory.create().await {
                Ok(c) => {
                    let mut state = self.state.lock().await;
                    state.stats.total_created += 1;
                    conn = Some(c);
                }
                Err(e) => {
                    self.semaphore.add_permits(1); // Return permit
                    let mut state = self.state.lock().await;
                    state.stats.total_acquire_failures += 1;
                    return Err(KizzasiError::invalid_state(format!(
                        "Failed to create connection: {}",
                        e
                    )));
                }
            }
        }

        let conn = conn.expect("invariant: conn is Some after create or early return on error");

        // Validate if configured
        if self.config.validate_on_acquire && !self.factory.validate(&conn).await {
            self.factory.destroy(conn).await;
            self.semaphore.add_permits(1);
            let mut state = self.state.lock().await;
            state.stats.total_destroyed += 1;
            state.stats.total_acquire_failures += 1;
            return Err(KizzasiError::invalid_state("Connection validation failed"));
        }

        // Update stats
        {
            let mut state = self.state.lock().await;
            state.stats.total_acquires += 1;
            state.stats.active_connections += 1;
        }

        tracing::debug!("Acquired connection in {:?}", acquire_start.elapsed());

        Ok(conn)
    }

    /// Release a connection back to the pool.
    pub async fn release(&self, conn: T) {
        // Validate if configured
        if self.config.validate_on_release && !self.factory.validate(&conn).await {
            self.factory.destroy(conn).await;
            self.semaphore.add_permits(1);
            let mut state = self.state.lock().await;
            state.stats.total_destroyed += 1;
            state.stats.active_connections = state.stats.active_connections.saturating_sub(1);
            return;
        }

        // Add back to idle pool
        let mut pooled = PooledConnection::new(conn);
        pooled.touch();

        {
            let mut state = self.state.lock().await;
            state.idle.push_back(pooled);
            state.stats.total_releases += 1;
            state.stats.idle_connections += 1;
            state.stats.active_connections = state.stats.active_connections.saturating_sub(1);
        }

        self.semaphore.add_permits(1);
    }

    /// Get current pool statistics.
    pub async fn stats(&self) -> PoolStats {
        let state = self.state.lock().await;
        state.stats
    }

    /// Ensure minimum number of connections are created.
    async fn ensure_min_connections(&self) -> KizzasiResult<()> {
        let current_count = {
            let state = self.state.lock().await;
            state.stats.total_connections()
        };

        for _ in current_count..self.config.min_connections {
            match self.factory.create().await {
                Ok(conn) => {
                    let mut state = self.state.lock().await;
                    state.idle.push_back(PooledConnection::new(conn));
                    state.stats.total_created += 1;
                    state.stats.idle_connections += 1;
                }
                Err(e) => {
                    tracing::warn!("Failed to create min connection: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Try to acquire an idle connection, removing expired ones.
    async fn try_acquire_idle(&self) -> Option<T> {
        let mut state = self.state.lock().await;

        // Remove expired connections
        while let Some(pooled) = state.idle.front() {
            if pooled.is_idle_expired(self.config.idle_timeout) {
                if let Some(expired) = state.idle.pop_front() {
                    let conn = expired.take();

                    state.stats.total_destroyed += 1;
                    state.stats.idle_connections = state.stats.idle_connections.saturating_sub(1);

                    // Destroy outside the lock
                    drop(state);
                    self.factory.destroy(conn).await;
                    state = self.state.lock().await;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Get an idle connection
        if let Some(mut pooled) = state.idle.pop_front() {
            state.stats.idle_connections = state.stats.idle_connections.saturating_sub(1);
            pooled.touch();
            Some(pooled.take())
        } else {
            None
        }
    }

    /// Shrink the pool by removing idle connections above minimum.
    pub async fn shrink(&self) {
        let mut state = self.state.lock().await;

        while state.stats.idle_connections > self.config.min_connections {
            if let Some(pooled) = state.idle.pop_back() {
                let conn = pooled.take();

                state.stats.idle_connections = state.stats.idle_connections.saturating_sub(1);
                state.stats.total_destroyed += 1;

                drop(state);
                self.factory.destroy(conn).await;
                self.semaphore.add_permits(1);
                state = self.state.lock().await;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestConnection {
        id: usize,
        valid: Arc<Mutex<bool>>,
    }

    struct TestFactory {
        counter: Arc<AtomicUsize>,
        create_delay: Duration,
    }

    #[async_trait::async_trait]
    impl ConnectionFactory<TestConnection> for TestFactory {
        async fn create(&self) -> Result<TestConnection, Box<dyn std::error::Error + Send + Sync>> {
            tokio::time::sleep(self.create_delay).await;
            let id = self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(TestConnection {
                id,
                valid: Arc::new(Mutex::new(true)),
            })
        }

        async fn validate(&self, conn: &TestConnection) -> bool {
            *conn.valid.lock().await
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let factory = Arc::new(TestFactory {
            counter: Arc::new(AtomicUsize::new(0)),
            create_delay: Duration::from_millis(1),
        });

        let config = PoolConfig::default()
            .with_min_connections(2)
            .with_max_connections(5);

        let pool = ConnectionPool::new(factory, config).await.unwrap();
        let stats = pool.stats().await;

        assert_eq!(stats.idle_connections, 2);
        assert_eq!(stats.total_created, 2);
    }

    #[tokio::test]
    async fn test_acquire_release() {
        let factory = Arc::new(TestFactory {
            counter: Arc::new(AtomicUsize::new(0)),
            create_delay: Duration::from_millis(1),
        });

        let config = PoolConfig::default()
            .with_min_connections(1)
            .with_max_connections(3);

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        let conn1 = pool.acquire().await.unwrap();
        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.idle_connections, 0);

        pool.release(conn1).await;
        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.idle_connections, 1);
    }

    #[tokio::test]
    async fn test_max_connections() {
        let factory = Arc::new(TestFactory {
            counter: Arc::new(AtomicUsize::new(0)),
            create_delay: Duration::from_millis(1),
        });

        let config = PoolConfig::default()
            .with_min_connections(0)
            .with_max_connections(2)
            .with_acquire_timeout(Duration::from_millis(100));

        let pool = Arc::new(ConnectionPool::new(factory, config).await.unwrap());

        let conn1 = pool.acquire().await.unwrap();
        let conn2 = pool.acquire().await.unwrap();

        // Third acquire should timeout
        let pool_clone = pool.clone();
        let result = tokio::spawn(async move { pool_clone.acquire().await })
            .await
            .unwrap();

        assert!(result.is_err());

        // Release and retry
        pool.release(conn1).await;
        let conn3 = pool.acquire().await.unwrap();
        assert!(conn3.id < 2); // Should reuse connection

        pool.release(conn2).await;
        pool.release(conn3).await;
    }

    #[tokio::test]
    async fn test_validation() {
        let factory = Arc::new(TestFactory {
            counter: Arc::new(AtomicUsize::new(0)),
            create_delay: Duration::from_millis(1),
        });

        let config = PoolConfig::default()
            .with_min_connections(1)
            .with_max_connections(3)
            .with_validate_on_acquire(true);

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        let conn = pool.acquire().await.unwrap();
        *conn.valid.lock().await = false; // Invalidate
        pool.release(conn).await;

        // Next acquire should fail validation and create new connection
        let result = pool.acquire().await;
        assert!(result.is_err()); // Validation failed
    }

    #[tokio::test]
    async fn test_shrink() {
        let factory = Arc::new(TestFactory {
            counter: Arc::new(AtomicUsize::new(0)),
            create_delay: Duration::from_millis(1),
        });

        let config = PoolConfig::default()
            .with_min_connections(1)
            .with_max_connections(5);

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Acquire and release multiple connections
        let mut conns = vec![];
        for _ in 0..5 {
            conns.push(pool.acquire().await.unwrap());
        }

        for conn in conns {
            pool.release(conn).await;
        }

        let stats = pool.stats().await;
        assert!(stats.idle_connections >= 5);

        pool.shrink().await;

        let stats = pool.stats().await;
        assert_eq!(stats.idle_connections, 1); // Down to min
    }
}
