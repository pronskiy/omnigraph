//! The public HTTP listener. Each accepted socket is one HTTP/1 connection,
//! and a peer holds one only while it keeps to three bounds: it sends each
//! request's headers, the first one included, within [`HEADER_READ_TIMEOUT`]
//! of the connection becoming ready to read them; it takes the response's
//! bytes, so no write waits on it longer than [`WRITE_STALL_TIMEOUT`]; and the
//! process holds at most a bounded number of connections at once. A peer that
//! connects and says nothing, or stops reading a response, is closed at the
//! deadline, and the response body with everything it holds (a graph request
//! a deployment drains, a stream's export slot) is dropped with it. A peer
//! past the cap waits in the kernel's accept queue instead of taking a
//! descriptor and a task.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use color_eyre::eyre::{Result, eyre};
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, watch};
use tracing::{debug, error};

/// How long a connection may take to deliver a request's complete headers,
/// counted from when it is ready to read them: from accept for the first
/// request, from the end of the previous response on a kept-alive connection,
/// so it is also the keep-alive idle limit. `OMNIGRAPH_HEADER_TIMEOUT_SECONDS`
/// sets it.
pub const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// The environment variable that sets [`HEADER_READ_TIMEOUT`], in seconds.
pub const HEADER_TIMEOUT_ENV: &str = "OMNIGRAPH_HEADER_TIMEOUT_SECONDS";

/// How long one write to a connection may wait on a peer that is not reading
/// before the connection is closed. `OMNIGRAPH_WRITE_STALL_TIMEOUT_SECONDS`
/// sets it.
pub const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// The environment variable that sets [`WRITE_STALL_TIMEOUT`], in seconds.
pub const WRITE_STALL_TIMEOUT_ENV: &str = "OMNIGRAPH_WRITE_STALL_TIMEOUT_SECONDS";

/// The largest connection cap accepted: far above any descriptor limit, and
/// below what one semaphore can hold.
const MAX_CONNECTIONS_CEILING: usize = 1_000_000;

/// The environment variable that sets the most connections the listener
/// holds at once.
pub const MAX_CONNECTIONS_ENV: &str = "OMNIGRAPH_MAX_CONNECTIONS";

/// The bounds one listener serves under.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ListenerLimits {
    pub(crate) header_read_timeout: Duration,
    pub(crate) write_stall_timeout: Duration,
    pub(crate) max_connections: usize,
}

impl ListenerLimits {
    /// The process's bounds: [`HEADER_READ_TIMEOUT`] and
    /// [`resolve_max_connections`].
    pub(crate) fn from_env() -> Result<Self> {
        Ok(Self {
            header_read_timeout: resolve_seconds(
                HEADER_TIMEOUT_ENV,
                std::env::var(HEADER_TIMEOUT_ENV).ok(),
                HEADER_READ_TIMEOUT,
            )?,
            write_stall_timeout: resolve_seconds(
                WRITE_STALL_TIMEOUT_ENV,
                std::env::var(WRITE_STALL_TIMEOUT_ENV).ok(),
                WRITE_STALL_TIMEOUT,
            )?,
            max_connections: resolve_max_connections(std::env::var(MAX_CONNECTIONS_ENV).ok())?,
        })
    }
}

/// The connection cap: `OMNIGRAPH_MAX_CONNECTIONS` when set, a positive
/// whole number, else half the process's open-file limit (at least 16, at
/// most 4096), which leaves the other half to storage I/O.
pub(crate) fn resolve_max_connections(env_value: Option<String>) -> Result<usize> {
    match env_value {
        Some(value) => match value.trim().parse::<usize>() {
            Ok(limit) if (1..=MAX_CONNECTIONS_CEILING).contains(&limit) => Ok(limit),
            _ => Err(eyre!(
                "{MAX_CONNECTIONS_ENV} must be a whole number from 1 to {MAX_CONNECTIONS_CEILING}, got `{value}`"
            )),
        },
        None => Ok(default_max_connections()),
    }
}

/// A deadline the variable `name` sets: a whole number of seconds from 1 to
/// 3600 when set, else `default`.
pub(crate) fn resolve_seconds(
    name: &str,
    env_value: Option<String>,
    default: Duration,
) -> Result<Duration> {
    match env_value {
        Some(value) => match value.trim().parse::<u64>() {
            Ok(seconds) if (1..=3600).contains(&seconds) => Ok(Duration::from_secs(seconds)),
            _ => Err(eyre!(
                "{name} must be a whole number of seconds from 1 to 3600, got `{value}`"
            )),
        },
        None => Ok(default),
    }
}

fn default_max_connections() -> usize {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes only the struct it is given.
    let open_files = if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0 {
        usize::try_from(limit.rlim_cur).unwrap_or(usize::MAX)
    } else {
        1024
    };
    (open_files / 2).clamp(16, 4096)
}

/// Serve `app` on `listener` until `shutdown` resolves, then stop accepting,
/// let every open connection finish its in-flight request, and return once
/// all have closed.
pub(crate) async fn serve(
    listener: TcpListener,
    app: Router,
    limits: ListenerLimits,
    shutdown: impl Future<Output = ()>,
) -> io::Result<()> {
    let slots = Arc::new(Semaphore::new(limits.max_connections));
    let (stop_tx, stop_rx) = watch::channel(false);
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        // A slot first, then the socket: past the cap a peer waits in the
        // accept queue and holds no descriptor of this process.
        let slot = tokio::select! {
            () = &mut shutdown => break,
            slot = Arc::clone(&slots).acquire_owned() => {
                slot.expect("the listener never closes its connection slots")
            }
        };
        let stream = tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(error) => {
                    accept_failed(error).await;
                    continue;
                }
            },
        };
        let app = app.clone();
        let mut stop = stop_rx.clone();
        tokio::spawn(async move {
            let _slot = slot;
            let connection = http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(limits.header_read_timeout)
                .serve_connection(
                    TokioIo::new(WriteStall::new(stream, limits.write_stall_timeout)),
                    TowerToHyperService::new(app),
                );
            let mut connection = std::pin::pin!(connection);
            let stopping = tokio::select! {
                served = connection.as_mut() => Some(served),
                _ = stop.wait_for(|stopping| *stopping) => None,
            };
            let served = match stopping {
                Some(served) => served,
                None => {
                    connection.as_mut().graceful_shutdown();
                    connection.await
                }
            };
            if let Err(error) = served {
                debug!(%error, "HTTP connection ended with an error");
            }
        });
    }
    drop(listener);
    let _ = stop_tx.send(true);
    let all = u32::try_from(limits.max_connections).unwrap_or(u32::MAX);
    let _drained = slots
        .acquire_many(all)
        .await
        .expect("the listener never closes its connection slots");
    Ok(())
}

/// A socket whose writes must make progress: a write, flush or shutdown the
/// peer leaves pending for `limit` fails, so hyper closes the connection and
/// drops the response it was writing. Reads are untouched; the header
/// deadline bounds those.
struct WriteStall<S> {
    inner: S,
    limit: Duration,
    stalled: Option<Pin<Box<tokio::time::Sleep>>>,
}

impl<S> WriteStall<S> {
    fn new(inner: S, limit: Duration) -> Self {
        Self {
            inner,
            limit,
            stalled: None,
        }
    }

    /// Track one poll of a write side operation: progress clears the clock,
    /// a pending poll starts it (or checks it).
    fn watch<T>(&mut self, cx: &mut Context<'_>, poll: Poll<io::Result<T>>) -> Poll<io::Result<T>> {
        if poll.is_ready() {
            self.stalled = None;
            return poll;
        }
        let limit = self.limit;
        let deadline = self
            .stalled
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
        if deadline.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the peer stopped reading the response",
            )));
        }
        Poll::Pending
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for WriteStall<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for WriteStall<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_write(cx, buf);
        this.watch(cx, poll)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_write_vectored(cx, bufs);
        this.watch(cx, poll)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_flush(cx);
        this.watch(cx, poll)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_shutdown(cx);
        this.watch(cx, poll)
    }
}

/// A failed accept leaves the listener serving: a peer's own reset is
/// skipped, and anything else (descriptors exhausted) is logged and retried
/// after a second, as `axum::serve` does.
async fn accept_failed(error: io::Error) {
    if matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    ) {
        return;
    }
    error!(%error, "accepting an HTTP connection failed; retrying in 1 s");
    tokio::time::sleep(Duration::from_secs(1)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    const TIMEOUT: Duration = Duration::from_millis(300);

    async fn start(max_connections: usize) -> (std::net::SocketAddr, watch::Sender<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let app = Router::new().route("/", get(|| async { "ok" }));
        let limits = ListenerLimits {
            header_read_timeout: TIMEOUT,
            write_stall_timeout: TIMEOUT,
            max_connections,
        };
        tokio::spawn(serve(listener, app, limits, async move {
            let _ = stop_rx.wait_for(|stop| *stop).await;
        }));
        (address, stop_tx)
    }

    async fn get_root(address: std::net::SocketAddr) -> String {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET / HTTP/1.1\r\nhost: test\r\nconnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    }

    /// A peer that connects and sends nothing, or stops partway through its
    /// headers, is closed at the header deadline instead of holding its
    /// socket and task for good.
    #[tokio::test]
    async fn a_silent_or_slow_peer_is_closed_at_the_header_deadline() {
        let (address, _stop) = start(8).await;
        for partial in [&b""[..], &b"GET / HTTP/1.1\r\nhost: te"[..]] {
            let mut peer = TcpStream::connect(address).await.unwrap();
            peer.write_all(partial).await.unwrap();
            let started = Instant::now();
            let mut buffer = Vec::new();
            let read = tokio::time::timeout(TIMEOUT * 10, peer.read_to_end(&mut buffer))
                .await
                .expect("the server closes a peer that never finishes its headers");
            read.unwrap();
            assert!(started.elapsed() >= TIMEOUT / 2, "{:?}", started.elapsed());
        }
        assert!(get_root(address).await.ends_with("ok"));
    }

    /// Past the cap a new peer is not served until a held connection closes,
    /// and a silent peer holds its slot no longer than the header deadline.
    #[tokio::test]
    async fn a_peer_past_the_connection_cap_waits_for_a_free_slot() {
        let (address, _stop) = start(1).await;
        let _silent = TcpStream::connect(address).await.unwrap();
        // Let the listener accept the silent peer before the second arrives.
        tokio::time::sleep(TIMEOUT / 6).await;
        let started = Instant::now();
        let response = tokio::time::timeout(TIMEOUT * 10, get_root(address))
            .await
            .expect("the slot frees at the silent peer's deadline");
        assert!(response.ends_with("ok"), "{response}");
        assert!(
            started.elapsed() >= TIMEOUT / 2,
            "served before the held slot freed: {:?}",
            started.elapsed()
        );
    }

    /// Shutdown stops accepting, lets a request in flight finish, and returns
    /// once every connection has closed.
    #[tokio::test]
    async fn shutdown_finishes_the_request_in_flight_and_returns() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (release_tx, release_rx) = watch::channel(false);
        let app = Router::new().route(
            "/",
            get(move || {
                let mut release = release_rx.clone();
                async move {
                    let _ = release.wait_for(|go| *go).await;
                    "done"
                }
            }),
        );
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let limits = ListenerLimits {
            header_read_timeout: TIMEOUT,
            write_stall_timeout: TIMEOUT,
            max_connections: 4,
        };
        let served = tokio::spawn(serve(listener, app, limits, async move {
            let _ = stop_rx.wait_for(|stop| *stop).await;
        }));
        let request = tokio::spawn(get_root(address));
        tokio::time::sleep(TIMEOUT / 3).await;
        stop_tx.send(true).unwrap();
        tokio::time::sleep(TIMEOUT / 3).await;
        assert!(!served.is_finished(), "a request in flight holds shutdown");
        assert!(
            TcpStream::connect(address).await.is_err(),
            "the listener stopped accepting"
        );
        release_tx.send(true).unwrap();
        let response = tokio::time::timeout(TIMEOUT * 10, request)
            .await
            .unwrap()
            .unwrap();
        assert!(response.ends_with("done"), "{response}");
        tokio::time::timeout(TIMEOUT * 10, served)
            .await
            .expect("serve returns once its connections close")
            .unwrap()
            .unwrap();
    }

    /// A client that sends a request and then reads nothing loses its
    /// connection once a write has waited on it for the stall limit, and the
    /// response body (with whatever it holds) is dropped, where it used to live
    /// as long as the socket.
    #[tokio::test]
    async fn a_client_that_stops_reading_loses_its_connection_and_response() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&dropped);
        let app = Router::new().route(
            "/",
            get(move || {
                let guard = Dropped(Arc::clone(&flag));
                async move {
                    let chunks = futures::stream::repeat_with(move || {
                        let _held = &guard;
                        Ok::<_, io::Error>(axum::body::Bytes::from(vec![b'x'; 64 * 1024]))
                    });
                    axum::body::Body::from_stream(chunks)
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let limits = ListenerLimits {
            header_read_timeout: Duration::from_secs(30),
            write_stall_timeout: TIMEOUT,
            max_connections: 4,
        };
        tokio::spawn(serve(listener, app, limits, std::future::pending()));
        let mut client = TcpStream::connect(address).await.unwrap();
        client
            .write_all(b"GET / HTTP/1.1\r\nhost: test\r\n\r\n")
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(20), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::time::sleep(TIMEOUT / 4).await;
            }
        })
        .await
        .expect("the stalled response is dropped");
        drop(client);
    }

    #[test]
    fn the_header_timeout_is_whole_seconds() {
        let resolve = |value: Option<&str>| {
            resolve_seconds("X", value.map(str::to_string), HEADER_READ_TIMEOUT)
        };
        assert_eq!(resolve(Some("75")).unwrap(), Duration::from_secs(75));
        assert_eq!(resolve(None).unwrap(), HEADER_READ_TIMEOUT);
        for refused in ["0", "3601", "1.5", ""] {
            assert!(resolve(Some(refused)).is_err(), "{refused}");
        }
    }

    #[test]
    fn the_connection_cap_is_positive() {
        assert_eq!(resolve_max_connections(Some(" 64 ".into())).unwrap(), 64);
        for refused in ["0", "-1", "many", ""] {
            assert!(
                resolve_max_connections(Some(refused.into())).is_err(),
                "{refused}"
            );
        }
        let default = resolve_max_connections(None).unwrap();
        assert!((16..=4096).contains(&default), "{default}");
    }
}
