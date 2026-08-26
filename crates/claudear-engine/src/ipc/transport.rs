//! Platform-abstracted IPC transport.
//!
//! On Unix, uses Unix domain sockets.  On Windows, uses TCP on localhost with
//! a port file for daemon discovery.
//!
//! Consumers import [`IpcListener`], [`IpcStream`], [`connect`], [`bind`], and
//! [`check_connection`] without any `#[cfg]` in their own code.

use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// Type aliases — one concrete type per platform, transparent to callers
// ---------------------------------------------------------------------------

#[cfg(not(windows))]
pub type IpcListener = tokio::net::UnixListener;
#[cfg(windows)]
pub type IpcListener = tokio::net::TcpListener;

#[cfg(not(windows))]
pub type IpcStream = tokio::net::UnixStream;
#[cfg(windows)]
pub type IpcStream = tokio::net::TcpStream;

// ---------------------------------------------------------------------------
// Connection helpers
// ---------------------------------------------------------------------------

/// Connect to an IPC endpoint at `path`.
///
/// * **Unix** — connects to the Unix domain socket at `path`.
/// * **Windows** — reads a TCP port number from `path` and connects to
///   `127.0.0.1:<port>`.
pub async fn connect(path: &Path) -> io::Result<IpcStream> {
    #[cfg(not(windows))]
    {
        IpcStream::connect(path).await
    }
    #[cfg(windows)]
    {
        let port = read_port(path)?;
        IpcStream::connect(("127.0.0.1", port)).await
    }
}

/// Bind an IPC listener at `path`.
///
/// * **Unix** — binds a Unix domain socket at `path`.
/// * **Windows** — binds TCP on `127.0.0.1:0` (ephemeral port) and writes the
///   assigned port to `path`.
pub fn bind(path: &Path) -> io::Result<IpcListener> {
    #[cfg(not(windows))]
    {
        IpcListener::bind(path)
    }
    #[cfg(windows)]
    {
        // Use std to bind synchronously so we get the port immediately.
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = std_listener.local_addr()?.port();
        write_port(path, port)?;
        std_listener.set_nonblocking(true)?;
        IpcListener::from_std(std_listener)
    }
}

/// Synchronously check whether a daemon is reachable at `path`.
///
/// Returns `true` if a connection can be established.
pub fn check_connection(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    #[cfg(not(windows))]
    {
        std::os::unix::net::UnixStream::connect(path).is_ok()
    }
    #[cfg(windows)]
    {
        read_port(path)
            .map(|port| std::net::TcpStream::connect(("127.0.0.1", port)).is_ok())
            .unwrap_or(false)
    }
}

/// Check whether the IPC endpoint at `path` is stale (file exists but nobody
/// is listening).
pub fn is_stale(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    #[cfg(not(windows))]
    {
        std::os::unix::net::UnixStream::connect(path).is_err()
    }
    #[cfg(windows)]
    {
        read_port(path)
            .map(|port| std::net::TcpStream::connect(("127.0.0.1", port)).is_err())
            .unwrap_or(true)
    }
}

// ---------------------------------------------------------------------------
// Port file helpers (Windows only)
// ---------------------------------------------------------------------------

#[cfg(windows)]
pub(crate) fn read_port(path: &Path) -> io::Result<u16> {
    let contents = std::fs::read_to_string(path)?;
    contents
        .trim()
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(windows)]
fn write_port(path: &Path, port: u16) -> io::Result<()> {
    std::fs::write(path, port.to_string())
}
