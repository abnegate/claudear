//! Vectorlite integration for HNSW-based similarity search.
//!
//! Uses the vectorlite SQLite extension for efficient approximate nearest neighbor search.
//! The actual HNSW table creation and querying is done in `sqlite.rs` via
//! `ensure_*_vector_table()` / `find_similar_*_vector()` methods.

use claudear_core::error::{Error, Result};
use rusqlite::Connection;
use std::path::Path;

/// Load the vectorlite extension into a SQLite connection.
///
/// # Safety
/// This enables extension loading which can execute arbitrary code.
/// Only load trusted extensions.
pub fn load_vectorlite_extension(conn: &Connection, extension_path: &Path) -> Result<()> {
    // Enable extension loading
    unsafe {
        conn.load_extension_enable()?;
    }

    // Load vectorlite
    // SAFETY: We only load the vectorlite extension from trusted paths
    unsafe {
        conn.load_extension(extension_path, None::<&str>)
            .map_err(|e| Error::Other(format!("Failed to load vectorlite extension: {}", e)))?;
    }

    // Disable extension loading for safety
    conn.load_extension_disable()?;

    Ok(())
}

/// Try to load vectorlite from common paths.
pub fn try_load_vectorlite(conn: &Connection) -> Result<bool> {
    let common_paths = [
        // Docker/Linux paths
        "/usr/local/lib/vectorlite.so",
        "/usr/lib/vectorlite.so",
        "/app/lib/vectorlite.so",
        // macOS paths
        "/usr/local/lib/vectorlite.dylib",
        "/opt/homebrew/lib/vectorlite.dylib",
        // Local development
        "./vectorlite.so",
        "./vectorlite.dylib",
        // Windows paths
        "./vectorlite.dll",
    ];

    for path in common_paths {
        let path = Path::new(path);
        if path.exists() {
            match load_vectorlite_extension(conn, path) {
                Ok(()) => {
                    tracing::info!("Loaded vectorlite from {:?}", path);
                    return Ok(true);
                }
                Err(e) => {
                    tracing::warn!("Failed to load vectorlite from {:?}: {}", path, e);
                }
            }
        }
    }

    // Also try from CLAUDEAR_VECTORLITE_PATH env var
    if let Ok(path) = std::env::var("CLAUDEAR_VECTORLITE_PATH") {
        let path = Path::new(&path);
        if path.exists() {
            load_vectorlite_extension(conn, path)?;
            tracing::info!(
                "Loaded vectorlite from CLAUDEAR_VECTORLITE_PATH: {:?}",
                path
            );
            return Ok(true);
        }
    }

    tracing::warn!("Vectorlite extension not found, vector search will be disabled");
    Ok(false)
}

/// Check if vectorlite is available in this connection.
pub fn is_vectorlite_available(conn: &Connection) -> bool {
    conn.execute_batch("SELECT vectorlite_info()").is_ok()
}
