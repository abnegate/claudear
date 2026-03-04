//! TLS auto-provisioning via Let's Encrypt ACME (TLS-ALPN-01).
//!
//! When `tls.enabled = true`, the server provisions and auto-renews
//! certificates using `rustls-acme`. Otherwise, plain HTTP is served.

use crate::port_forward;
use axum::Router;
use claudear_config::config::TlsConfig;
use claudear_core::error::Result;
use std::net::{Ipv4Addr, SocketAddr};

/// Serve an Axum app over HTTPS with auto-provisioned Let's Encrypt certificates.
///
/// - Provisions certs via TLS-ALPN-01 on `tls_config.https_port` (default 443).
/// - Persists certs to `tls_config.cache_dir` via `DirCache`.
/// - Optionally spawns an HTTP→HTTPS redirect on `tls_config.http_redirect_port`.
pub async fn serve_with_tls(tls_config: &TlsConfig, bind_address: &str, app: Router) -> Result<()> {
    use rustls_acme::caches::DirCache;
    use rustls_acme::AcmeConfig;
    use tokio_stream::StreamExt;

    // Ensure cache directory exists
    if let Err(e) = std::fs::create_dir_all(&tls_config.cache_dir) {
        tracing::warn!(
            error = %e,
            path = %tls_config.cache_dir.display(),
            "Failed to create ACME cache directory"
        );
    }

    // --- macOS port forwarding for privileged ports ---
    let mut redirects = Vec::new();

    let actual_https_port = if port_forward::needs_port_forward(tls_config.https_port) {
        let high = port_forward::forwarded_port(tls_config.https_port);
        redirects.push((tls_config.https_port, high));
        high
    } else {
        tls_config.https_port
    };

    let actual_redirect_port = if tls_config.http_redirect_port > 0
        && port_forward::needs_port_forward(tls_config.http_redirect_port)
    {
        let high = port_forward::forwarded_port(tls_config.http_redirect_port);
        redirects.push((tls_config.http_redirect_port, high));
        high
    } else {
        tls_config.http_redirect_port
    };

    // Set up pf rules — the guard cleans them up when the server stops.
    let _pf_guard = if !redirects.is_empty() {
        Some(
            port_forward::setup_port_forward(&redirects, bind_address)
                .map_err(std::io::Error::other)?,
        )
    } else {
        None
    };
    // --- end port forwarding ---

    let domains = tls_config.domains.clone();
    let cache_dir = tls_config.cache_dir.clone();

    let mut acme_config = AcmeConfig::new(domains)
        .cache(DirCache::new(cache_dir))
        .directory_lets_encrypt(tls_config.production);

    if let Some(ref email) = tls_config.email {
        acme_config = acme_config.contact_push(format!("mailto:{email}"));
    }

    let mut state = acme_config.state();
    let acceptor = state.axum_acceptor(state.default_rustls_config());

    // Spawn ACME event logger
    tokio::spawn(async move {
        loop {
            match state.next().await {
                Some(Ok(ok)) => tracing::info!("ACME event: {:?}", ok),
                Some(Err(err)) => tracing::error!("ACME error: {:?}", err),
                None => break,
            }
        }
    });

    // Optionally spawn HTTP→HTTPS redirect
    if actual_redirect_port > 0 {
        let https_port = tls_config.https_port;
        let redirect_addr: SocketAddr = format!("{bind_address}:{actual_redirect_port}")
            .parse()
            .unwrap_or_else(|_| SocketAddr::from((Ipv4Addr::UNSPECIFIED, actual_redirect_port)));

        tokio::spawn(async move {
            let redirect_app =
                Router::new().fallback(move |req: axum::extract::Request| async move {
                    let host = req
                        .headers()
                        .get(axum::http::header::HOST)
                        .and_then(|h| h.to_str().ok())
                        .unwrap_or("localhost");
                    // Strip port from Host header if present
                    let host_without_port = host.split(':').next().unwrap_or(host);
                    let path = req
                        .uri()
                        .path_and_query()
                        .map(|pq| pq.as_str())
                        .unwrap_or("/");
                    let location = if https_port == 443 {
                        format!("https://{host_without_port}{path}")
                    } else {
                        format!("https://{host_without_port}:{https_port}{path}")
                    };
                    axum::response::Redirect::permanent(&location)
                });

            let listener = match tokio::net::TcpListener::bind(redirect_addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        addr = %redirect_addr,
                        "Failed to bind HTTP redirect listener"
                    );
                    return;
                }
            };
            tracing::info!("HTTP→HTTPS redirect listening on {}", redirect_addr);
            if let Err(e) = axum::serve(listener, redirect_app).await {
                tracing::error!(error = %e, "HTTP redirect server error");
            }
        });
    }

    // Serve the main app over HTTPS
    let addr: SocketAddr = format!("{bind_address}:{actual_https_port}")
        .parse()
        .unwrap_or_else(|_| SocketAddr::from((Ipv4Addr::UNSPECIFIED, actual_https_port)));

    tracing::info!("HTTPS server listening on {}", addr);
    if actual_https_port != tls_config.https_port {
        tracing::info!(
            "Traffic on :{} is forwarded to :{} via port forwarding",
            tls_config.https_port,
            actual_https_port,
        );
    }
    tracing::info!(
        domains = ?tls_config.domains,
        production = tls_config.production,
        cache_dir = %tls_config.cache_dir.display(),
        "ACME certificate auto-provisioning active"
    );

    axum_server::bind(addr)
        .acceptor(acceptor)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

/// Serve an Axum app over plain HTTP.
pub async fn serve_plain_http(bind_address: &str, port: u16, app: Router) -> Result<()> {
    // Set up port forwarding if needed (macOS + privileged port + non-root).
    let (actual_port, _pf_guard) = if port_forward::needs_port_forward(port) {
        let high = port_forward::forwarded_port(port);
        let guard = port_forward::setup_port_forward(&[(port, high)], bind_address)
            .map_err(std::io::Error::other)?;
        (high, Some(guard))
    } else {
        (port, None)
    };

    let addr = format!("{bind_address}:{actual_port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied && actual_port < 1024 {
            std::io::Error::new(
                e.kind(),
                format!(
                    "Cannot bind to port {} (privileged ports < 1024 require root). \
                     Use a port >= 1024 or run with elevated privileges.",
                    actual_port
                ),
            )
        } else {
            e
        }
    })?;

    if actual_port != port {
        tracing::info!(
            "Listening on :{actual_port}, traffic on :{port} forwarded via port forwarding"
        );
    }

    axum::serve(listener, app).await?;

    Ok(())
}
