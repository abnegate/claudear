//! Core foundation types for the claudear application.
//!
//! This crate provides the shared types, error handling, HTTP abstractions,
//! secret management, and template rendering used across all claudear crates.

pub mod error;
pub mod http;
pub mod platform;
pub mod secret;
pub mod templates;
pub mod types;

pub use error::{Error, Result};
pub use secret::SecretValue;
