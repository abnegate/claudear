//! External service integrations for claudear.
//!
//! This crate provides issue sources, notifiers, agent runners, SCM providers,
//! webhook handlers, Discord/Slack clients, GitHub App auth, and telemetry.

pub mod ask_reply_inbox;
pub mod chat;
pub mod discord;
pub mod github;
pub mod github_app;
pub mod gitlab;
pub mod notifier;
pub mod port_forward;
pub mod reports;
pub mod runner;
pub mod scm;
pub mod source;
pub mod telemetry;
pub mod tls;
pub mod webhook;
