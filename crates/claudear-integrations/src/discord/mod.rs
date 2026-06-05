//! Advanced Discord integration with threading support.
//!
//! This module provides Discord thread management for PR-related conversations,
//! allowing follow-up messages to be posted in dedicated threads.

mod client;
mod thread_manager;
mod types;

pub use client::DiscordClient;
pub use thread_manager::ThreadManager;
pub use types::{
    CreateMessageParams, CreateThreadParams, DiscordChannel, DiscordMessage,
    DiscordMessageReference, DiscordThread, DiscordUser, MessageEmbed, ThreadState,
};
