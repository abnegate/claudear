pub mod discord;

pub use discord::{
    format_discord_reference_links, format_discord_search_context, DiscordIndexer,
    DiscordMessageInput, DiscordSearchService,
    DISCORD_INDEX_VERSION,
};
