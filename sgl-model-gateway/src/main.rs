use anyhow::{Context, Result};
use clap::Parser;
use smg::{config::cli::Cli, server, version};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // support --verbose-version flag
    if cli.wants_verbose_version() {
        println!("{}", version::get_verbose_version_string());
        return Ok(());
    }

    let config = cli
        .try_into_config()
        .context("resolve gateway configuration from cli arguments")?;

    server::startup(config).await.context("run server")
}
