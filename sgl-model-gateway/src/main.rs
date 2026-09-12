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

    let log_guard = server::init_tracing(&config.observability)?;

    let server_result = server::startup(config).await;
    let shutdown_result = server::shutdown_tracing(log_guard).await;

    server_result.context("run gateway")?;
    shutdown_result.context("shutdown tracing")?;

    Ok(())
}
