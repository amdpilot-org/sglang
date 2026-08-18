use clap::Parser;
use smg::{
    config::cli::Cli,
    observability::otel_trace::{is_otel_enabled, shutdown_otel},
    server, version,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.wants_verbose_version() {
        println!("{}", version::get_verbose_version_string());
        return Ok(());
    }

    let cfg = cli.complete_config()?;

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move { server::startup(cfg).await })?;
    if is_otel_enabled() {
        shutdown_otel();
    }
    Ok(())
}
