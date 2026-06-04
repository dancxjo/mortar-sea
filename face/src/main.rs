#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "face=info,psyche=info,tower_http=warn,axum=warn".into()),
        )
        .init();

    face::run().await
}
