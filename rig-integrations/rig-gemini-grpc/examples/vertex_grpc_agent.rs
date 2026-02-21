use rig::completion::Prompt;
use rig::prelude::*;
use rig_gemini_grpc::VertexClient;

#[tracing::instrument(ret)]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize the Vertex AI gRPC client (ADC + GOOGLE_CLOUD_PROJECT).
    let client = VertexClient::from_env();

    let agent = client
        .agent("gemini-2.5-flash")
        .preamble("Be creative and concise. Answer directly and clearly.")
        .temperature(0.5)
        .build();

    tracing::info!("Prompting the agent via Vertex AI gRPC...");

    let response = agent.prompt("Hello from Vertex AI!").await;
    tracing::info!("Response: {:?}", response);

    match response {
        Ok(response) => println!("{response}"),
        Err(e) => {
            tracing::error!("Error: {:?}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
