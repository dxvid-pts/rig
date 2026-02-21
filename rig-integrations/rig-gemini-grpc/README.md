# Rig-Gemini-gRPC

This companion crate integrates Google Gemini gRPC API with Rig, offering better performance and type safety compared to the REST API.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-gemini-grpc = "0.1.0"
rig-core = "0.30.0"
```

You can also run `cargo add rig-gemini-grpc rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for more usage examples.

## Setup

Set your Gemini API key as an environment variable:

```shell
export GEMINI_API_KEY=your_api_key_here
```

### Vertex AI setup

To use the Vertex AI gRPC client (`VertexClient`), set your Google Cloud project:

```shell
export GOOGLE_CLOUD_PROJECT=your_gcp_project_id
```

Then authenticate using either:

- A Vertex AI API key:
  ```shell
  export VERTEX_API_KEY=your_api_key_here
  ```
- Application Default Credentials (ADC):
  ```shell
  gcloud auth application-default login
  ```

Optionally set a location (defaults to `global`):

```shell
export GOOGLE_CLOUD_LOCATION=us-central1
```

## Example

```rust
use rig::prelude::*;
use rig_gemini_grpc::Client;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::from_env();

    let agent = client
        .agent("gemini-2.5-flash")
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{}", response);

    Ok(())
}
```

## Features

- Full completion support with streaming
- Embedding generation
- Tool calling support
- Reasoning and thought signatures
- Image input support
