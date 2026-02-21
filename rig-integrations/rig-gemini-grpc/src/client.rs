use rig::prelude::*;
use std::fmt::Debug;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use super::GenerativeServiceClient;
use crate::completion::CompletionModel;
use crate::embedding::EmbeddingModel;

// ================================================================
// Google Gemini gRPC Client
// ================================================================
const GOOGLE_STUDIO_GRPC_ENDPOINT: &str = "https://generativelanguage.googleapis.com";
const GOOGLE_STUDIO_GRPC_TLS_DOMAIN: &str = "generativelanguage.googleapis.com";

/// User agent identifier for API tracking
const RIG_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.1.0";

/// Endpoint routing for Gemini gRPC requests.
///
/// For Vertex AI (`aiplatform.googleapis.com`) support, use [`crate::VertexClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiGrpcEndpoint {
    /// Google AI Studio Gemini endpoint.
    GoogleStudio,
    /// Custom endpoint override, for example an internal gateway/proxy URL.
    Custom { url: String },
}

impl Default for GeminiGrpcEndpoint {
    fn default() -> Self {
        Self::GoogleStudio
    }
}

#[derive(Clone)]
pub struct Client {
    api_key: String,
    channel: Channel,
}

impl Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("api_key", &"******")
            .field("channel", &"Channel")
            .finish()
    }
}

// Interceptor to add API key and client identification to metadata
#[derive(Clone)]
pub struct ApiKeyInterceptor {
    api_key: MetadataValue<tonic::metadata::Ascii>,
    client_id: MetadataValue<tonic::metadata::Ascii>,
}

impl Interceptor for ApiKeyInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert("x-goog-api-key", self.api_key.clone());
        request
            .metadata_mut()
            .insert("x-goog-api-client", self.client_id.clone());
        Ok(request)
    }
}

impl Client {
    /// Create a gRPC client for Google AI Studio with the given API key.
    pub async fn new(
        api_key: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_endpoint(api_key, GeminiGrpcEndpoint::GoogleStudio).await
    }

    /// Create a gRPC client with the given API key and endpoint selection.
    pub async fn new_with_endpoint(
        api_key: impl Into<String>,
        endpoint: GeminiGrpcEndpoint,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = api_key.into();
        let endpoint = match endpoint {
            GeminiGrpcEndpoint::GoogleStudio => Endpoint::from_static(GOOGLE_STUDIO_GRPC_ENDPOINT)
                .tls_config(
                    tonic::transport::ClientTlsConfig::new()
                        .with_webpki_roots()
                        .domain_name(GOOGLE_STUDIO_GRPC_TLS_DOMAIN),
                )?,
            GeminiGrpcEndpoint::Custom { url } => {
                let is_https = url.starts_with("https://");
                let endpoint = Endpoint::from_shared(url)?;

                if is_https {
                    endpoint
                        .tls_config(tonic::transport::ClientTlsConfig::new().with_webpki_roots())?
                } else {
                    endpoint
                }
            }
        };

        let channel = endpoint.connect().await?;

        Ok(Self { api_key, channel })
    }

    /// Create a client from `GEMINI_API_KEY` and a custom endpoint selection.
    /// Panics if `GEMINI_API_KEY` is not set or client creation fails.
    pub fn from_env_with_endpoint(endpoint: GeminiGrpcEndpoint) -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set");
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(Self::new_with_endpoint(api_key, endpoint))
                .expect("Failed to create Gemini gRPC client")
        })
    }

    /// Create a client from an API key value and endpoint selection.
    /// Panics if client creation fails.
    pub fn from_val_with_endpoint(
        api_key: impl Into<String>,
        endpoint: GeminiGrpcEndpoint,
    ) -> Self {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(Self::new_with_endpoint(api_key, endpoint))
                .expect("Failed to create Gemini gRPC client")
        })
    }

    /// Get a gRPC client with API key interceptor
    pub(crate) fn grpc_client(
        &self,
    ) -> Result<
        GenerativeServiceClient<
            tonic::service::interceptor::InterceptedService<Channel, ApiKeyInterceptor>,
        >,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let api_key = MetadataValue::try_from(&self.api_key)?;
        let client_id = MetadataValue::try_from(RIG_GRPC_CLIENT_IDENTIFIER)?;
        let interceptor = ApiKeyInterceptor { api_key, client_id };

        Ok(GenerativeServiceClient::with_interceptor(
            self.channel.clone(),
            interceptor,
        ))
    }
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Google Gemini gRPC client from the `GEMINI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        Self::from_env_with_endpoint(GeminiGrpcEndpoint::GoogleStudio)
    }

    fn from_val(input: Self::Input) -> Self {
        Self::from_val_with_endpoint(input, GeminiGrpcEndpoint::GoogleStudio)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    fn embedding_model(&self, model: impl Into<String>) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }
}
