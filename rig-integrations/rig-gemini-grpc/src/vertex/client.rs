use crate::vertex::completion::CompletionModel;
use crate::vertex::embedding::EmbeddingModel;
use google_cloud_auth::credentials;
use google_cloud_auth::credentials::{CacheableResource, Credentials, EntityTag};
use rig::client::{EmbeddingsClient, Nothing};
use rig::prelude::*;
use std::fmt::Debug;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::metadata::{MetadataKey, MetadataValue};
use tonic::transport::{Channel, Endpoint};

const DEFAULT_LOCATION: &str = "global";

const VERTEX_GLOBAL_GRPC_ENDPOINT: &str = "https://aiplatform.googleapis.com";
const VERTEX_GLOBAL_GRPC_TLS_DOMAIN: &str = "aiplatform.googleapis.com";

/// User agent identifier for API tracking.
const RIG_VERTEX_GRPC_CLIENT_IDENTIFIER: &str = "rig-grpc/0.1.0";

/// Endpoint routing for Vertex AI gRPC requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexGrpcEndpoint {
    /// Global Vertex endpoint: `https://aiplatform.googleapis.com`.
    Global,
    /// Regional Vertex endpoint: `https://{region}-aiplatform.googleapis.com`.
    Regional { region: String },
    /// Custom endpoint override, for example an internal gateway/proxy URL.
    Custom { url: String },
}

impl Default for VertexGrpcEndpoint {
    fn default() -> Self {
        Self::Global
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VertexClientError {
    #[error(
        "missing Google Cloud project; set it via VertexClientBuilder::with_project() or GOOGLE_CLOUD_PROJECT"
    )]
    MissingProject,

    #[error("gRPC transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("failed to build Google Cloud credentials: {0}")]
    Credentials(#[source] google_cloud_auth::build_errors::Error),

    #[error("failed to build impersonated credentials: {0}")]
    Impersonation(#[source] google_cloud_auth::build_errors::Error),

    #[error("failed to fetch auth headers: {0}")]
    AuthHeaders(#[source] google_cloud_auth::errors::CredentialsError),

    #[error("auth headers were reported as not modified but no cached headers are available")]
    MissingCachedAuthHeaders,

    #[error("auth header value is not valid UTF-8: {0}")]
    HeaderValueToStr(#[from] http::header::ToStrError),

    #[error("invalid gRPC metadata key: {0}")]
    InvalidMetadataKey(#[from] tonic::metadata::errors::InvalidMetadataKey),

    #[error("invalid gRPC metadata value: {0}")]
    InvalidMetadataValue(#[from] tonic::metadata::errors::InvalidMetadataValue),
}

/// Builder for [`VertexClient`].
#[derive(Clone, Debug, Default)]
pub struct VertexClientBuilder {
    project: Option<String>,
    location: Option<String>,
    endpoint: Option<VertexGrpcEndpoint>,
    credentials: Option<Credentials>,
}

impl VertexClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Google Cloud project ID explicitly.
    ///
    /// If not set, falls back to `GOOGLE_CLOUD_PROJECT`.
    pub fn with_project(mut self, project: &str) -> Self {
        self.project = Some(project.to_string());
        self
    }

    /// Set the Google Cloud location explicitly.
    ///
    /// If not set, falls back to `GOOGLE_CLOUD_LOCATION`, or defaults to `"global"`.
    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    /// Override the gRPC endpoint.
    pub fn with_endpoint(mut self, endpoint: VertexGrpcEndpoint) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Set credentials explicitly.
    ///
    /// If not set, uses Application Default Credentials (ADC), with optional
    /// service account impersonation via `GOOGLE_CLOUD_SERVICE_ACCOUNT`.
    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    pub async fn build(self) -> Result<VertexClient, VertexClientError> {
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or(VertexClientError::MissingProject)?;

        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let credentials = build_credentials(self.credentials)?;

        let endpoint = self.endpoint.unwrap_or_else(|| {
            if location == DEFAULT_LOCATION {
                VertexGrpcEndpoint::Global
            } else {
                VertexGrpcEndpoint::Regional {
                    region: location.clone(),
                }
            }
        });

        let channel = build_channel(&endpoint)
            .await
            .map_err(VertexClientError::Transport)?;

        Ok(VertexClient {
            project,
            location,
            endpoint,
            channel,
            credentials,
            auth_header_cache: Arc::new(Mutex::new(AuthHeaderCache::default())),
        })
    }
}

#[derive(Debug, Default)]
struct AuthHeaderCache {
    entity_tag: Option<EntityTag>,
    headers: Option<http::HeaderMap>,
}

#[derive(Clone)]
pub struct VertexClient {
    project: String,
    location: String,
    endpoint: VertexGrpcEndpoint,
    channel: Channel,
    credentials: Credentials,
    auth_header_cache: Arc<Mutex<AuthHeaderCache>>,
}

impl Debug for VertexClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexClient")
            .field("project", &self.project)
            .field("location", &self.location)
            .field("endpoint", &self.endpoint)
            .field("channel", &"Channel")
            .finish()
    }
}

impl VertexClient {
    pub fn builder() -> VertexClientBuilder {
        VertexClientBuilder::new()
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    pub(crate) fn resolve_model_path(&self, model: &str) -> String {
        if model.starts_with("projects/")
            || model.starts_with("publishers/")
            || model.starts_with("endpoints/")
        {
            model.to_string()
        } else {
            format!(
                "projects/{}/locations/{}/publishers/google/models/{model}",
                self.project, self.location
            )
        }
    }

    pub(crate) fn grpc_client(&self) -> crate::vertex::PredictionServiceClient<Channel> {
        crate::vertex::PredictionServiceClient::new(self.channel.clone())
    }

    pub(crate) async fn apply_auth<T>(
        &self,
        request: &mut tonic::Request<T>,
    ) -> Result<(), VertexClientError> {
        let headers = self.auth_headers().await?;

        for (name, value) in headers.iter() {
            let key = MetadataKey::from_bytes(name.as_str().as_bytes())?;
            let value = MetadataValue::try_from(value.to_str()?)?;
            request.metadata_mut().insert(key, value);
        }

        request.metadata_mut().insert(
            "x-goog-api-client",
            MetadataValue::try_from(RIG_VERTEX_GRPC_CLIENT_IDENTIFIER)?,
        );

        Ok(())
    }

    async fn auth_headers(&self) -> Result<http::HeaderMap, VertexClientError> {
        let (etag, cached_headers) = {
            let guard = self.auth_header_cache.lock().await;
            (guard.entity_tag.clone(), guard.headers.clone())
        };

        let mut extensions = http::Extensions::new();
        if let Some(etag) = etag {
            extensions.insert(etag);
        }

        let resource = self
            .credentials
            .headers(extensions)
            .await
            .map_err(VertexClientError::AuthHeaders)?;

        match resource {
            CacheableResource::NotModified => {
                cached_headers.ok_or(VertexClientError::MissingCachedAuthHeaders)
            }
            CacheableResource::New { entity_tag, data } => {
                let mut guard = self.auth_header_cache.lock().await;
                guard.entity_tag = Some(entity_tag);
                guard.headers = Some(data.clone());
                Ok(data)
            }
        }
    }
}

impl ProviderClient for VertexClient {
    type Input = Nothing;

    /// Create a Vertex AI client from environment variables.
    ///
    /// Panics if the environment is improperly configured.
    fn from_env() -> Self {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(VertexClientBuilder::new().build())
                .expect(
                    "Failed to build Vertex gRPC client. Ensure GOOGLE_CLOUD_PROJECT is set and ADC is configured (e.g. `gcloud auth application-default login`).",
                )
        })
    }

    fn from_val(_: Self::Input) -> Self {
        panic!(
            "Vertex AI uses Application Default Credentials (ADC). Use `VertexClient::from_env()` or `VertexClient::builder()`."
        );
    }
}

impl CompletionClient for VertexClient {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for VertexClient {
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

fn build_credentials(
    explicit_creds: Option<Credentials>,
) -> Result<Credentials, VertexClientError> {
    if let Some(creds) = explicit_creds {
        return Ok(creds);
    }

    let source_credentials = credentials::Builder::default()
        .build()
        .map_err(VertexClientError::Credentials)?;

    if let Ok(service_account) = std::env::var("GOOGLE_CLOUD_SERVICE_ACCOUNT") {
        credentials::impersonated::Builder::from_source_credentials(source_credentials)
            .with_target_principal(service_account)
            .build()
            .map_err(VertexClientError::Impersonation)
    } else {
        Ok(source_credentials)
    }
}

async fn build_channel(endpoint: &VertexGrpcEndpoint) -> Result<Channel, tonic::transport::Error> {
    let endpoint = match endpoint {
        VertexGrpcEndpoint::Global => Endpoint::from_static(VERTEX_GLOBAL_GRPC_ENDPOINT)
            .tls_config(
                tonic::transport::ClientTlsConfig::new()
                    .with_webpki_roots()
                    .domain_name(VERTEX_GLOBAL_GRPC_TLS_DOMAIN),
            )?,
        VertexGrpcEndpoint::Regional { region } => {
            let url = format!("https://{region}-aiplatform.googleapis.com");
            Endpoint::from_shared(url)?
                .tls_config(tonic::transport::ClientTlsConfig::new().with_webpki_roots())?
        }
        VertexGrpcEndpoint::Custom { url } => {
            let is_https = url.starts_with("https://");
            let endpoint = Endpoint::from_shared(url.clone())?;

            if is_https {
                endpoint.tls_config(tonic::transport::ClientTlsConfig::new().with_webpki_roots())?
            } else {
                endpoint
            }
        }
    };

    endpoint.connect().await
}
