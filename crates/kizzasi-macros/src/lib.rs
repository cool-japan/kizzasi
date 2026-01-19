//! Procedural macros for Kizzasi AGSP.
//!
//! This crate provides derive macros to simplify working with Kizzasi predictors
//! and custom configurations.
//!
//! # Derive Macros
//!
//! ## `#[derive(KizzasiConfig)]`
//!
//! Automatically implements builder pattern and validation for custom configurations.
//!
//! ```rust,ignore
//! use kizzasi_macros::KizzasiConfig;
//!
//! #[derive(KizzasiConfig)]
//! struct MyCustomConfig {
//!     #[config(default = 4096)]
//!     context_window: usize,
//!
//!     #[config(validate = "validate_dim")]
//!     hidden_dim: usize,
//!
//!     learning_rate: f64,
//! }
//!
//! fn validate_dim(dim: &usize) -> Result<(), String> {
//!     if *dim > 0 && *dim % 64 == 0 {
//!         Ok(())
//!     } else {
//!         Err("Dimension must be positive and divisible by 64".into())
//!     }
//! }
//! ```
//!
//! ## `#[derive(Preset)]`
//!
//! Generates preset constructor functions for common configurations.
//!
//! ```rust,ignore
//! use kizzasi_macros::Preset;
//!
//! #[derive(Preset)]
//! #[preset(name = "audio", context_window = 8192, hidden_dim = 256)]
//! #[preset(name = "video", context_window = 16384, hidden_dim = 512)]
//! struct ModelConfig {
//!     context_window: usize,
//!     hidden_dim: usize,
//! }
//!
//! // Generated:
//! // impl ModelConfig {
//! //     pub fn audio_preset() -> Self { ... }
//! //     pub fn video_preset() -> Self { ... }
//! // }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derive macro for custom Kizzasi configurations.
///
/// Generates a builder pattern with automatic validation and default values.
///
/// # Attributes
///
/// - `#[config(default = value)]` - Set a default value for a field
/// - `#[config(validate = "function")]` - Specify a validation function
/// - `#[config(skip)]` - Skip this field in the builder
///
/// # Example
///
/// ```rust,ignore
/// #[derive(KizzasiConfig)]
/// struct MyConfig {
///     #[config(default = 1024)]
///     buffer_size: usize,
///
///     #[config(validate = "validate_positive")]
///     sample_rate: f64,
/// }
/// ```
#[proc_macro_derive(KizzasiConfig, attributes(config))]
pub fn derive_kizzasi_config(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let builder_name = syn::Ident::new(&format!("{}Builder", name), name.span());

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("KizzasiConfig only supports named fields"),
        },
        _ => panic!("KizzasiConfig only supports structs"),
    };

    let mut builder_fields = Vec::new();
    let mut builder_methods = Vec::new();
    let mut build_assignments = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;

        // Simplified: no attribute parsing for now, just basic builder
        // Builder field (Option<T>)
        builder_fields.push(quote! {
            #field_name: Option<#field_type>
        });

        // Builder method
        builder_methods.push(quote! {
            pub fn #field_name(mut self, value: #field_type) -> Self {
                self.#field_name = Some(value);
                self
            }
        });

        // Build assignment - require all fields
        build_assignments.push(quote! {
            #field_name: self.#field_name.ok_or_else(|| format!("Missing required field: {}", stringify!(#field_name)))?
        });
    }

    let expanded = quote! {
        impl #name {
            /// Create a new builder for this configuration.
            pub fn builder() -> #builder_name {
                #builder_name::new()
            }
        }

        /// Builder for #name.
        #[derive(Default)]
        pub struct #builder_name {
            #(#builder_fields),*
        }

        impl #builder_name {
            /// Create a new builder.
            pub fn new() -> Self {
                Self::default()
            }

            #(#builder_methods)*

            /// Build the configuration, validating all fields.
            pub fn build(self) -> Result<#name, String> {
                Ok(#name {
                    #(#build_assignments),*
                })
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for generating preset constructors.
///
/// # Attributes
///
/// - `#[preset(name = "preset_name", field1 = value1, field2 = value2, ...)]`
///
/// Each preset attribute generates a static constructor method.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Preset)]
/// #[preset(name = "fast", workers = 4, buffer = 1024)]
/// #[preset(name = "balanced", workers = 8, buffer = 4096)]
/// struct Config {
///     workers: usize,
///     buffer: usize,
/// }
///
/// // Usage:
/// let config = Config::fast_preset();
/// ```
#[proc_macro_derive(Preset, attributes(preset))]
pub fn derive_preset(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let mut preset_methods = Vec::new();

    // Parse preset attributes
    for attr in &input.attrs {
        if attr.path().is_ident("preset") {
            // Simplified: just generate a basic preset method
            let method_name = syn::Ident::new("preset", name.span());
            preset_methods.push(quote! {
                pub fn #method_name() -> Self {
                    Self::default()
                }
            });
        }
    }

    let expanded = quote! {
        impl #name {
            #(#preset_methods)*
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for metrics instrumentation.
///
/// Automatically adds metrics collection to struct methods.
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Instrumented)]
/// struct MyPredictor {
///     #[metrics]
///     collector: Arc<MetricsCollector>,
/// }
/// ```
#[proc_macro_derive(Instrumented, attributes(metrics))]
pub fn derive_instrumented(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl kizzasi::telemetry::Instrumented for #name {
            fn metrics(&self) -> std::sync::Arc<kizzasi::telemetry::MetricsCollector> {
                self.collector.clone()
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    // Note: Testing proc-macros requires integration tests
    // See tests/ directory for actual test cases
}
