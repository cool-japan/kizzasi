use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, Ident, Result};

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(Error::new_spanned(
                    name,
                    "Instrumented only supports structs with named fields",
                ))
            }
        },
        _ => {
            return Err(Error::new_spanned(
                name,
                "Instrumented only supports structs",
            ))
        }
    };

    // Find the #[metrics]-annotated field, or fall back to a field named `collector`.
    let mut metrics_field: Option<Ident> = None;
    let mut metrics_count = 0usize;

    for field in fields {
        let has_metrics_attr = field.attrs.iter().any(|a| a.path().is_ident("metrics"));
        if has_metrics_attr {
            metrics_count += 1;
            if metrics_count > 1 {
                return Err(Error::new_spanned(
                    field
                        .ident
                        .as_ref()
                        .expect("invariant: named fields always have idents"),
                    "only one field may be annotated with #[metrics]",
                ));
            }
            metrics_field = Some(
                field
                    .ident
                    .as_ref()
                    .expect("invariant: named fields always have idents")
                    .clone(),
            );
        }
    }

    // Fallback: look for a field named `collector`
    if metrics_field.is_none() {
        for field in fields {
            if field
                .ident
                .as_ref()
                .map(|i| i == "collector")
                .unwrap_or(false)
            {
                metrics_field = Some(
                    field
                        .ident
                        .as_ref()
                        .expect("invariant: named fields always have idents")
                        .clone(),
                );
                break;
            }
        }
    }

    let field_ident = metrics_field.ok_or_else(|| {
        Error::new_spanned(
            name,
            "Instrumented requires a field annotated with #[metrics] or a field named `collector`",
        )
    })?;

    Ok(quote! {
        impl #impl_generics kizzasi::telemetry::Instrumented for #name #ty_generics #where_clause {
            fn metrics(&self) -> ::std::sync::Arc<kizzasi::telemetry::MetricsCollector> {
                self.#field_ident.clone()
            }
        }
    })
}
