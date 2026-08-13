use proc_macro2::{Span, TokenStream};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, Ident, Result};

/// Returns `true` if `ty`'s last path segment is `Arc` (matches `Arc<T>`,
/// `std::sync::Arc<T>`, and `::std::sync::Arc<T>` alike). Used to guard the
/// name-based `collector` fallback below against selecting a same-named
/// field of an unrelated type (e.g. `collector: String`).
fn is_arc_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "Arc"),
        _ => false,
    }
}

/// Resolve the path to the `kizzasi` crate as seen from whichever crate is
/// being expanded, so the generated `impl` keeps working when `kizzasi` is
/// renamed in `Cargo.toml` (e.g. `agsp = { package = "kizzasi" }`) or when
/// this derive is used from within the `kizzasi` crate itself (`crate::...`
/// rather than a self-referential `kizzasi::...`).
///
/// Falls back to a leading-`::`-qualified `::kizzasi` when resolution fails
/// (e.g. `kizzasi` is not a direct dependency of the expanding crate). That
/// fallback is still strictly better than the previous bare `kizzasi::...`
/// path: a leading `::` can never be shadowed by a local item or module
/// named `kizzasi`.
fn kizzasi_crate_path() -> TokenStream {
    match crate_name("kizzasi") {
        Ok(FoundCrate::Itself) => quote! { crate },
        Ok(FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, Span::call_site());
            quote! { ::#ident }
        }
        Err(_) => quote! { ::kizzasi },
    }
}

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

    // Find the #[metrics]-annotated field, or fall back to a field named
    // `collector` whose type's last path segment is `Arc`.
    let mut metrics_field: Option<Ident> = None;
    let mut metrics_count = 0usize;

    for field in fields {
        let Some(fname) = field.ident.as_ref() else {
            // Unreachable: `Fields::Named` guarantees every field has an ident.
            continue;
        };
        let has_metrics_attr = field.attrs.iter().any(|a| a.path().is_ident("metrics"));
        if has_metrics_attr {
            metrics_count += 1;
            if metrics_count > 1 {
                return Err(Error::new_spanned(
                    fname,
                    "only one field may be annotated with #[metrics]",
                ));
            }
            metrics_field = Some(fname.clone());
        }
    }

    // Fallback: a field literally named `collector`, but only if its type
    // actually looks like `Arc<...>` — otherwise a same-named field of an
    // unrelated type (e.g. `collector: String`) would be silently selected
    // and fail with a confusing type mismatch inside generated code instead
    // of the clear "no field found" error below.
    if metrics_field.is_none() {
        for field in fields {
            let is_collector = field
                .ident
                .as_ref()
                .is_some_and(|ident| ident == "collector");
            if is_collector && is_arc_type(&field.ty) {
                metrics_field = field.ident.clone();
                break;
            }
        }
    }

    let field_ident = metrics_field.ok_or_else(|| {
        Error::new_spanned(
            name,
            "Instrumented requires a field annotated with #[metrics], or a field named \
             `collector` whose type is `Arc<...>`",
        )
    })?;

    let krate = kizzasi_crate_path();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #krate::telemetry::Instrumented for #name #ty_generics #where_clause {
            fn metrics(&self) -> ::std::sync::Arc<#krate::telemetry::MetricsCollector> {
                self.#field_ident.clone()
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::DeriveInput;

    fn expand_str(src: &str) -> Result<TokenStream> {
        let input: DeriveInput = syn::parse_str(src).expect("valid test struct");
        expand(input)
    }

    /// Strips all whitespace so substring checks on a stringified
    /// `TokenStream` don't depend on proc-macro2's token-spacing rules.
    fn normalize(tokens: &TokenStream) -> String {
        tokens
            .to_string()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect()
    }

    #[test]
    fn rejects_tuple_struct() {
        let err = expand_str("struct S(i32);").expect_err("tuple struct must be rejected");
        assert!(err.to_string().contains("named fields"));
    }

    #[test]
    fn rejects_enum() {
        let err = expand_str("enum S { A }").expect_err("enum must be rejected");
        assert!(err.to_string().contains("only supports structs"));
    }

    #[test]
    fn rejects_two_metrics_fields() {
        let err = expand_str(
            "struct S { #[metrics] a: std::sync::Arc<i32>, #[metrics] b: std::sync::Arc<i32> }",
        )
        .expect_err("two #[metrics] fields must be rejected");
        assert!(err.to_string().contains("only one field"));
    }

    #[test]
    fn rejects_no_metrics_field_found() {
        let err =
            expand_str("struct S { x: i32 }").expect_err("missing metrics field must be rejected");
        assert!(err.to_string().contains("requires a field"));
    }

    #[test]
    fn accepts_explicit_metrics_attribute_regardless_of_name() {
        assert!(expand_str("struct S { #[metrics] whatever: std::sync::Arc<i32> }").is_ok());
    }

    // ---- id184: name-based `collector` fallback is now type-checked ------

    #[test]
    fn collector_fallback_requires_arc_type() {
        // A same-named field of an unrelated type must not be silently
        // selected — it should fall through to the "no field found" error.
        let err = expand_str("struct S { collector: String }")
            .expect_err("non-Arc `collector` field must not be selected");
        assert!(err.to_string().contains("requires a field"));
    }

    #[test]
    fn collector_fallback_accepts_arc_type() {
        assert!(expand_str("struct S { collector: std::sync::Arc<i32> } ").is_ok());
    }

    #[test]
    fn collector_fallback_accepts_fully_qualified_arc_type() {
        assert!(expand_str("struct S { collector: ::std::sync::Arc<i32> } ").is_ok());
    }

    // ---- id182: crate path no longer hardcodes a bare `kizzasi::` --------

    #[test]
    fn falls_back_to_leading_colon_kizzasi_path() {
        // `kizzasi-macros`'s own Cargo.toml does not (and, being the defining
        // crate of these derives, cannot) depend on `kizzasi`, so
        // `crate_name("kizzasi")` resolves to `Err` here and the fallback
        // path is exercised for real.
        let tokens =
            expand_str("struct S { collector: std::sync::Arc<i32> } ").expect("valid input");
        let n = normalize(&tokens);
        assert!(
            n.contains("::kizzasi::telemetry::Instrumented"),
            "expected a leading-`::`-qualified path, got: {n}"
        );
        // The old bug: a bare, unqualified `kizzasi::` with no leading `::`,
        // shadowable by any local item/module named `kizzasi`.
        assert!(!n.contains("implkizzasi::telemetry::Instrumented"));
    }

    // ---- id185/186: generics still work after the expect()/unwrap_or cleanup

    #[test]
    fn accepts_lifetime_generic_struct() {
        assert!(expand_str(
            "struct S<'a> { collector: std::sync::Arc<i32>, _p: std::marker::PhantomData<&'a ()> }"
        )
        .is_ok());
    }

    #[test]
    fn accepts_type_param_generic_struct() {
        assert!(expand_str("struct S<T> { collector: std::sync::Arc<T> }").is_ok());
    }
}
