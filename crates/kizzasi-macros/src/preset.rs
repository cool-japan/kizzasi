use crate::attrs::parse_preset_attr;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{Data, DeriveInput, Error, Fields, Result};

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let struct_fields: Vec<syn::Ident> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => f
                .named
                .iter()
                .filter_map(|field| field.ident.clone())
                .collect(),
            _ => {
                return Err(Error::new_spanned(
                    name,
                    "Preset only supports structs with named fields",
                ))
            }
        },
        _ => return Err(Error::new_spanned(name, "Preset only supports structs")),
    };
    let struct_field_set: HashSet<String> = struct_fields.iter().map(|i| i.to_string()).collect();
    let total_field_count = struct_fields.len();

    let mut preset_methods = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut found_preset_attr = false;

    for attr in &input.attrs {
        if !attr.path().is_ident("preset") {
            continue;
        }
        found_preset_attr = true;
        let spec = parse_preset_attr(attr)?;
        let preset_name = spec.name.value();

        // Check for duplicate preset names
        if !seen_names.insert(preset_name.clone()) {
            return Err(Error::new_spanned(
                &spec.name,
                format!("duplicate preset name `{preset_name}`"),
            ));
        }

        let method_ident = format_ident!("{}_preset", preset_name);

        // Validate all preset fields exist in the struct
        for (field_ident, _) in &spec.fields {
            if !struct_field_set.contains(&field_ident.to_string()) {
                return Err(Error::new_spanned(
                    field_ident,
                    format!("#[preset(...)] field `{field_ident}` is not a field of `{name}`"),
                ));
            }
        }

        let set_fields: Vec<TokenStream> = spec
            .fields
            .iter()
            .map(|(ident, expr)| quote! { #ident: #expr })
            .collect();

        // `parse_preset_attr` rejects duplicate field keys, so `spec.fields.len()`
        // is the number of *distinct* fields set — safe to compare directly
        // against the struct's total field count to decide whether every field
        // is covered.
        let full_coverage = spec.fields.len() == total_field_count;
        let body = if full_coverage {
            // All fields covered — no `..Default::default()` needed
            quote! {
                Self { #(#set_fields,)* }
            }
        } else {
            quote! {
                Self { #(#set_fields,)* ..::core::default::Default::default() }
            }
        };

        // Only partial-coverage presets need the struct-update fallback, so
        // only they require `Self: Default` — attached per-method rather than
        // on the whole impl, so a generic struct whose type parameters are
        // not `Default` can still use full-coverage presets.
        let method = if full_coverage {
            quote! {
                pub fn #method_ident() -> Self {
                    #body
                }
            }
        } else {
            quote! {
                pub fn #method_ident() -> Self
                where
                    Self: ::core::default::Default,
                {
                    #body
                }
            }
        };
        preset_methods.push(method);
    }

    if !found_preset_attr {
        return Err(Error::new_spanned(
            &input.ident,
            "#[derive(Preset)] requires at least one #[preset(name = \"...\", ...)] attribute",
        ));
    }

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #name #ty_generics #where_clause {
            #(#preset_methods)*
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
        let err = expand_str(r#"#[preset(name = "a")] struct S(i32);"#)
            .expect_err("tuple struct must be rejected");
        assert!(err.to_string().contains("named fields"));
    }

    #[test]
    fn rejects_enum() {
        let err =
            expand_str(r#"#[preset(name = "a")] enum S { A }"#).expect_err("enum must be rejected");
        assert!(err.to_string().contains("only supports structs"));
    }

    #[test]
    fn rejects_unknown_field() {
        let err = expand_str(r#"#[preset(name = "a", nope = 1)] struct S { x: i32 }"#)
            .expect_err("unknown field must be rejected");
        assert!(err.to_string().contains("is not a field of"));
    }

    #[test]
    fn rejects_duplicate_preset_name_across_attrs() {
        let err = expand_str(
            r#"
            #[preset(name = "a", x = 1)]
            #[preset(name = "a", x = 2)]
            struct S { x: i32 }
            "#,
        )
        .expect_err("duplicate preset name must be rejected");
        assert!(err.to_string().contains("duplicate preset name"));
    }

    // ---- id183: no #[preset(...)] attribute at all -----------------------

    #[test]
    fn rejects_derive_with_no_preset_attribute() {
        let err =
            expand_str("struct S { x: i32 }").expect_err("missing attribute must be rejected");
        assert!(err.to_string().contains("requires at least one"));
    }

    #[test]
    fn rejects_misspelled_attribute_name() {
        // `#[presets(...)]` (plural) is not registered under `attributes(preset)`,
        // so it is simply never observed by this loop — same failure as writing
        // no attribute at all, now with a real diagnostic instead of silence.
        let err = expand_str(r#"#[presets(name = "a")] struct S { x: i32 }"#)
            .expect_err("misspelled attribute must be rejected");
        assert!(err.to_string().contains("requires at least one"));
    }

    // ---- id180: full-coverage check is sound after dedup -----------------

    #[test]
    fn full_coverage_with_would_be_duplicate_is_rejected_before_codegen() {
        // Previously: `#[preset(name = "a", x = 1, x = 2)]` on a 2-field struct
        // reached codegen with `fields.len() == 2 == total_field_count` and
        // emitted `Self { x: 1, x: 2, }`, producing two unrelated rustc errors.
        // Now the duplicate is caught in `parse_preset_attr` first.
        let err = expand_str(r#"#[preset(name = "a", x = 1, x = 2)] struct S { x: i32, y: i32 }"#)
            .expect_err("duplicate field must be rejected before codegen");
        assert!(err.to_string().contains("duplicate field"));
    }

    // ---- id185: generics are supported -----------------------------------

    #[test]
    fn accepts_lifetime_generic_struct() {
        assert!(expand_str(
            r#"#[preset(name = "a", name_field = "x")] struct S<'a> { name_field: &'a str }"#
        )
        .is_ok());
    }

    #[test]
    fn full_coverage_preset_on_generic_struct_needs_no_default_bound() {
        let tokens = expand_str(r#"#[preset(name = "a", value = 1)] struct S<T> { value: T }"#)
            .expect("valid input");
        // Full coverage sets every field explicitly and never falls back to
        // `..Default::default()`, so — unlike the partial-coverage case below
        // — nothing in the generated method should mention `Default` at all
        // (neither the struct-update fallback nor the `where Self: Default`
        // bound gating it).
        let n = normalize(&tokens);
        assert!(
            !n.contains("Default"),
            "expected no `Default` reference, got: {n}"
        );
    }

    #[test]
    fn partial_coverage_preset_on_generic_struct_requires_default_bound() {
        let tokens =
            expand_str(r#"#[preset(name = "a", value = 1)] struct S<T> { value: i32, extra: T }"#)
                .expect("valid input");
        // Partial coverage needs `..Default::default()` for the uncovered
        // field, which requires `Self: Default` — present as a per-method
        // bound so full-coverage presets on the same struct are unaffected.
        let n = normalize(&tokens);
        assert!(
            n.contains("Default"),
            "expected a `Default` reference, got: {n}"
        );
        assert!(
            n.contains("whereSelf"),
            "expected a per-method `where` bound, got: {n}"
        );
    }
}
