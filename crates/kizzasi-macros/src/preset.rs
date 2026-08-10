use crate::attrs::parse_preset_attr;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashSet;
use syn::{Data, DeriveInput, Error, Fields, Result};

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;

    // Guard: no generics
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &input.generics,
            "Preset does not support generic structs",
        ));
    }

    let struct_fields: Vec<syn::Ident> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => f
                .named
                .iter()
                .map(|field| {
                    field
                        .ident
                        .as_ref()
                        .expect("invariant: named fields always have idents")
                        .clone()
                })
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

    for attr in &input.attrs {
        if !attr.path().is_ident("preset") {
            continue;
        }
        let spec = parse_preset_attr(attr)?;
        let preset_name = spec.name.value();

        // Check for duplicate preset names
        if !seen_names.insert(preset_name.clone()) {
            return Err(Error::new_spanned(
                &spec.name,
                format!("duplicate preset name `{}`", preset_name),
            ));
        }

        let method_ident = format_ident!("{}_preset", preset_name);

        // Validate all preset fields exist in the struct
        for (field_ident, _) in &spec.fields {
            if !struct_field_set.contains(&field_ident.to_string()) {
                return Err(Error::new_spanned(
                    field_ident,
                    format!(
                        "#[preset(...)] field `{}` is not a field of `{}`",
                        field_ident, name
                    ),
                ));
            }
        }

        let set_fields: Vec<TokenStream> = spec
            .fields
            .iter()
            .map(|(ident, expr)| quote! { #ident: #expr })
            .collect();

        // Decide whether to use struct-update syntax
        let body = if spec.fields.len() == total_field_count {
            // All fields covered — no `..Default::default()` needed
            quote! {
                Self { #(#set_fields,)* }
            }
        } else {
            quote! {
                Self { #(#set_fields,)* ..::core::default::Default::default() }
            }
        };

        preset_methods.push(quote! {
            pub fn #method_ident() -> Self {
                #body
            }
        });
    }

    Ok(quote! {
        impl #name {
            #(#preset_methods)*
        }
    })
}
