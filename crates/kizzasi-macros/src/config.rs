use crate::attrs::parse_config_field_attrs;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Result};

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;
    let builder_name = format_ident!("{}Builder", name);

    // Guard: no generics
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &input.generics,
            "KizzasiConfig does not support generic structs",
        ));
    }

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(Error::new_spanned(
                    name,
                    "KizzasiConfig only supports structs with named fields",
                ))
            }
        },
        _ => {
            return Err(Error::new_spanned(
                name,
                "KizzasiConfig only supports structs",
            ))
        }
    };

    let mut builder_fields = Vec::new();
    let mut builder_methods = Vec::new();
    let mut build_assignments = Vec::new();
    let mut validate_calls = Vec::new();

    for field in fields {
        let fname = field
            .ident
            .as_ref()
            .expect("invariant: named fields always have idents");
        let fty = &field.ty;
        let attrs = parse_config_field_attrs(field)?;

        if attrs.skip {
            // Excluded from builder entirely. Filled by default expr or Default::default().
            let fill = if let Some(default_expr) = &attrs.default {
                quote! { #fname: #default_expr }
            } else {
                quote! { #fname: ::core::default::Default::default() }
            };
            build_assignments.push(fill);
            // Validate skipped-but-validate fields against their resolved value
            if let Some(vpath) = &attrs.validate {
                validate_calls.push(quote! {
                    if let ::core::result::Result::Err(msg) = #vpath(&built.#fname) {
                        return ::core::result::Result::Err(msg);
                    }
                });
            }
        } else if let Some(default_expr) = &attrs.default {
            // Optional in builder — uses default when unset
            builder_fields.push(quote! { #fname: ::core::option::Option<#fty> });
            builder_methods.push(quote! {
                pub fn #fname(mut self, value: #fty) -> Self {
                    self.#fname = ::core::option::Option::Some(value);
                    self
                }
            });
            build_assignments.push(quote! {
                #fname: self.#fname.unwrap_or_else(|| #default_expr)
            });
            if let Some(vpath) = &attrs.validate {
                validate_calls.push(quote! {
                    if let ::core::result::Result::Err(msg) = #vpath(&built.#fname) {
                        return ::core::result::Result::Err(msg);
                    }
                });
            }
        } else {
            // Plain required field (current behavior preserved)
            builder_fields.push(quote! { #fname: ::core::option::Option<#fty> });
            builder_methods.push(quote! {
                pub fn #fname(mut self, value: #fty) -> Self {
                    self.#fname = ::core::option::Option::Some(value);
                    self
                }
            });
            build_assignments.push(quote! {
                #fname: self.#fname.ok_or_else(|| ::std::format!("Missing required field: {}", stringify!(#fname)))?
            });
            if let Some(vpath) = &attrs.validate {
                validate_calls.push(quote! {
                    if let ::core::result::Result::Err(msg) = #vpath(&built.#fname) {
                        return ::core::result::Result::Err(msg);
                    }
                });
            }
        }
    }

    Ok(quote! {
        impl #name {
            /// Create a new builder for this configuration.
            pub fn builder() -> #builder_name {
                #builder_name::new()
            }
        }

        #[derive(Default)]
        pub struct #builder_name {
            #(#builder_fields,)*
        }

        impl #builder_name {
            pub fn new() -> Self {
                Self::default()
            }

            #(#builder_methods)*

            pub fn build(self) -> ::std::result::Result<#name, ::std::string::String> {
                let built = #name {
                    #(#build_assignments,)*
                };
                #(#validate_calls)*
                ::std::result::Result::Ok(built)
            }
        }
    })
}
