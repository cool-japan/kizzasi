use crate::attrs::parse_config_field_attrs;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Result};

/// Returns the inner type `T` if `ty` is `Option<T>` (matched on the last
/// path segment, so `Option<T>`, `std::option::Option<T>`, and
/// `::core::option::Option<T>` are all recognized; type aliases that hide
/// `Option` behind another name are not, since that is not visible from the
/// syntax alone).
fn option_inner_type(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    match args.args.first()? {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;
    let vis = &input.vis;
    let builder_name = format_ident!("{}Builder", name);
    let error_name = format_ident!("{}BuilderError", name);
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

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
    let mut builder_defaults = Vec::new();
    let mut builder_methods = Vec::new();
    let mut build_assignments = Vec::new();
    let mut validate_calls = Vec::new();

    for field in fields {
        let Some(fname) = field.ident.as_ref() else {
            // Unreachable: `Fields::Named` guarantees every field has an ident.
            continue;
        };
        let fty = &field.ty;
        let attrs = parse_config_field_attrs(field)?;

        if attrs.skip {
            // Excluded from builder entirely. Filled by default expr or Default::default().
            // A skipped field never gets a generated setter, so it cannot collide
            // with `new`/`build` regardless of its name.
            let fill = if let Some(default_expr) = &attrs.default {
                quote! { #fname: #default_expr }
            } else {
                quote! { #fname: ::core::default::Default::default() }
            };
            build_assignments.push(fill);
        } else {
            // Every branch below generates a `#vis fn #fname(...)` setter in
            // `impl #builder_name`, which already defines `new` and `build`;
            // reusing either name would be a duplicate-definition error deep
            // inside macro-generated code with no hint about the cause.
            if fname == "new" || fname == "build" {
                return Err(Error::new_spanned(
                    fname,
                    format!(
                        "field `{fname}` collides with the generated `{builder_name}::{fname}` \
                         method; rename the field, or add `#[config(skip)]` to exclude it from \
                         the builder"
                    ),
                ));
            }

            if let Some(default_expr) = &attrs.default {
                // Optional in builder — uses default when unset.
                builder_fields.push(quote! { #fname: ::core::option::Option<#fty> });
                builder_defaults.push(quote! { #fname: ::core::option::Option::None });
                builder_methods.push(quote! {
                    #vis fn #fname(mut self, value: #fty) -> Self {
                        self.#fname = ::core::option::Option::Some(value);
                        self
                    }
                });
                build_assignments.push(quote! {
                    #fname: self.#fname.unwrap_or_else(|| #default_expr)
                });
            } else if let Some(inner_ty) = option_inner_type(fty) {
                // `Option<T>` fields are implicitly optional: the builder slot
                // stays `Option<T>` (not `Option<Option<T>>`), the setter takes
                // the unwrapped `T`, and simply not calling it builds to `None`
                // — no more forced `.field(None)` for an already-optional field.
                builder_fields.push(quote! { #fname: ::core::option::Option<#inner_ty> });
                builder_defaults.push(quote! { #fname: ::core::option::Option::None });
                builder_methods.push(quote! {
                    #vis fn #fname(mut self, value: #inner_ty) -> Self {
                        self.#fname = ::core::option::Option::Some(value);
                        self
                    }
                });
                build_assignments.push(quote! { #fname: self.#fname });
            } else {
                // Plain required field.
                builder_fields.push(quote! { #fname: ::core::option::Option<#fty> });
                builder_defaults.push(quote! { #fname: ::core::option::Option::None });
                builder_methods.push(quote! {
                    #vis fn #fname(mut self, value: #fty) -> Self {
                        self.#fname = ::core::option::Option::Some(value);
                        self
                    }
                });
                build_assignments.push(quote! {
                    #fname: self.#fname.ok_or_else(|| #error_name::MissingField(stringify!(#fname)))?
                });
            }
        }

        // Validation is uniform regardless of which branch produced `built.#fname`.
        if let Some(vpath) = &attrs.validate {
            validate_calls.push(quote! {
                if let ::core::result::Result::Err(message) = #vpath(&built.#fname) {
                    return ::core::result::Result::Err(#error_name::Validation {
                        field: stringify!(#fname),
                        message,
                    });
                }
            });
        }
    }

    let builder_doc = format!("Builder for [`{name}`], generated by `#[derive(KizzasiConfig)]`.");
    let error_doc = format!(
        "Error returned by [`{builder_name}::build`] when a required field is missing or a \
         `#[config(validate = ...)]` check fails."
    );

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #name #ty_generics #where_clause {
            /// Create a new builder for this configuration.
            #vis fn builder() -> #builder_name #ty_generics {
                #builder_name::new()
            }
        }

        #[doc = #builder_doc]
        #vis struct #builder_name #impl_generics #where_clause {
            #(#builder_fields,)*
        }

        #[automatically_derived]
        impl #impl_generics ::core::default::Default for #builder_name #ty_generics #where_clause {
            fn default() -> Self {
                Self {
                    #(#builder_defaults,)*
                }
            }
        }

        #[doc = #error_doc]
        #[derive(Debug, Clone, PartialEq, Eq)]
        #vis enum #error_name {
            /// A required field (no `#[config(default = ...)]`) was never set.
            MissingField(&'static str),
            /// A `#[config(validate = ...)]` function rejected the built value.
            Validation {
                /// Name of the field that failed validation.
                field: &'static str,
                /// Message returned by the validation function.
                message: ::std::string::String,
            },
        }

        #[automatically_derived]
        impl ::std::fmt::Display for #error_name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                match self {
                    Self::MissingField(field) => {
                        write!(f, "missing required field `{}`", field)
                    }
                    Self::Validation { field, message } => {
                        write!(f, "validation failed for field `{}`: {}", field, message)
                    }
                }
            }
        }

        #[automatically_derived]
        impl ::std::error::Error for #error_name {}

        #[automatically_derived]
        impl ::std::convert::From<#error_name> for ::std::string::String {
            fn from(err: #error_name) -> Self {
                err.to_string()
            }
        }

        #[automatically_derived]
        impl #impl_generics #builder_name #ty_generics #where_clause {
            #vis fn new() -> Self {
                ::core::default::Default::default()
            }

            #(#builder_methods)*

            #vis fn build(self) -> ::std::result::Result<#name #ty_generics, #error_name> {
                let built = #name {
                    #(#build_assignments,)*
                };
                #(#validate_calls)*
                ::std::result::Result::Ok(built)
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
    /// `TokenStream` don't depend on proc-macro2's token-spacing rules
    /// (e.g. whether `pub(crate)` renders with a space before `(`).
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
        let err = expand_str("enum S { A, B }").expect_err("enum must be rejected");
        assert!(err.to_string().contains("only supports structs"));
    }

    // ---- id185: generics are now supported, not rejected ----------------

    #[test]
    fn accepts_lifetime_generic_struct() {
        assert!(expand_str("struct S<'a> { name: &'a str }").is_ok());
    }

    #[test]
    fn accepts_type_param_generic_struct() {
        assert!(expand_str("struct S<T> { value: T }").is_ok());
    }

    #[test]
    fn accepts_bounded_type_param_generic_struct() {
        assert!(expand_str("struct S<T: Clone> { value: T }").is_ok());
    }

    // ---- id187: field name collisions ------------------------------------

    #[test]
    fn rejects_field_named_build() {
        let err = expand_str("struct S { build: i32 }")
            .expect_err("field named `build` must be rejected");
        assert!(err.to_string().contains("collides"));
    }

    #[test]
    fn rejects_field_named_new() {
        let err =
            expand_str("struct S { new: i32 }").expect_err("field named `new` must be rejected");
        assert!(err.to_string().contains("collides"));
    }

    #[test]
    fn allows_field_named_builder() {
        // `Name::builder()` is an associated function, not a method; a field
        // named `builder` does not collide with it (verified empirically:
        // `struct Foo { builder: i32 } impl Foo { fn builder() -> X {..} }`
        // compiles fine — field access and path-qualified calls are
        // different namespaces). Do not over-reject working code.
        assert!(expand_str("struct S { builder: i32 } ").is_ok());
    }

    #[test]
    fn allows_skipped_field_named_build() {
        // A skipped field never generates a setter, so no collision is possible.
        assert!(expand_str("struct S { #[config(skip)] build: i32 }").is_ok());
    }

    // ---- id176: generated builder mirrors the struct's own visibility ----

    #[test]
    fn builder_struct_is_pub_for_pub_struct() {
        let tokens = expand_str("pub struct S { x: i32 }").expect("valid input");
        assert!(normalize(&tokens).contains("pubstructSBuilder"));
    }

    #[test]
    fn builder_struct_matches_pub_crate_visibility() {
        let tokens = expand_str("pub(crate) struct S { x: i32 }").expect("valid input");
        let n = normalize(&tokens);
        assert!(
            n.contains("pub(crate)structSBuilder"),
            "expected a `pub(crate) struct SBuilder` in generated tokens, got: {n}"
        );
        assert!(
            !n.contains("Result<S,String>"),
            "build() must not use the old untyped String error"
        );
    }

    #[test]
    fn builder_struct_is_private_for_private_struct() {
        let tokens = expand_str("struct S { x: i32 }").expect("valid input");
        let n = normalize(&tokens);
        assert!(!n.contains("pubstructSBuilder"));
        assert!(!n.contains("pub(crate)structSBuilder"));
    }

    // ---- id179: Option<T> fields are implicitly optional -----------------

    #[test]
    fn option_field_setter_takes_unwrapped_type() {
        let tokens = expand_str("struct S { value: Option<i32> } ").expect("valid input");
        let n = normalize(&tokens);
        // The setter must take `i32`, not `Option<i32>`.
        assert!(
            n.contains("fnvalue(mutself,value:i32)"),
            "expected an unwrapped `value: i32` setter, got: {n}"
        );
        assert!(!n.contains("fnvalue(mutself,value:Option<i32>)"));
    }

    #[test]
    fn option_field_does_not_use_missing_field_error() {
        let tokens = expand_str("struct S { value: Option<i32> } ").expect("valid input");
        assert!(!normalize(&tokens).contains("MissingField(stringify!(value)"));
    }

    #[test]
    fn required_field_still_uses_missing_field_error() {
        let tokens = expand_str("struct S { value: i32 } ").expect("valid input");
        assert!(normalize(&tokens).contains("MissingField(stringify!(value)"));
    }
}
