use std::collections::HashSet;
use syn::parse::ParseStream;
use syn::{punctuated::Punctuated, Expr, Ident, LitStr, MetaNameValue, Path, Token};

/// Returns `true` when `expr` is a bare, single-segment path with no
/// arguments — i.e. what a plain identifier like `None` or `MAX` parses to.
///
/// This is used to detect the dangerous case in [`parse_default_expr`]: a
/// quoted string that happens to *also* be a valid identifier is ambiguous
/// between "the path/const named X" and "the literal text X", and silently
/// picking the former has caused real bugs (see `#[config(default = "...")]`
/// documentation below).
fn is_bare_single_segment_path(expr: &Expr) -> bool {
    let Expr::Path(p) = expr else {
        return false;
    };
    if p.qself.is_some() || p.path.leading_colon.is_some() || p.path.segments.len() != 1 {
        return false;
    }
    p.path
        .segments
        .first()
        .is_some_and(|segment| segment.arguments.is_empty())
}

/// Parse a `default = ...` value, supporting either:
/// - `default = "some_expr_string"` (string literal that is itself parsed as an Expr)
/// - `default = <any valid Rust expression>` (e.g. integer literal, path, call, etc.)
///
/// Using a lookahead here is critical: we must not consume tokens from a failed parse.
///
/// The quoted form re-parses the string's *contents* as Rust source, mirroring
/// `#[config(validate = "path::to::fn")]`. That means it must not be used to
/// spell a literal string value: `#[config(default = "hello")]` does not mean
/// "the string `hello`", it means "the expression `hello`" (a path). To keep
/// that mistake from silently binding the wrong value, a quoted string that
/// reduces to a bare single-identifier path is rejected outright — write the
/// path unquoted (`default = hello`) if that's what you meant, or escape a
/// genuine string/`String` value as source text, e.g.
/// `default = "\"hello\""` (`&str`) or `default = "\"hello\".to_string()"`
/// (`String`).
fn parse_default_expr(input: ParseStream<'_>) -> syn::Result<Expr> {
    // Peek: if the next token is a string literal, try interpreting it as a source expression.
    let lookahead = input.lookahead1();
    if lookahead.peek(LitStr) {
        let lit: LitStr = input.parse()?;
        let expr =
            syn::parse_str::<Expr>(&lit.value()).map_err(|e| syn::Error::new(lit.span(), e))?;
        if is_bare_single_segment_path(&expr) {
            let text = lit.value();
            return Err(syn::Error::new(
                lit.span(),
                format!(
                    "#[config(default = \"{text}\")] is ambiguous: the string was reinterpreted \
                     as Rust source and parsed as the path expression `{text}`. If you meant \
                     that path or constant, write it unquoted: `default = {text}`. If you meant \
                     the literal text \"{text}\", escape it as source instead, e.g. \
                     `default = \"\\\"{text}\\\"\"` for `&str` or \
                     `default = \"\\\"{text}\\\".to_string()\"` for `String`."
                ),
            ));
        }
        Ok(expr)
    } else {
        input.parse::<Expr>()
    }
}

/// Parsed `#[config(...)]` attribute options on a single struct field.
pub struct ConfigFieldAttrs {
    /// `#[config(default = EXPR)]` or `#[config(default = "expr_string")]`
    pub default: Option<Expr>,
    /// `#[config(validate = "path::to::fn")]` — fn(&T) -> Result<(), String>
    pub validate: Option<Path>,
    /// `#[config(skip)]` — field excluded from builder, filled by Default::default() or default expr
    pub skip: bool,
}

/// Parse `#[config(...)]` attributes from a field. Returns error on unknown
/// keys and on any key repeated (whether within one `#[config(...)]`
/// attribute or across several stacked on the same field), since a repeat
/// would otherwise silently overwrite the earlier value with no diagnostic.
pub fn parse_config_field_attrs(field: &syn::Field) -> syn::Result<ConfigFieldAttrs> {
    let mut out = ConfigFieldAttrs {
        default: None,
        validate: None,
        skip: false,
    };
    for attr in &field.attrs {
        if !attr.path().is_ident("config") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                if out.skip {
                    return Err(meta.error("duplicate `skip` key in #[config(...)]"));
                }
                out.skip = true;
                Ok(())
            } else if meta.path.is_ident("default") {
                if out.default.is_some() {
                    return Err(meta.error("duplicate `default` key in #[config(...)]"));
                }
                let value = meta.value()?;
                // Support both `default = "expr string"` (string lit → parse as expr)
                // and `default = EXPR` (any expression, incl. integer literals).
                // We use a lookahead to decide which path to take to avoid consuming
                // tokens from a failed parse attempt.
                let expr: Expr = parse_default_expr(value)?;
                out.default = Some(expr);
                Ok(())
            } else if meta.path.is_ident("validate") {
                if out.validate.is_some() {
                    return Err(meta.error("duplicate `validate` key in #[config(...)]"));
                }
                let lit: LitStr = meta.value()?.parse()?;
                out.validate = Some(syn::parse_str::<Path>(&lit.value())?);
                Ok(())
            } else {
                Err(meta.error(
                    "unsupported #[config(...)] key; expected `default`, `validate`, or `skip`",
                ))
            }
        })?;
    }
    Ok(out)
}

/// One parsed `#[preset(name = "NAME", field1 = val1, ...)]` attribute.
pub struct PresetSpec {
    pub name: LitStr,
    pub fields: Vec<(Ident, Expr)>,
}

/// Validate that `name` is usable as the identifier fragment of the
/// generated `<name>_preset` associated function: it must (a) combine with
/// the `_preset` suffix into a syntactically valid Rust identifier — this is
/// what protects `format_ident!` from panicking — and (b) be snake_case, so
/// the generated function does not trigger a `non_snake_case` warning in the
/// consuming crate.
fn validate_preset_name(name: &LitStr) -> syn::Result<()> {
    let value = name.value();
    if value.is_empty() {
        return Err(syn::Error::new(
            name.span(),
            "#[preset(name = \"...\")] must not be empty",
        ));
    }
    let method_name = format!("{value}_preset");
    if syn::parse_str::<syn::Ident>(&method_name).is_err() {
        return Err(syn::Error::new(
            name.span(),
            format!(
                "#[preset(name = \"{value}\")] cannot be used as an identifier: it must generate \
                 a valid Rust function name (`{method_name}`)"
            ),
        ));
    }
    if value.contains(char::is_uppercase) {
        return Err(syn::Error::new(
            name.span(),
            format!(
                "#[preset(name = \"{value}\")] must be snake_case: `{method_name}` would \
                 trigger a `non_snake_case` warning"
            ),
        ));
    }
    Ok(())
}

/// Parse a single `#[preset(...)]` attribute into a `PresetSpec`. Rejects a
/// repeated `name = ...` key and repeated field keys within the same
/// attribute, since either would otherwise silently keep only the last
/// value with no diagnostic (and, for fields, would corrupt the
/// full-coverage field count used to decide whether `..Default::default()`
/// is needed).
pub fn parse_preset_attr(attr: &syn::Attribute) -> syn::Result<PresetSpec> {
    let pairs = attr.parse_args_with(Punctuated::<MetaNameValue, Token![,]>::parse_terminated)?;

    let mut name: Option<LitStr> = None;
    let mut fields: Vec<(Ident, Expr)> = Vec::new();
    let mut seen_fields: HashSet<String> = HashSet::new();

    for nv in &pairs {
        if nv.path.is_ident("name") {
            if name.is_some() {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    "duplicate `name` key in #[preset(...)]",
                ));
            }
            match &nv.value {
                Expr::Lit(expr_lit) => {
                    if let syn::Lit::Str(s) = &expr_lit.lit {
                        validate_preset_name(s)?;
                        name = Some(s.clone());
                    } else {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "#[preset(name = \"...\")] must be a string literal",
                        ));
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "#[preset(name = \"...\")] must be a string literal",
                    ))
                }
            }
        } else {
            let ident = nv.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(&nv.path, "preset field must be a simple identifier")
            })?;
            if !seen_fields.insert(ident.to_string()) {
                return Err(syn::Error::new_spanned(
                    &ident,
                    format!("duplicate field `{ident}` in #[preset(...)]"),
                ));
            }
            fields.push((ident, nv.value.clone()));
        }
    }

    let name = name.ok_or_else(|| {
        syn::Error::new_spanned(attr, "#[preset(...)] requires a `name = \"...\"` key")
    })?;
    Ok(PresetSpec { name, fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::DeriveInput;

    fn parse_first_attr(src: &str) -> syn::Attribute {
        let input: DeriveInput = syn::parse_str(src).expect("valid test struct");
        input
            .attrs
            .into_iter()
            .next()
            .expect("test struct has at least one attribute")
    }

    fn parse_first_field(src: &str) -> syn::Field {
        let input: DeriveInput = syn::parse_str(src).expect("valid test struct");
        match input.data {
            syn::Data::Struct(data) => match data.fields {
                syn::Fields::Named(named) => named
                    .named
                    .into_iter()
                    .next()
                    .expect("test struct has at least one field"),
                _ => panic!("test struct must have named fields"),
            },
            _ => panic!("test input must be a struct"),
        }
    }

    /// Like `Result::expect_err`, but without requiring `T: Debug` — several
    /// of the `Ok` types here (`PresetSpec`, `ConfigFieldAttrs`) wrap `syn`
    /// AST nodes that only implement `Debug` behind `syn`'s `extra-traits`
    /// feature, which this crate does not otherwise need.
    fn err_of<T, E>(result: Result<T, E>, msg: &str) -> E {
        match result {
            Err(e) => e,
            Ok(_) => panic!("{msg}"),
        }
    }

    // ---- parse_preset_attr: valid inputs -------------------------------

    #[test]
    fn preset_attr_valid_snake_case_name_ok() {
        let attr = parse_first_attr(r#"#[preset(name = "audio", x = 1)] struct S { x: i32 }"#);
        let spec = parse_preset_attr(&attr).expect("valid preset attr");
        assert_eq!(spec.name.value(), "audio");
        assert_eq!(spec.fields.len(), 1);
    }

    #[test]
    fn preset_attr_underscore_name_ok() {
        let attr = parse_first_attr(r#"#[preset(name = "_internal")] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_ok());
    }

    // ---- id175: preset name validation ---------------------------------

    #[test]
    fn preset_name_with_hyphen_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = "fast-mode")] struct S { x: i32 }"#);
        let err = err_of(parse_preset_attr(&attr), "hyphenated name must be rejected");
        assert!(err.to_string().contains("identifier"));
    }

    #[test]
    fn preset_name_with_space_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = "low latency")] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_err());
    }

    #[test]
    fn preset_name_leading_digit_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = "2x")] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_err());
    }

    #[test]
    fn preset_name_empty_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = "")] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_err());
    }

    #[test]
    fn preset_name_uppercase_is_rejected_for_snake_case() {
        let attr = parse_first_attr(r#"#[preset(name = "Fast")] struct S { x: i32 }"#);
        let err = err_of(
            parse_preset_attr(&attr),
            "non-snake_case name must be rejected",
        );
        assert!(err.to_string().contains("snake_case"));
    }

    // ---- id180: duplicate keys inside one #[preset(...)] ----------------

    #[test]
    fn preset_attr_duplicate_field_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = "a", x = 1, x = 2)] struct S { x: i32 }"#);
        let err = err_of(
            parse_preset_attr(&attr),
            "duplicate field key must be rejected",
        );
        assert!(err.to_string().contains("duplicate field"));
    }

    #[test]
    fn preset_attr_duplicate_name_is_rejected() {
        let attr =
            parse_first_attr(r#"#[preset(name = "a", name = "b", x = 1)] struct S { x: i32 }"#);
        let err = err_of(
            parse_preset_attr(&attr),
            "duplicate name key must be rejected",
        );
        assert!(err.to_string().contains("duplicate `name`"));
    }

    // ---- pre-existing error sites (previously untested per audit) -------

    #[test]
    fn preset_attr_non_literal_name_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(name = SOME_CONST)] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_err());
    }

    #[test]
    fn preset_attr_missing_name_is_rejected() {
        let attr = parse_first_attr(r#"#[preset(x = 1)] struct S { x: i32 }"#);
        let err = err_of(
            parse_preset_attr(&attr),
            "missing name key must be rejected",
        );
        assert!(err.to_string().contains("name"));
    }

    #[test]
    fn preset_attr_non_ident_field_is_rejected() {
        let attr =
            parse_first_attr(r#"#[preset(name = "a", "not_an_ident" = 1)] struct S { x: i32 }"#);
        assert!(parse_preset_attr(&attr).is_err());
    }

    // ---- parse_config_field_attrs ---------------------------------------

    #[test]
    fn config_field_attrs_unknown_key_is_rejected() {
        let field = parse_first_field(r#"struct S { #[config(bogus = 1)] x: i32 }"#);
        let err = err_of(
            parse_config_field_attrs(&field),
            "unknown key must be rejected",
        );
        assert!(err.to_string().contains("unsupported"));
    }

    // ---- id190: duplicate #[config(...)] keys ----------------------------

    #[test]
    fn config_field_attrs_duplicate_default_same_attr_is_rejected() {
        let field = parse_first_field(r#"struct S { #[config(default = 1, default = 2)] x: i32 }"#);
        let err = err_of(
            parse_config_field_attrs(&field),
            "duplicate default must be rejected",
        );
        assert!(err.to_string().contains("duplicate `default`"));
    }

    #[test]
    fn config_field_attrs_duplicate_default_separate_attrs_is_rejected() {
        let field = parse_first_field(
            r#"struct S { #[config(default = 1)] #[config(default = 2)] x: i32 }"#,
        );
        assert!(parse_config_field_attrs(&field).is_err());
    }

    #[test]
    fn config_field_attrs_duplicate_validate_is_rejected() {
        let field =
            parse_first_field(r#"struct S { #[config(validate = "a", validate = "b")] x: i32 }"#);
        let err = err_of(
            parse_config_field_attrs(&field),
            "duplicate validate must be rejected",
        );
        assert!(err.to_string().contains("duplicate `validate`"));
    }

    #[test]
    fn config_field_attrs_duplicate_skip_is_rejected() {
        let field = parse_first_field(r#"struct S { #[config(skip, skip)] x: i32 }"#);
        let err = err_of(
            parse_config_field_attrs(&field),
            "duplicate skip must be rejected",
        );
        assert!(err.to_string().contains("duplicate `skip`"));
    }

    // ---- id178: ambiguous string-as-source default ----------------------

    #[test]
    fn config_default_string_bare_path_is_rejected() {
        let field = parse_first_field(r#"struct S { #[config(default = "None")] x: i32 }"#);
        let err = err_of(
            parse_config_field_attrs(&field),
            "bare-path default must be rejected",
        );
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn config_default_string_bare_path_uppercase_is_rejected() {
        let field = parse_first_field(r#"struct S { #[config(default = "MAX")] x: i32 }"#);
        assert!(parse_config_field_attrs(&field).is_err());
    }

    #[test]
    fn config_default_string_numeric_literal_still_works() {
        // Regression guard: this exact form is relied on by tests/config_derive.rs
        // and examples/kizzasi_config_derive.rs and must keep working.
        let field = parse_first_field(r#"struct S { #[config(default = "256_usize")] x: usize }"#);
        let attrs = parse_config_field_attrs(&field).expect("numeric literal string must parse");
        assert!(attrs.default.is_some());
    }

    #[test]
    fn config_default_string_method_call_works_for_string_default() {
        // The escape hatch for a genuine `String` default: quote the literal
        // string itself as source text.
        let field = parse_first_field(
            r#"struct S { #[config(default = "\"hello\".to_string()")] x: String }"#,
        );
        let attrs = parse_config_field_attrs(&field).expect("method-call string must parse");
        assert!(attrs.default.is_some());
    }

    #[test]
    fn config_default_bare_expr_still_works() {
        let field = parse_first_field(r#"struct S { #[config(default = 4096)] x: usize }"#);
        assert!(parse_config_field_attrs(&field).is_ok());
    }
}
