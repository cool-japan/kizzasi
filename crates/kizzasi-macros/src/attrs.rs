use syn::parse::ParseStream;
use syn::{punctuated::Punctuated, Expr, Ident, LitStr, MetaNameValue, Path, Token};

/// Parse a `default = ...` value, supporting either:
/// - `default = "some_expr_string"` (string literal that is itself parsed as an Expr)
/// - `default = <any valid Rust expression>` (e.g. integer literal, path, call, etc.)
///
/// Using a lookahead here is critical: we must not consume tokens from a failed parse.
fn parse_default_expr(input: ParseStream<'_>) -> syn::Result<Expr> {
    // Peek: if the next token is a string literal, try interpreting it as a source expression.
    let lookahead = input.lookahead1();
    if lookahead.peek(LitStr) {
        let lit: LitStr = input.parse()?;
        syn::parse_str::<Expr>(&lit.value()).map_err(|e| syn::Error::new(lit.span(), e))
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

/// Parse `#[config(...)]` attributes from a field. Returns error on unknown keys.
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
                out.skip = true;
                Ok(())
            } else if meta.path.is_ident("default") {
                let value = meta.value()?;
                // Support both `default = "expr string"` (string lit → parse as expr)
                // and `default = EXPR` (any expression, incl. integer literals).
                // We use a lookahead to decide which path to take to avoid consuming
                // tokens from a failed parse attempt.
                let expr: Expr = parse_default_expr(value)?;
                out.default = Some(expr);
                Ok(())
            } else if meta.path.is_ident("validate") {
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

/// Parse a single `#[preset(...)]` attribute into a `PresetSpec`.
pub fn parse_preset_attr(attr: &syn::Attribute) -> syn::Result<PresetSpec> {
    let pairs = attr.parse_args_with(Punctuated::<MetaNameValue, Token![,]>::parse_terminated)?;

    let mut name: Option<LitStr> = None;
    let mut fields: Vec<(Ident, Expr)> = Vec::new();

    for nv in &pairs {
        if nv.path.is_ident("name") {
            match &nv.value {
                Expr::Lit(expr_lit) => {
                    if let syn::Lit::Str(s) = &expr_lit.lit {
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
            fields.push((ident, nv.value.clone()));
        }
    }

    let name = name.ok_or_else(|| {
        syn::Error::new_spanned(attr, "#[preset(...)] requires a `name = \"...\"` key")
    })?;
    Ok(PresetSpec { name, fields })
}
