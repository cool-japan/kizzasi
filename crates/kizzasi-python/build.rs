// Build scripts communicate failure to cargo only via panic or a non-zero
// exit code (there is no `PyResult`-style error channel available here), so
// the panics below are this file's equivalent of a returned `Err` — not an
// instance of the crate's "no panic!/unwrap/expect in library code" policy,
// which governs `src/`, not `build.rs`. Each one is written to explain
// *what* failed and *how to fix it* (typically: set `PYO3_PYTHON`), matching
// the spirit of that policy even though the mechanism differs.
fn main() {
    // When `extension-module` is enabled, pyo3 suppresses automatic libpython linkage
    // (the Python interpreter provides symbols at runtime for cdylib extensions).
    // Test binaries are standalone executables and need explicit linkage.
    if std::env::var("CARGO_FEATURE_EXTENSION_MODULE").is_ok() {
        let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let output = std::process::Command::new(&python)
            .args([
                "-c",
                "import sysconfig; \
                 v = sysconfig.get_config_var('LDVERSION') or sysconfig.get_python_version(); \
                 d = sysconfig.get_config_var('LIBDIR') or ''; \
                 print(v); print(d)",
            ])
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to spawn Python interpreter '{python}' to query linking \
                     config: {e}\nSet the PYO3_PYTHON environment variable to a \
                     working `python3` executable."
                )
            });

        // The interpreter may have been spawned successfully but still
        // exited non-zero (missing sysconfig data, a broken Python install,
        // etc.) — previously this was never checked, so a failing probe
        // silently produced empty stdout and fell through to the
        // empty-version guard below with no indication of *why*.
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Python interpreter '{python}' exited with {status} while querying \
                 sysconfig for linking info; stderr:\n{stderr}\n\
                 Set the PYO3_PYTHON environment variable to a working `python3` \
                 executable.",
                status = output.status,
            );
        }

        let stdout = String::from_utf8(output.stdout).unwrap_or_else(|e| {
            panic!(
                "Python interpreter '{python}' produced non-UTF8 output while \
                 querying sysconfig for linking info: {e}"
            )
        });
        let mut lines = stdout.lines();
        let version = lines.next().unwrap_or("").trim().to_string();
        let libdir = lines.next().unwrap_or("").trim().to_string();

        // Unlike the `libdir` branch below (already guarded), an empty
        // `version` was previously allowed straight through to
        // `cargo:rustc-link-lib=python` — a bare `-lpython` flag that fails
        // the link step with an opaque linker error far removed from the
        // real cause (a Python whose sysconfig has neither `LDVERSION` nor
        // a usable `get_python_version()`).
        if version.is_empty() {
            panic!(
                "Python interpreter '{python}' returned an empty version string \
                 from sysconfig (both LDVERSION and get_python_version() were \
                 unavailable); refusing to emit a bare `-lpython` link flag. \
                 Set the PYO3_PYTHON environment variable to a working `python3` \
                 executable."
            );
        }

        println!("cargo:rustc-link-lib=python{version}");
        if !libdir.is_empty() {
            println!("cargo:rustc-link-search=native={libdir}");
        }
    }
}
