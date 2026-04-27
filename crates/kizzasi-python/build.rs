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
            .expect("failed to query Python config for linking");
        let stdout = String::from_utf8(output.stdout).expect("non-UTF8 python output");
        let mut lines = stdout.lines();
        let version = lines.next().unwrap_or("").trim().to_string();
        let libdir = lines.next().unwrap_or("").trim().to_string();

        println!("cargo:rustc-link-lib=python{version}");
        if !libdir.is_empty() {
            println!("cargo:rustc-link-search=native={libdir}");
        }
    }
}
