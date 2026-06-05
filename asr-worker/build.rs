fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux") {
        println!("cargo:rustc-link-lib=gomp");
    }
}
