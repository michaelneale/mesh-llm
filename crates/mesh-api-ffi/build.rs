fn main() {
    uniffi::generate_scaffolding("src/mesh_ffi.udl").unwrap();
}
