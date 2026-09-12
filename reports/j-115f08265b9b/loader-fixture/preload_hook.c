extern void fake_cuda_runtime_symbol(void);

__attribute__((constructor)) static void load_hook(void) {
    fake_cuda_runtime_symbol();
}
