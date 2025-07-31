# Stolen from Flux as well

for n in [2, 20, 200, 2000]
    x = randn(Float32, n, 2000, 100)
    model = FourierLayer(n, n, 2000, 100, 16)
    println("CPU n=$n")
    run_benchmark(model, x, cuda = false)
    println("CUDA n=$n")
    run_benchmark(model, x, cuda = true)
end
