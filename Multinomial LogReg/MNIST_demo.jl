### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ 65fd3240-225c-11eb-35db-b72bda34ff84
using MLDatasets

# ╔═╡ f4da9cd0-225e-11eb-0047-550092ee7fd4
using ImageCore

# ╔═╡ b79da670-225c-11eb-371a-4f0eb314b24a
MNIST.traintensor(1)

# ╔═╡ fa87dc0e-225e-11eb-253c-2d0e75829263
MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ 06e9df80-225f-11eb-14e3-2b2e5ac3f84a
begin
	N = 1000
	A = MNIST.traintensor(Float32, 1:N)
	y = MNIST.trainlabels(1:N) .+ 1
end

# ╔═╡ 35964490-225f-11eb-1bea-17d1f65914bf
size(A)

# ╔═╡ 40ee2b00-225f-11eb-3eaf-53cbd940eba9
begin
	D = 28 * 28
	X = zeros(N, D)
	for i = 1:N
		X[i,:] = vec(A[:, :, i])
	end
end

# ╔═╡ b4e276b0-225f-11eb-08e6-af63478fb686
D

# ╔═╡ b982b2c0-225f-11eb-2b6f-715f0661ce10
X[1,:]

# ╔═╡ bf92d280-225f-11eb-3d57-6f10bbb441f9
θ = zeros(10, D)

# ╔═╡ 4210bc90-2260-11eb-1d52-879275178f1c
# Implpementaion MLR model
# X: N*D matrix, y: N*1 col vector, θ: K*D matrix
function cost(X::Array{Float64,2}, y::Array{Int}, θ::Array{Float64,2})::Float64
	u = sum(θ[y, :] .* X, dims = 2)
	v = sum(exp.(θ * X'), dims = 1)
	J = -sum(u) + sum(log.(v))
	J/N + 0.
end

# ╔═╡ 32d34ca0-2262-11eb-3c16-d35fcb529c90
cost(X, y, θ)

# ╔═╡ 51a55062-2262-11eb-2850-b738438d4897
function gradient(X::Array{Float64,2}, y::Array{Int}, θ::Array{Float64,2})::Array{Float64}

# ╔═╡ Cell order:
# ╠═65fd3240-225c-11eb-35db-b72bda34ff84
# ╠═b79da670-225c-11eb-371a-4f0eb314b24a
# ╠═f4da9cd0-225e-11eb-0047-550092ee7fd4
# ╠═fa87dc0e-225e-11eb-253c-2d0e75829263
# ╠═06e9df80-225f-11eb-14e3-2b2e5ac3f84a
# ╠═35964490-225f-11eb-1bea-17d1f65914bf
# ╠═40ee2b00-225f-11eb-3eaf-53cbd940eba9
# ╠═b4e276b0-225f-11eb-08e6-af63478fb686
# ╠═b982b2c0-225f-11eb-2b6f-715f0661ce10
# ╠═bf92d280-225f-11eb-3d57-6f10bbb441f9
# ╠═4210bc90-2260-11eb-1d52-879275178f1c
# ╠═32d34ca0-2262-11eb-3c16-d35fcb529c90
# ╠═51a55062-2262-11eb-2850-b738438d4897
