### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ e418bd00-2d67-11eb-0ef3-115fbdf2946c
using Flux

# ╔═╡ a6c04800-2d68-11eb-3174-05596a9e0194
using Flux: onehotbatch

# ╔═╡ f2f47300-2d67-11eb-2fe1-e1a394963618
X = Flux.Data.Iris.features()

# ╔═╡ 01d2e822-2d68-11eb-2b69-6b3dd4ccbd31
labels = Flux.Data.Iris.labels()

# ╔═╡ 1cf3f1d0-2d68-11eb-0826-fbe74790b5f0
unique(labels)

# ╔═╡ 3bc13aa0-2d68-11eb-394e-8940a113b917
z = indexin(labels, unique(labels))

# ╔═╡ 347c4eee-2d69-11eb-1bf4-33aaaedc92fe
y = onehotbatch(z, 1:3)

# ╔═╡ fb0f6c00-2d69-11eb-073d-937e53734939
#model = Chain(Dense(4, 3), softmax) #MLR model
model = Chain(Dense(4, 3, σ), Dense(3,3), softmax)

# ╔═╡ 1c234970-2d6a-11eb-34fe-7930c301b5de
loss(x, y) = Flux.crossentropy(model(x), y)

# ╔═╡ 881dcdce-2d6a-11eb-37ad-7f7f13d9eb23
θ = Flux.params(model)

# ╔═╡ af90ae50-2d6a-11eb-1eeb-8f6fb8809f22
θ[1]

# ╔═╡ c076da50-2d6a-11eb-03f5-6b6ba6e16221
θ[2]

# ╔═╡ c4b7df60-2d6a-11eb-15a6-9db5e2843a7b
θ[3]

# ╔═╡ c8e05360-2d6a-11eb-2045-33e5119540e3
loss(X, y)

# ╔═╡ d5c6f022-2d6a-11eb-17f7-a3e1c67dda20


# ╔═╡ Cell order:
# ╠═e418bd00-2d67-11eb-0ef3-115fbdf2946c
# ╠═f2f47300-2d67-11eb-2fe1-e1a394963618
# ╠═01d2e822-2d68-11eb-2b69-6b3dd4ccbd31
# ╠═1cf3f1d0-2d68-11eb-0826-fbe74790b5f0
# ╠═3bc13aa0-2d68-11eb-394e-8940a113b917
# ╠═a6c04800-2d68-11eb-3174-05596a9e0194
# ╠═347c4eee-2d69-11eb-1bf4-33aaaedc92fe
# ╠═fb0f6c00-2d69-11eb-073d-937e53734939
# ╠═1c234970-2d6a-11eb-34fe-7930c301b5de
# ╠═881dcdce-2d6a-11eb-37ad-7f7f13d9eb23
# ╠═af90ae50-2d6a-11eb-1eeb-8f6fb8809f22
# ╠═c076da50-2d6a-11eb-03f5-6b6ba6e16221
# ╠═c4b7df60-2d6a-11eb-15a6-9db5e2843a7b
# ╠═c8e05360-2d6a-11eb-2045-33e5119540e3
# ╠═d5c6f022-2d6a-11eb-17f7-a3e1c67dda20
