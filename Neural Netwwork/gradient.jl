### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ c4168992-2d5a-11eb-046f-0d79a51c1298
using Flux

# ╔═╡ a6d2bf90-2d5d-11eb-39ab-7d9815191b2c
using Flux: gradient

# ╔═╡ 74b78ad0-2d63-11eb-1f46-e1813afccd1a
using Flux: params

# ╔═╡ ff615770-2d5d-11eb-3c3a-4fb8c5f074a7
f(x) = 3x^2 + 2x + 1

# ╔═╡ 09f63c00-2d5e-11eb-3f3f-6b97874cb03b
df(x) = gradient(f, x)

# ╔═╡ 6631fd60-2d5e-11eb-033f-0f454606681e
df(4)

# ╔═╡ 693ad090-2d5e-11eb-2274-bf0b9448ad52
mysin(x) = sum((-1)^k * x^(1+2k) / factorial(1+2k) for k = 0:5)

# ╔═╡ 572eccc0-2d5f-11eb-0340-5dcd5542498c
x = 0.5

# ╔═╡ 5d96f790-2d5f-11eb-311e-47d4a7e91938
mysin(x)

# ╔═╡ 61363e60-2d5f-11eb-2620-13e4bbb8c7a7
sin(x)

# ╔═╡ 6dbcb9c0-2d5f-11eb-2c55-2355254ea7b5
gradient(mysin, x)

# ╔═╡ 9cbae850-2d5f-11eb-2aad-9bc5fd2a641e
cos(x)

# ╔═╡ a88cb9b0-2d5f-11eb-2950-67175b5c3af1
myLoss(W, b, x) = sum(W * x .+ b)

# ╔═╡ 2a43f860-2d60-11eb-36a6-0b159f417a56
W = rand(3, 4)

# ╔═╡ 47c4c400-2d60-11eb-1238-e982921c60eb
b = zeros(3)

# ╔═╡ 60deda20-2d60-11eb-39e7-89296926b7ea
x0 = rand(4)

# ╔═╡ 79ebd0e0-2d60-11eb-3948-3d4d0d2bee8d
myLoss(W, b, x0)

# ╔═╡ 88008d60-2d60-11eb-129c-1b0daa629d77
grad = gradient(myLoss, W, b, x0)

# ╔═╡ dd5cd110-2d60-11eb-1580-1bbf2a95acb8
grad[1]

# ╔═╡ 767fb290-2d61-11eb-3002-89ee7548967d
grad[2]

# ╔═╡ 7ba39d90-2d61-11eb-1fc3-3b3ffa95d0cf
grad[3]

# ╔═╡ db4868a2-2d63-11eb-23b0-39654ac23be3
loss(x) = sum(W * x .+ b)

# ╔═╡ e5a37880-2d63-11eb-0645-7f37b8057ef6
gs = gradient(() -> loss(x), params([W, b]))

# ╔═╡ fffe7310-2d63-11eb-2a1e-15984b7cca45
gs[W]

# ╔═╡ 0646b9d0-2d64-11eb-2e04-c9c3487b1f1f
gs[b]

# ╔═╡ 0d5f6ff0-2d64-11eb-3429-551abe7959ed
model = Dense(10, 5)

# ╔═╡ 8e1c4912-2d64-11eb-3df8-2f05ed25e14d
ps = params(model)

# ╔═╡ b542de4e-2d64-11eb-09a9-d3709043234b
ps[1]

# ╔═╡ bbd17ce0-2d64-11eb-0a95-47653fd9953e
ps[2]

# ╔═╡ de623650-2d64-11eb-1797-7520c9270f81
x1 = rand(10)

# ╔═╡ 09d2d152-2d65-11eb-29af-3584081b0e07
ps[1] * x1 .+ ps[2]

# ╔═╡ 19dea2e2-2d65-11eb-1587-5f6dd6f8350a
model(x1)

# ╔═╡ 56256450-2d65-11eb-18ad-571734d8f904
u = rand(4)

# ╔═╡ 9699c9e0-2d65-11eb-1bcf-577d10a04a02
v = softmax(u)

# ╔═╡ ae0ac3e2-2d65-11eb-08b7-451a0e51c45f
m = Chain(Dense(10, 5), Dense(5, 3), softmax)

# ╔═╡ 9bc9c2d2-2d65-11eb-0009-3b91553e22d9
m(x1)

# ╔═╡ 53085060-2d66-11eb-0447-5b7b5861c486
params(m)[1]

# ╔═╡ 95f0ea40-2d66-11eb-0291-23fe9fab2c8c
params(m)[2]

# ╔═╡ a8c21d60-2d66-11eb-1b8d-87b7e6e09d40
params(m)[3]

# ╔═╡ f0edea10-2d66-11eb-108c-392ff143dcd8
params(m)[4]

# ╔═╡ f56e9030-2d66-11eb-037a-c9b8a85b3a12


# ╔═╡ Cell order:
# ╠═c4168992-2d5a-11eb-046f-0d79a51c1298
# ╠═a6d2bf90-2d5d-11eb-39ab-7d9815191b2c
# ╠═ff615770-2d5d-11eb-3c3a-4fb8c5f074a7
# ╠═09f63c00-2d5e-11eb-3f3f-6b97874cb03b
# ╠═6631fd60-2d5e-11eb-033f-0f454606681e
# ╠═693ad090-2d5e-11eb-2274-bf0b9448ad52
# ╠═572eccc0-2d5f-11eb-0340-5dcd5542498c
# ╠═5d96f790-2d5f-11eb-311e-47d4a7e91938
# ╠═61363e60-2d5f-11eb-2620-13e4bbb8c7a7
# ╠═6dbcb9c0-2d5f-11eb-2c55-2355254ea7b5
# ╠═9cbae850-2d5f-11eb-2aad-9bc5fd2a641e
# ╠═a88cb9b0-2d5f-11eb-2950-67175b5c3af1
# ╠═2a43f860-2d60-11eb-36a6-0b159f417a56
# ╠═47c4c400-2d60-11eb-1238-e982921c60eb
# ╠═60deda20-2d60-11eb-39e7-89296926b7ea
# ╠═79ebd0e0-2d60-11eb-3948-3d4d0d2bee8d
# ╠═88008d60-2d60-11eb-129c-1b0daa629d77
# ╠═dd5cd110-2d60-11eb-1580-1bbf2a95acb8
# ╠═767fb290-2d61-11eb-3002-89ee7548967d
# ╠═7ba39d90-2d61-11eb-1fc3-3b3ffa95d0cf
# ╠═74b78ad0-2d63-11eb-1f46-e1813afccd1a
# ╠═db4868a2-2d63-11eb-23b0-39654ac23be3
# ╠═e5a37880-2d63-11eb-0645-7f37b8057ef6
# ╠═fffe7310-2d63-11eb-2a1e-15984b7cca45
# ╠═0646b9d0-2d64-11eb-2e04-c9c3487b1f1f
# ╠═0d5f6ff0-2d64-11eb-3429-551abe7959ed
# ╠═8e1c4912-2d64-11eb-3df8-2f05ed25e14d
# ╠═b542de4e-2d64-11eb-09a9-d3709043234b
# ╠═bbd17ce0-2d64-11eb-0a95-47653fd9953e
# ╠═de623650-2d64-11eb-1797-7520c9270f81
# ╠═09d2d152-2d65-11eb-29af-3584081b0e07
# ╠═19dea2e2-2d65-11eb-1587-5f6dd6f8350a
# ╠═56256450-2d65-11eb-18ad-571734d8f904
# ╠═9699c9e0-2d65-11eb-1bcf-577d10a04a02
# ╠═ae0ac3e2-2d65-11eb-08b7-451a0e51c45f
# ╠═9bc9c2d2-2d65-11eb-0009-3b91553e22d9
# ╠═53085060-2d66-11eb-0447-5b7b5861c486
# ╠═95f0ea40-2d66-11eb-0291-23fe9fab2c8c
# ╠═a8c21d60-2d66-11eb-1b8d-87b7e6e09d40
# ╠═f0edea10-2d66-11eb-108c-392ff143dcd8
# ╠═f56e9030-2d66-11eb-037a-c9b8a85b3a12
