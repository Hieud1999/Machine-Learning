### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ a0306e80-32da-11eb-2c41-558c63efaf94
begin
	using MLDatasets
	using ImageCore
end

# ╔═╡ 1bb46322-32dd-11eb-1be8-f1e0819530e9
using Flux:onehotbatch

# ╔═╡ dc29d670-32de-11eb-3afd-81eaeb17b026
using Flux

# ╔═╡ 67608ab0-32e2-11eb-33f2-31f1de17f966
using Flux:@epochs

# ╔═╡ 6ac93bb0-32e3-11eb-0113-d757fbc576cf
using PlutoUI

# ╔═╡ 97375410-32e4-11eb-1bcd-11d2601fcc44
using Statistics

# ╔═╡ 488451f0-32db-11eb-3fe5-f159303cd746
function readTrainingData(N)
	A = MNIST.traintensor(Float32, 1:N)
	X = Float32.(zeros(28 * 28, N))
	for i = 1:N
		X[:, i] = vec(A[:, :, i])
	end
	y = MNIST.trainlabels(1:N) .+ 1
	(X, y)
end

# ╔═╡ 372f1b00-32dc-11eb-2310-1d3833ff1ca5
N = 60000

# ╔═╡ 4df86350-32dc-11eb-0daa-276675a3d961
X_train, y_train = readTrainingData(N)

# ╔═╡ 576569b0-32dc-11eb-3a0e-655b24db92b7
function readTestData(M)
	A = MNIST.traintensor(Float32, 1:M)
	X = Float32.(zeros(28 * 28, M))
	for i = 1:M
		X[:, i] = vec(A[:, :, i])
	end
	y = MNIST.trainlabels(1:M) .+ 1
	(X, y)
end

# ╔═╡ a415d420-32dc-11eb-3be9-abc0226d1726
M = 5000

# ╔═╡ ab949f10-32dc-11eb-2428-a182fa327d16
X_test, y_test = readTestData(M)

# ╔═╡ b5e1a530-32dc-11eb-3040-675be8aa4231
first_x = MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ 0a97e4e0-32dd-11eb-0c82-b1bd36e26c2e
first_y = y_train[1]

# ╔═╡ 842db000-32dd-11eb-2554-13be22ac66f5
Y_train = onehotbatch(y_train, 1:10)

# ╔═╡ a7cb1e80-32dd-11eb-2490-a373bd6e3537
model = Chain(Dense(784, 128, σ), Dense(128, 10), softmax)

# ╔═╡ ac966cc0-32de-11eb-112e-a338edf7e848
θ = Flux.params(model)

# ╔═╡ 36c55550-32df-11eb-32b3-b157cfe2f2fe
θ[4]

# ╔═╡ 478a64c0-32df-11eb-37f3-9b511ee8a4aa
first_pred = model(X_train[:, 1])

# ╔═╡ 09ad7d80-32e0-11eb-34bc-1d69e0baa26e
argmax(first_pred)

# ╔═╡ db59ca50-32e0-11eb-1629-796d292753fe
loss(x, y) = Flux.crossentropy(model(x), y)

# ╔═╡ 068a00f2-32e1-11eb-06e7-83303b3b9169
loss(X_train, Y_train)

# ╔═╡ 7e763abe-32e1-11eb-019d-0381a4c47da5
optimizer = ADAM()

# ╔═╡ beb22b30-32e1-11eb-1661-e79f7ed1a957
Flux.train!(loss, θ, [(X_train, Y_train)], optimizer)

# ╔═╡ 226627d0-32e2-11eb-1b76-9b2e04139729
θ[4]

# ╔═╡ 30b16830-32e3-11eb-2683-15b88d2ba322
function train(numEpochs = 20)
	with_terminal() do 
		@epochs numEpochs Flux.train!(loss, θ, [(X_train, Y_train)], optimizer; cb = () -> println(loss(X_train[:, 1:1000], Y_train[:, 1:1000])))
	end
end

# ╔═╡ c6b88ac2-32e3-11eb-0951-d165b5cfa955
train(500)

# ╔═╡ d2db3b40-32e3-11eb-139b-5b1627cc342d
θ[4]

# ╔═╡ 2e426e90-32e4-11eb-0671-1973ab64e61d
accuracy(x, y) = mean(Flux.onecold(model(x)) .== y)

# ╔═╡ 51773030-32e4-11eb-092c-b55661f53ffa
accuracy(X_train, y_train)

# ╔═╡ 701a0d9e-32e4-11eb-2e7e-0f31f79ec502
accuracy(X_test, y_test)

# ╔═╡ Cell order:
# ╠═a0306e80-32da-11eb-2c41-558c63efaf94
# ╠═488451f0-32db-11eb-3fe5-f159303cd746
# ╠═372f1b00-32dc-11eb-2310-1d3833ff1ca5
# ╠═4df86350-32dc-11eb-0daa-276675a3d961
# ╠═576569b0-32dc-11eb-3a0e-655b24db92b7
# ╠═a415d420-32dc-11eb-3be9-abc0226d1726
# ╠═ab949f10-32dc-11eb-2428-a182fa327d16
# ╠═b5e1a530-32dc-11eb-3040-675be8aa4231
# ╠═0a97e4e0-32dd-11eb-0c82-b1bd36e26c2e
# ╠═1bb46322-32dd-11eb-1be8-f1e0819530e9
# ╠═842db000-32dd-11eb-2554-13be22ac66f5
# ╠═dc29d670-32de-11eb-3afd-81eaeb17b026
# ╠═a7cb1e80-32dd-11eb-2490-a373bd6e3537
# ╠═ac966cc0-32de-11eb-112e-a338edf7e848
# ╠═36c55550-32df-11eb-32b3-b157cfe2f2fe
# ╠═478a64c0-32df-11eb-37f3-9b511ee8a4aa
# ╠═09ad7d80-32e0-11eb-34bc-1d69e0baa26e
# ╠═db59ca50-32e0-11eb-1629-796d292753fe
# ╠═068a00f2-32e1-11eb-06e7-83303b3b9169
# ╠═7e763abe-32e1-11eb-019d-0381a4c47da5
# ╠═beb22b30-32e1-11eb-1661-e79f7ed1a957
# ╠═226627d0-32e2-11eb-1b76-9b2e04139729
# ╠═67608ab0-32e2-11eb-33f2-31f1de17f966
# ╠═6ac93bb0-32e3-11eb-0113-d757fbc576cf
# ╠═30b16830-32e3-11eb-2683-15b88d2ba322
# ╠═c6b88ac2-32e3-11eb-0951-d165b5cfa955
# ╠═d2db3b40-32e3-11eb-139b-5b1627cc342d
# ╠═97375410-32e4-11eb-1bcd-11d2601fcc44
# ╠═2e426e90-32e4-11eb-0671-1973ab64e61d
# ╠═51773030-32e4-11eb-092c-b55661f53ffa
# ╠═701a0d9e-32e4-11eb-2e7e-0f31f79ec502
