### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ b1413310-3863-11eb-2331-676f97540587
using PlutoUI

# ╔═╡ 33f72efe-3862-11eb-3f05-cda01a68d63a
months = rand(1:12, 20)

# ╔═╡ 8bad6840-3862-11eb-219a-f5a99698b042
count = [sum(months .== i) for i = 1:12]

# ╔═╡ dc2736c0-3862-11eb-1be0-97c3a50e7e68
sum(count .== 2) == 4 && sum(count .== 3) == 4

# ╔═╡ 60a4c162-3863-11eb-1d94-25d0e939c161
function birthday(times::Int)
	res = 0
	for t = 1:times
		months = months = rand(1:12, 20)
		count = [sum(months .== i) for i = 1:12]
		res += sum(count .== 2) == 4 && sum(count .== 3) == 4
	end
	res / times
end

# ╔═╡ a814b822-3863-11eb-2c02-83629c2dd189
birthday(1000000)

# ╔═╡ 04cf6d80-3864-11eb-0b2e-0f22b00cfc75
with_terminal() do
	@time birthday(1000000)
end

# ╔═╡ 176f094e-3864-11eb-0e0c-cfcf2c3a4f42


# ╔═╡ Cell order:
# ╠═33f72efe-3862-11eb-3f05-cda01a68d63a
# ╠═8bad6840-3862-11eb-219a-f5a99698b042
# ╠═dc2736c0-3862-11eb-1be0-97c3a50e7e68
# ╠═60a4c162-3863-11eb-1d94-25d0e939c161
# ╠═a814b822-3863-11eb-2c02-83629c2dd189
# ╠═b1413310-3863-11eb-2331-676f97540587
# ╠═04cf6d80-3864-11eb-0b2e-0f22b00cfc75
# ╠═176f094e-3864-11eb-0e0c-cfcf2c3a4f42
