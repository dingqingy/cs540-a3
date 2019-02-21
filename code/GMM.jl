using LinearAlgebra, Random
include("misc.jl")
include("gaussianDensity.jl")

function GMM(X, k; max_iter=200)
	(n,d) = size(X)

	# random initial an cluster assignment
	init_assign = rand(1:k, n)
	r = zeros(n, k)
	for i in 1:n
		c = init_assign[i]
		r[i, c] = 1 # note r[i][c] = 1 would not work
	end
	pi = zeros(k)
	for c in 1:k
		pi[c] = sum(r[:, c]) / n
	end
	
	# init Gaussian
	subModel = Array{DensityModel}(undef,k)
	for c in 1:k
		subModel[c] = gaussianDensity(X[init_assign .== c, :])
	end

	# EM iterations
	joint = zeros(n, k)
	Q = -Inf
	for i = 1:max_iter
		# E step
		for c in 1:k
			# calculate r[:, c]
			joint[:, c] = subModel[c].pdf(X) .* pi[c]
		end
		for i in 1:n
			r[i, :] = joint[i, :] ./ sum(joint[i, :])
		end

		# M step 
		for c in 1:length(subModel)
			pi[c] = sum(r[:, c]) / n
			subModel[c] = weightedGaussianDensity(X, r[:, c])
		end
		# check convergence
		# evaluate Q function
		Q_old = Q
		Q = sum(r .* log.(joint))
		if abs(Q - Q_old) < 1e-8
			println("optimality obtained at iteration: ", i)
			break
		end
	end

	function PDF(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)

		for c in 1:k
			PDFs += pi[c]*subModel[c].pdf(Xhat)
		end
		return PDFs
	end

	return DensityModel(PDF)
end