using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef

function gaussianDensity(X)
	(n,d) = size(X)

	mu = (1/n)sum(X,dims=1)'
	Xc = X - repeat(mu',n)
	Sigma = (1/n)*(Xc'Xc)
	SigmaInv = Sigma^-1

	function PDF(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)

		logZ = (d/2)log(2pi) + (1/2)logdet(Sigma)  
		for i in 1:t
			xc = Xhat[i,:] - mu
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[i] = exp(loglik)
		end
		return PDFs
	end

	return DensityModel(PDF)
end

function weightedGaussianDensity(X, r)
	(n,d) = size(X)
	# MAP estimation
	lambda = 1e-4

	mu = (1/sum(r))sum(X .* r,dims=1)'
	Xc = X - repeat(mu',n)
	Sigma = (1/sum(r))*((Xc .* r)'Xc) + lambda * Matrix{Float64}(I, d, d)
	SigmaInv = Sigma^-1

	function PDF(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)

		logZ = (d/2)log(2pi) + (1/2)logdet(Sigma)  
		for i in 1:t
			xc = Xhat[i,:] - mu
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[i] = exp(loglik)
		end
		return PDFs
	end

	return DensityModel(PDF)
end


