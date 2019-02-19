include("misc.jl") # Includes mode function and GenericModel typedef
include("gaussianDensity.jl")

function gda_predict(Xhat,X,y)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = maximum(y)

  # calculate MLP for p(y)
  theta = zeros(k)
  for c = 1:k
    theta[c] = sum(y .== c) / n
  end

  # fit k submodels
  subModel = Array{DensityModel}(undef,k)
  for c in 1:k
    subModel[c] = gaussianDensity(X[y .== c, :])
  end

  pdf_mat = zeros(t, k)
  for c in 1:k
    pdf_mat[:,c] = log.(subModel[c].pdf(Xhat)) .+ log(theta[c])
  end

  yhat = zeros(t)  
  for i in 1:t
    yhat[i] = argmax(pdf_mat[i, :])
  end 

  return yhat
end

function gda(X,y)
	# Implementation of GDA classifier
  predict(Xhat) = gda_predict(Xhat,X,y)
  return GenericModel(predict)
end
