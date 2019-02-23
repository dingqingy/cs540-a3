include("misc.jl") # Includes mode function and GenericModel typedef
include("studentT.jl")

function tda_predict(Xhat,X,y)
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
    subModel[c] = studentT(X[y .== c, :])
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

function tda(X,y)
	# Implementation of TDA classifier
  predict(Xhat) = tda_predict(Xhat,X,y)
  return GenericModel(predict)
end