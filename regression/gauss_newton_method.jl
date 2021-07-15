using LinearAlgebra

"""
Calculates the jacobian matrix
"""
function genjacobian(fun, X, w)
    n = size(w)[1]
    J = zeros(size(X)[1], n)
    h = zeros(n)
    for i in 1:n
        h[i] = 10^-5
        J[:, i] = (fun(w+h, X) - fun(w-h, X)) / h[i]
        h[i] = 0
    end
    return J
end


"""
implements the gauss newton method.
fit's a non-linear model (fun) to data points X, Y.

...
# Arguments
- `fun::function`: the model, form: fun(w, X), where w are the weights
- `X::Array/Vector`: X data.
- `Y::Array/Vector`: Y data.
- `b0::Array/Vector`: initial guess.
...
"""
function gaussnewton(fun, X, Y, w0)
    r(w, X) = fun(w, X) - Y             # Anonymous function for the residual (error)
    f_grad(r_jac, r) = 2 * r_jac' * r   # Anonymous function for the gradient of the residual

    alpha = 1                           # Static Step width, alternatively use Arijo-Goldstein or Wolfe-Powell line search
    r_jac = genjacobian(fun, w0, X)
    rw = r(w0, X)                       # The actual residual 
    w = w0
    iter = 0

    # if gradient is app. 0 -> local minima reached
    while norm(f_grad(r_jac, rw)) > 1^-8

        # calculate step direction d
        C = cholesky(r_jac' * r_jac)
        z = C.L \ (r_jac' * rw)
        d = C.U \ z

        # update weights w
        w = w - alpha * d
        r_jac = genjacobian(fun, w, X)
        rw = r(w, X)

        iter = iter + 1
        if iter == 10000
            break
        end
    end
    return w
end

