module SolveDiffEq

const SQRT5 = sqrt(5.0)

struct ButcherTab{N,Ta,Tb,Tc}
    mat_a::NTuple{N,NTuple{N,Ta}}
    vec_b::NTuple{N,Tb}
    vec_c::NTuple{N,Tc}
end

const RK4ClassicTab = ButcherTab(
    (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0)
    ),
    (1//6, 1//3, 1//3, 1//6),
    (0.0, 0.5, 0.5, 1.0)
)

const RK4RalstonTab = ButcherTab(
    (
        (0.0,                       0.0,                       0.0,                             0.0),
        (0.4,                       0.0,                       0.0,                             0.0),
        ((1428SQRT5 - 2889) / 1024, (3785 - 1620SQRT5) / 1024, 0.0,                             0.0),
        ((2094SQRT5 - 3365) / 6040, -(3046SQRT5 + 975) / 2552, (203968SQRT5 + 467040) / 240845, 0.0)
    ),
    (
        (22 - 3SQRT5) / (168 - 36SQRT5),
        (125SQRT5 - 250) / (180SQRT5 - 456),
        1024 / (4869SQRT5 - 10038),
        2SQRT5 / (9SQRT5 + 6)
    ),
    (0.0, 0.4, (14 - 3SQRT5) / 16, 1.0)
)

@generated function stepRungeKutta(yn::Real, tn::Real, h::Real, f::Function, tab::ButcherTab{N}) where N
    sym_list = Vector{Symbol}(undef, N)
    blk_expr = Expr(:block)
    ret_expr = Expr(:call, :+, :yn)

    push!(blk_expr.args, :(mat_a = tab.mat_a))
    push!(blk_expr.args, :(vec_b = tab.vec_b))
    push!(blk_expr.args, :(vec_c = tab.vec_c))

    for n in 1:N
        sym_list[n] = Symbol(:k, n)
        a = Expr[]
        for i in 1:n-1
            push!(a, :(mat_a[$n][$i] * $(sym_list[i])))
        end
        e = Expr(
            :call, :*, :h, Expr(
                :call, :f, :(tn + vec_c[$n] * h),
                Expr(:call, :+, :yn, a...)
            )
        )
        push!(blk_expr.args, :($(sym_list[n]) = @inbounds $e))
        push!(ret_expr.args, :(vec_b[$n] * $(sym_list[n])))
    end
    push!(blk_expr.args, ret_expr)
    return blk_expr
end

function stepRosen(yn::Real, tn::Real, h::Real, f::Function, ∂f∂y::Function, ∂f∂t::Function)
    invh = inv(h)
    invA = inv(4.0invh - ∂f∂y(tn, yn)) # 1 / γh - ∂f∂y(tn, yn), γ = 0.25
    h∂tf = h * ∂f∂t(tn, yn)

    # ------------------------------------------------------------

    a21 = +1.544

    a31 = +0.9466785280815826e+00
    a32 = +0.2557011698983284e+00

    a41 = +0.3314825187068521e+01
    a42 = +0.2896124015972201e+01
    a43 = +0.9986419139977817e+00

    a51 = +0.1221224509226641e+01
    a52 = +0.6019134481288629e+01
    a53 = +0.1253708332932087e+02
    a54 = -0.6878860361058950e+00

    # ------------------------------------------------------------

    c21 = -0.5668800000000000e+1

    c31 = -0.2430093356833875e+1
    c32 = -0.2063599157091915e+0

    c41 = -0.1073529058151375e+0
    c42 = -0.9594562251023355e+1
    c43 = -0.2047028614809616e+2

    c51 = +0.7496443313967647e+1
    c52 = -0.1024680431464352e+2
    c53 = -0.3399990352819905e+2
    c54 = +0.1170890893206160e+2

    c61 = +0.8083246795921522e+1
    c62 = -0.7981132988064893e+1
    c63 = -0.3152159432874371e+2
    c64 = +0.1631930543123136e+2
    c65 = -0.6058818238834054e+1

    # ------------------------------------------------------------

    α1 = 0.0
    α2 = 0.386
    α3 = 0.21
    α4 = 0.63
    α5 = 1.0

    # ------------------------------------------------------------

    γ1 =  0.25
    γ2 = -0.1043
    γ3 =  0.1035
    γ4 = -0.0362
    γ5 =  0.0
    γ6 =  0.0

    # ------------------------------------------------------------

    ytmp = yn
    dydt = f(tn + α1 * h, ytmp)
    u1 = invA * (dydt + γ1 * h∂tf)

    ytmp = yn + a21 * u1
    dydt = f(tn + α2 * h, ytmp)
    u2 = invA * (dydt + γ2 * h∂tf + invh * c21 * u1)

    ytmp = yn + a31 * u1 + a32 * u2
    dydt = f(tn + α3 * h, ytmp)
    u3 = invA * (dydt + γ3 * h∂tf + invh * (c31 * u1 + c32 * u2))

    ytmp = yn + a41 * u1 + a42 * u2 + a43 * u3
    dydt = f(tn + α4 * h, ytmp)
    u4 = invA * (dydt + γ4 * h∂tf + invh * (c41 * u1 + c42 * u2 + c43 * u3))

    ytmp = yn + a51 * u1 + a52 * u2 + a53 * u3 + a54 * u4
    dydt = f(tn + h, ytmp)
    u5 = invA * (dydt + γ5 * h∂tf + invh * (c51 * u1 + c52 * u2 + c53 * u3 + c54 * u4))

    ytmp += u5
    dydt = f(tn + h, ytmp)
    Δy = invA * (dydt + γ6 * h∂tf + invh * (c61 * u1 + c62 * u2 + c63 * u3 + c64 * u4 + c65 * u5))

    return ytmp + Δy, Δy
end

end # module SolveDiffEq
