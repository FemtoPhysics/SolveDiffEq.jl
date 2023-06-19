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

end # module SolveDiffEq
