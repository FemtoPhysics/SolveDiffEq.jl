module SolveDiffEq

function stepRungeKutta(yₙ::Real, tₙ::Real, h::Real, f::Function)
    h2 = h / 2
    h6 = h / 6

    k₁ = f(tₙ,      yₙ)
    k₂ = f(tₙ + h2, yₙ + h2 * k₁)
    k₃ = f(tₙ + h2, yₙ + h2 * k₂)
    k₄ = f(tₙ + h,  yₙ + h  * k₃)

    return yₙ + h6 * (k₁ + 2.0 * k₂ + 2.0 * k₃ + k₄)
end

end # module SolveDiffEq
