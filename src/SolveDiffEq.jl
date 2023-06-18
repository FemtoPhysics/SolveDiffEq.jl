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

const RKRalstonParams = (
    αs = (0.0, 0.4, (14 - 3 * SQRT5) / 16, 1.0),
    β2 = (0.4, 0.0, 0.0, 0.0),
    β3 = (
        3 * (476SQRT5 - 963) / 1024,
        5 * (757 - 324SQRT5) / 1024,
        0.0, 0.0
    ),
    β4 = (
        (2094SQRT5 - 3365) / 6040,
        -(3046SQRT5 + 975) / 2552,
        (203968SQRT5 + 467040) / 240845,
        0.0
    ),
    ws = (
        (22 - 3SQRT5) / (168 - 36SQRT5),
        (125SQRT5 - 250) / (180SQRT5 - 456),
        1024 / (4869SQRT5 - 10038),
        2SQRT5 / (9SQRT5 + 6)
    )
)

function stepRungeKuttaRalston(yₙ::Real, tₙ::Real, h::Real, f::Function)
    hs = ntuple(i -> RKRalstonParams.αs[i] * h, 4)

    k₁ = h * f(tₙ + hs[1], yₙ)
    k₂ = h * f(tₙ + hs[2], yₙ + RKRalstonParams.β2[1] * k₁)
    k₃ = h * f(tₙ + hs[3], yₙ + RKRalstonParams.β3[1] * k₁ + RKRalstonParams.β3[2] * k₂)
    k₄ = h * f(tₙ + hs[4], yₙ + RKRalstonParams.β4[1] * k₁ + RKRalstonParams.β4[2] * k₂ + RKRalstonParams.β4[3] * k₃)

    return yₙ + RKRalstonParams.ws[1] * k₁ + RKRalstonParams.ws[2] * k₂ + RKRalstonParams.ws[3] * k₃ + RKRalstonParams.ws[4] * k₄
end

end # module SolveDiffEq
