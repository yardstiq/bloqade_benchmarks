@option struct PyProject
    deps::Dict{String, String}
end

@option struct JuliaProject
    path::String
    deps::Dict{String, String}
    compat::Dict{String, String}
end

function instanitiate(proj::JuliaProject)
    Pkg.activate(proj.path)
    Pkg.instanitiate()
    return
end

function instanitiate(proj::PyProject)
end
