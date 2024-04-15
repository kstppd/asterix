module VDFOctreeApprox
#using Pkg; 
#Pkg.activate(".")
#Pkg.instantiate()

using RegionTrees

using StaticArrays: SVector, SArray
using Profile
using PaddedViews

FloatType=Float32

module PolyBases

using Symbolics
using SpecialPolynomials
using StaticArrays: SVector
@variables x y z

FloatType=Float64
DEBUG=false
polyord = 2

function makebases(T,N)
  basispol = Legendre{T}
  [basis(basispol,j)(x)*basis(basispol,k)(y)*basis(basispol,l)(z) for j in 0:N, k in 0:N, l in 0:N][:].|>Symbolics.toexpr
end

function cosbases(N)
  [cos(j*pi*x)*cos(k*pi*y)*cos(l*pi*z) for j in 0:N, k in 0:N, l in 0:N][:].|>Symbolics.toexpr
end


#bases = makebases(Rational{Int64},polyord)
basesf = makebases(FloatType,polyord)
#basesf=cosbases(polyord)

basislen = length(basesf)

eval(Expr(:function,:(phi(x,y,z)), quote SVector($(basesf...),) end))

bodystring = foldl(*,("phiwrk[$(i)] = "*string(Meta.parse(string(basesf[i])))*"\n" for i in axes(basesf,1)))

eval(Meta.parse(
    "function phi!(x::FloatType,y::FloatType,z::FloatType,phiwrk::Vector{FloatType})\n"*
    bodystring*
    "nothing\n"*
    "end\n"))

if DEBUG
  println(bodystring)
end
end



using .PolyBases

phifun3=PolyBases.phi

eval(Meta.parse(
"function phifun3!(x::T,y::T,z::T,phiwrk::Vector{T}) where {T}\n"*
    foldl(*,("phiwrk[$(i)] = "*string(Meta.parse(string(PolyBases.basesf[i])))*"\n" for i in axes(PolyBases.basesf,1)))*
    "nothing\n"*
    "end\n"))

function getdims_gen(solu::Cell{T,L,S,M}, kuva, root::Cell{T,L,S,M}) where {T,L,S,M}
  r0 = 1 .+ (solu.boundary.origin./root.boundary.widths) .* size(kuva) .|> ceil.|> Int64
  r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)./(root.boundary.widths)) .|> ceil .|> Int64 
  spans = [r0[k]:r1[k] for k in axes(r0,1)]
  h = solu.boundary.widths ./ length.(spans)
  spans, h
end

# TODO: these assume unit cuboid solu
function getdims(solu::Cell{T,2,S,M}, kuva) where {T,S,M}
    r0 = 1 .+ solu.boundary.origin .* size(kuva) .|> ceil.|> Int64
    r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)) .|> ceil .|> Int64
    spans = SVector(r0[1]:r1[1], r0[2]:r1[2])
    #[r0[k]:r1[k] for k in 1:length(r0)]
    h = solu.boundary.widths ./ length.(spans)
    spans, h
end


# TODO: these assume unit cuboid solu
function getdims(solu::Cell{T,3,S,M}, kuva) where {T,S,M}
  r0 = 1 .+ solu.boundary.origin .* size(kuva) .|> ceil.|> Int64
  r1 = ((solu.boundary.origin+solu.boundary.widths).*size(kuva)) .|> ceil .|> Int64
  #spans = [r0[k]:r1[k] for k in 1:length(r0)]
  spans = SVector(r0[1]:r1[1], r0[2]:r1[2], r0[3]:r1[3])
  h = solu.boundary.widths ./ length.(spans)
  spans, h
end



function approx(solu::Cell{T,2,S,M}, kuva) where {T,S,M}
  center = 0.5
  dims = getdims(solu,kuva)
  h = prod(dims[2])
  phi0 = phifun(0.0,0.0)
  A = zeros(size(phi0,1), size(phi0,1))
  b = zeros(size(phi0))
  bias = first.(dims[1]) .-1
  scale = 1. ./ length.(dims[1]) 
  for m in dims[1][1]
    x = scale[1]*(m-center-bias[1])#(m - bias[1] - 0.5) * dims[2][1]
    #display(x)
    for n in dims[1][2]
      y = scale[2]*(n-center-bias[2])#(n - bias[2] - 0.5) * dims[2][2]
      phi = phifun(x,y)
      b = b + phi * FloatType(kuva[m,n])*h
      A = A + phi * phi' * h # TODO calculate this exactly
    end
  end
  #A = h*fmat/scale[1]/scale[2]/4
  A,b
end

function approx!(solu::Cell{T,3,S,M}, kuva, A, b, phiwrk) where {T,S,M}
  e = one(FloatType)
  center = FloatType(0.5)
  spans, _ = getdims(solu,kuva)
  bias = first.(spans) .-1
  scale = e ./ length.(spans) 
  A .= zero(A[1])
  b .= zero(b[1])
  for k in spans[3]
    z = 2*scale[3]*(k-center-bias[3]) - e

    for n in spans[2]
      y = 2*scale[2]*(n-center-bias[2])  - e

      for m in spans[1] 
        x = 2*scale[1]*(m-center-bias[1]) - e
        phifun3!(x,y,z,phiwrk)
        for F in axes(b,1)
          b[F] = b[F] + phiwrk[F]*kuva[m,n,k]
          for Q in axes(b,1)
            A[Q,F] = A[Q,F] + phiwrk[Q]*phiwrk[F]
          end
        end
      end
    end
  end
    nothing
end

function approx(solu::Cell{T,3,S,M}, kuva)  where {T,S,M}
  center = 0.5
  dims = getdims(solu,kuva)
  h = prod(dims[2])
  phi0 = phifun3(0.0,0.0,0.0)
  phidim = size(phi0,1)
  A = zeros(phidim, phidim)
  b = zeros(phidim)
  bias = first.(dims[1]) .-1
  scale = 1. ./ length.(dims[1]) 
    for k in dims[1][3]
      z = 2*scale[3]*(k-center-bias[3]) -1.

      for n in dims[1][2]
        y = 2*scale[2]*(n-center-bias[2])-1.

        for m in dims[1][1] 
          x = 2*scale[1]*(m-center-bias[1])-1. #(m - bias[1] - 0.5) * dims[2][1]
          phi = phifun3(x,y,z)
          b = b + phi * FloatType(kuva[m,n,k])
          A = A + phi * phi'  
        end

      end
    end
  A*h,b*h
end

function calcerror!(solu::Cell{T,3,K,M}, A, b, phiwrk, kuva, reconstr; alpha=1.0, beta=1.0, nu=4) where {T,K,M}
  e = one(FloatType)
  center = FloatType(0.5)
  # c = lsqr(A,b; maxiter=length(b))
  c = A\b
  span, hs = getdims(solu,kuva)
  h = prod(hs)

  bias = first.(span) .- 1
  scale = e ./ length.(span) 
  err = zero(FloatType)
  hsq = sqrt(h)

  # Threads.@threads 
  for k in span[3]
    rk = k-bias[3]
    z = 2*scale[3]*(k-center-bias[3]) - e #(n - bias[2] - 0.5) * dims[2][2]

    for n in span[2]
      rn = n-bias[2]
      y = 2*scale[2]* (n - center -bias[2]) - e

      for m in span[1]
        rm = m-bias[1]
        x = 2*scale[1] * (m - center - bias[1]) - e

        phifun3!(x,y,z, phiwrk)
        reconstr[rm,rn,rk] = c'*phiwrk

        cterm = kuva[m,n,k]-reconstr[rm, rn, rk]

        err = err + alpha*h*(cterm^nu)

        # Do backward diff since the data should exist, skip boundary planes rk, rn, or rm = 0
        if rk>1 err = err + beta * hsq * ((cterm-kuva[m,n,k-1]+reconstr[rm,rn,rk-1])^nu) end
        if rn>1 err = err + beta * hsq * ((cterm-kuva[m,n-1,k]+reconstr[rm,rn-1,rk])^nu) end
        if rm>1 err = err + beta * hsq * ((cterm-kuva[m-1,n,k]+reconstr[rm-1,rn,rk])^nu) end

      end
    end
  end

  err = err

  FloatType(err) 
end

function calcerror(solu::Cell{T,3,K,M}, A, b, kuva;alpha=1.0, beta=1000.0, nu=4) where {T,K,M}
  center = 0.5
  # c = zeros(size(b))
  # Awrk = zeros(size(A))
  # bwrk = zeros(size(b))
  # Awrk .= A
  # bwrk .= b
  # lsqr!(c, Awrk, bwrk; maxiter=size(A,2))
  c = A\b
  dims = getdims(solu,kuva)
  h = prod(dims[2])

  reconstr = zeros((length.(dims[1]))...)
  bias = first.(dims[1]) .-1
  scale = 1. ./ length.(dims[1]) 
  #err = zero(FloatType)
  #energy = zero(FloatType)

  # Threads.@threads 
  for k in dims[1][3]
    z = 2*scale[3]*(k-center-bias[3])-1#(n - bias[2] - 0.5) * dims[2][2]
    for n in dims[1][2]
      y = 2*scale[2]* (n - center -bias[2]) - 1
      for m in dims[1][1]
        x = 2*scale[1] * (m - center - bias[1]) -1
        val = phifun3(x,y,z)
        reconstr[m-bias[1],n-bias[2],k-bias[3]] = c'*val
      end
      #err = err + h*(kuva[m,n]-reconstr[m-bias[1],n-bias[2]])^2
      #energy = h*(0.1 + kuva[m,n])^2
    end
  end

  S = kuva[dims[1]...] .- reconstr
  #R = (kuva[dims[1]...] .+ reconstr .+ 1e-6)./2
  err = (alpha*h * sum(S.^nu) .+
  beta*sqrt(h)*sum(diff(S,dims=1).^nu) + sum(diff(S,dims=2).^nu) + sum(diff(S, dims=3).^nu))^(1.0/nu)

  #energy = h*sum((0.1 .+ kuva[dims[1]...]).^2)

  reconstr, FloatType(err) #/ sqrt(FloatType(energy))
end

function calcerror(solu::Cell{T,2,L,M}, A, b, kuva;alpha=1.0, beta=1000.0, nu=4) where {T,L,M}
  center = 0.5
    c = A \ b
  
    dims = getdims(solu,kuva)
    h = prod(dims[2])

    reconstr = zeros((length.(dims[1]))...)
    bias = first.(dims[1]) .-1
    scale = 1. ./ length.(dims[1]) 
    #err = zero(FloatType)
    #energy = zero(FloatType)

    for m in dims[1][1]
      x = scale[1] * (m - center - bias[1]) 
      for n in dims[1][2]
        y = scale[2]* (n - center -bias[2]) 
        val = phifun(x,y)
        reconstr[m-bias[1],n-bias[2]] = c'*val
        #err = err + h*(kuva[m,n]-reconstr[m-bias[1],n-bias[2]])^2
        #energy = h*(0.1 + kuva[m,n])^2
      end
    end

    S = kuva[dims[1]...] .- reconstr
    err = (alpha*h * sum((kuva[dims[1]...] .- reconstr).^nu) +
    beta*sqrt(h)*sum(diff(S,dims=1).^nu) + sum(diff(S,dims=2).^nu))^(1.0/nu)

    #energy = h*sum((0.1 .+ kuva[dims[1]...]).^2)

    reconstr, FloatType(err) #/ sqrt(FloatType(energy))
end

function approx_and_recon(solu,kuva; alpha=1.0, beta=1.0, nu=4)
  A, b = approx(solu,kuva)
  
  reconstr, err = calcerror(solu, A, b, kuva; alpha=alpha, beta=beta, nu=nu)
end

function approx_and_recon!(solu,kuva, A, b, phiwrk;reconstr=nothing, alpha=1.0, beta=1.0, nu=4)
  approx!(solu,kuva, A, b, phiwrk)

  reco = if isnothing(reconstr) 
    span, _ = getdims(solu,kuva)
    similar(kuva[span...])
  else
    span, _ = getdims(solu,kuva)
    @view reconstr[span...]
  end
  calcerror!(solu, A, b, phiwrk, kuva, reco; alpha=alpha, beta=beta, nu=nu)
end

function improveold!(solu::Cell{T,N,S,M}, uudelleen, kuva) where {T,N,S,M}
  rec_new, err = approx_and_recon(solu, kuva)
  rec_new = rec_new
  dims = getdims(solu,kuva)
  uudelleen[dims[1]...] = rec_new
  err
end

function improve!(solu::Cell{T,N,S,M}, reconstr, kuva, A, b, phiwrk) where {T,N,S,M}
  #span,_ = getdims(solu, kuva)
  #reco = @view reconstr[span...]

  err = approx_and_recon!(solu, kuva, A, b, phiwrk; reconstr=reconstr)
  err
end

"""
W is the original padded size of the image (padded to 3*2^k)
"""
function getdims_new(solu::Cell{T,3,S,M}, W) where {T,S,M}
  r0 = 1 .+ solu.boundary.origin .* W .|> ceil .|> Int64
  r1 = ((solu.boundary.origin+solu.boundary.widths) .* W) .|> ceil .|> Int64
  SVector(r0[1]:r1[1], r0[2]:r1[2], r0[3]:r1[3])
end

function approx_new!(solu::Cell{T,3,S,M}, kuva, A, b, phiwrk) where {T,S,M}
  center = 0.5
  spans, _ = getdims(solu,kuva)
  bias = first.(spans) .-1
  scale = 1. ./ length.(spans) 
  for k in spans[3]
    z = 2*scale[3]*(k-center-bias[3]) -1.

    for n in spans[2]
      y = 2*scale[2]*(n-center-bias[2]) -1.

      for m in spans[1] 
        x = 2*scale[1]*(m-center-bias[1]) -1.
        phifun3!(x,y,z,phiwrk)
        for F in axes(b,1)
          b[F] = b[F] + phiwrk[F]*kuva[m,n,k]
          for Q in axes(b,1)
            A[Q,F] = A[Q,F] + phiwrk[Q]*phiwrk[F]
          end
        end
      end
    end
  end
    nothing
end


function compress(img; maxiter=100, nu=4, tol=1e-4, alpha=1.0, beta=1.0, verbose=True)
  bdim = size(PolyBases.basesf,1)
  A = zeros(FloatType, bdim, bdim)
  b = zeros(FloatType, bdim)
  phiwrk = similar(b)

  L = round(bdim^(1/3))|>Int64
  spans = size(img)
  k = ceil(log2(maximum(spans) ./ L))
  W = (PolyBases.polyord+1)*2^k 


  cell = Cell( SVector(0., 0., 0.), SVector(1., 1., 1.), 0.0 )
  scale = maximum(img[:])
  img = Array{FloatType}(img ./ scale)
  pad_span = getdims_new(cell, W)
  padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3]))

  span, _ = getdims(cell,padimg)
  reconstr=zeros(span...)

  initial_error = 0.0
  for iter in 1:maxiter
    tot_err = zero(FloatType)
    maxerr = zero(FloatType)
    for leaf in allleaves(cell)
      approx!(leaf, padimg, A, b, phiwrk)
      span, _ = getdims(leaf, padimg)
      
      err = calcerror!(leaf, A, b, phiwrk, padimg, @view reconstr[span...]; alpha=alpha, beta=beta, nu=nu)
      leaf.data = err
      tot_err = tot_err + err
      maxerr = if err > maxerr err else maxerr end

      if span[1] == L leaf.data = -1.0 end

    end
    if iter == 1 initial_error = tot_err end

    if verbose
      print((tot_err/initial_error).^(1. /nu))
      print("\t")
      println(maxerr.^(1. /nu))
    end

    for leaf in allleaves(cell)
      if leaf.data == maxerr 
        split!(leaf) 
      end
    end
    if (tot_err/initial_error).^(1. /nu) < tol
      break
    end
  end

  A, b, scale .* img, scale .* reconstr[(1:spans[j] for j in 1:3)...], cell, 
    map(collect(allleaves(cell))) do leaf
      approx!(leaf, padimg, A,b, phiwrk)
      spans, _ = getdims(leaf, padimg)
    (leaf, A\b, spans)
    end

end

function testapprox(spans::Tuple{Int64,Int64,Int64}; fillfun=nothing, maxiter=5, nu=4, tol=1e-3, alpha=1.0, beta=1.0)

  bdim = size(PolyBases.basesf,1)
  A = zeros(bdim, bdim)
  b = zeros(bdim)
  phiwrk = similar(b)

  L = round(bdim^(1/3))|>Int64
  k = ceil(log2(maximum(spans) ./ L))
  W = 3*2^k 


  cell = Cell( SVector(0., 0., 0.), SVector(1., 1., 1.), 0.0 )

  pad_span = getdims_new(cell, W)

  img = zeros(spans)
  if !isnothing(fillfun)
    for J in eachindex(view(img,(1:spans[j] for j in 1:3)...))
      img[J] = fillfun(J[1], J[2], J[3])
    end
  end
  padimg = PaddedView(zero(img[1]), img, (pad_span[1], pad_span[2], pad_span[3]))

  span, _ = getdims(cell,padimg)
  reconstr=zeros(span...)

  initial_error = 0.0
  for iter in 1:maxiter
    tot_err = zero(FloatType)
    maxerr = zero(FloatType)
    for leaf in allleaves(cell)
      approx!(leaf, padimg, A, b, phiwrk)
      span, _ = getdims(leaf, padimg)
      
      err = calcerror!(leaf, A, b, phiwrk, padimg, @view reconstr[span...]; alpha=alpha, beta=beta, nu=nu)
      leaf.data = err
      tot_err = tot_err + err
      maxerr = if err > maxerr err else maxerr end

      if span[1] == L leaf.data = -1.0 end

    end
    if iter == 1 initial_error = tot_err end


    print((tot_err/initial_error).^(1. /nu))
    print("\t")
    println(maxerr.^(1. /nu))

    for leaf in allleaves(cell)
      if leaf.data == maxerr 
        split!(leaf) 
      end
    end
    if (tot_err/initial_error).^(1. /nu) < tol
      break
    end
  end


  A, b, img, reconstr[(1:spans[j] for j in 1:3)...], cell, 
    map(collect(allleaves(cell))) do leaf
      approx!(leaf, padimg, A,b, phiwrk)
      spans, _ = getdim(leaf, padimg)
    (leaf, A\b, spans)
    end
  
  #reco = reconstr[(1:spans[j] for j in 1:3)...]
end

function visualcompare(reconstr, img; slice=5, epsterm=1e-3)

  fig = Makie.Figure()

  ax, imleft = heatmap(fig[1,1], reconstr[:,:,slice], interpolate=false)
  ax, imright=heatmap(fig[1,2], img[:,:,slice], interpolate=false)
  ax, imrightright=heatmap(fig[1,3], (img[:,:,slice]-reconstr[:,:,slice])./(img[:,:,slice].+reconstr[:,:,slice].+epsterm), interpolate=false)
  Colorbar(fig[1,1][1,2], imleft)
  Colorbar(fig[1,2][1,2], imright)
  Colorbar(fig[1,3][1,2], imrightright)

  fig

end

end


if false
  using .VDFOctreeApprox
  N=100
  include("loadvlasi.jl")
  vdf_image = vdf_image 
  img = vdf_image3

  #A,b,img,reco, cell, tree= testapprox(size(img); fillfun=(i,j,k)->img[i,j,k], alpha=1.0, beta=1.0, nu=12, tol=1e-8);
  A,b,img,reco, cell, tree = VDFOctreeApprox.compress(img;maxiter=100, alpha=1.0, beta=1.0, nu=12, tol=1e-3)

  let treesize = length(tree)*length(b), imgsize = size(img)|>prod
    println(treesize)
    println(imgsize)
    println(treesize / imgsize)
  end
end
