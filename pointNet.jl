using Flux, Metalhead, Statistics, LinearAlgebra, Zygote
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition

# # for gpu
# using CUDAapi, CuArrays, CUDAdrv, CUDAnative
# if has_cuda()
#     @info "CUDA is on"
#     CuArrays.allowscalar(false)
# end

STNkd(k::Int) = Chain(
    x -> reshape(x, :, 1, k, size(x, 3)),
    Conv((1, 1), k=>64, relu),
    BatchNorm(64),
    Conv((1, 1), 64=>128, relu),
    BatchNorm(128),
    Conv((1, 1), 128=>1024, relu),
    BatchNorm(1024),
    x -> reshape(maximum(reshape(x, :, 1024, size(x,4)), dims=1), :, size(x, 4)),
    Dense(1024, 512, relu),
    BatchNorm(512),
    Dense(512, 256, relu),
    BatchNorm(256),
    Dense(256,k*k),
    x -> reshape(x, k, k, size(x, 2)),
    x -> x .+ reshape(Matrix{Float64}(I, k, k), k, k, 1)
)

PointNet(k::Int) = Chain(
    x -> batched_mul(x, batched_transpose(stn(x))),
    x -> reshape(x, :, 1, size(x, 2), size(x, 3)),
    Conv((1,1), 3=>64),
    BatchNorm(64),
    x -> reshape(x, :, size(x, 3), size(x, 4)),
    x -> batched_mul(x, batched_transpose(fstn(x))),
    x -> reshape(x, :, 1, size(x, 2), size(x, 3)),
    Conv((1,1), 64=>128, relu),
    BatchNorm(128),
    Conv((1,1), 128=>1024),
    BatchNorm(1024),
    x -> reshape(x, :, size(x, 3), size(x, 4)),
    x -> reshape(maximum(x, dims=1), 1024, size(x, 3)),
    Dense(1024, 512, relu),
    BatchNorm(512),
    Dense(512, 256, relu),
    Dropout(0.4),
    BatchNorm(256),
    Dense(256, k, relu),
    softmax
)

function pcNormalize(pc)
    centroid = mean((x->x), pc, dims = 1)
    pc = pc .- centroid
    m = max((sum(pc.^2, dims=2).^.5)...)
    pc = pc ./ m
    return pc
end

function datapoint(fn)
    #fn = datapath[index]
    cls = [classes[fn[1]]]
    Pointset = Array{Float64}(undef, npoints, 3)
    stream = open(fn[2], "r")
    for i in 1:npoints
        tmp = readline(stream, keep=false)
        Pointset[i, :] = map((x->parse(Float64, x)), split(tmp, ",")[1:3])
    end
    Pointset = pcNormalize(Pointset)
    return (Pointset,cls)
end

root = "./data/"
batch_size = 4
num_classes = 10 #possible values {10,40}
npoints = 1024
catFile = joinpath(root, "modelnet$(num_classes)_shape_names.txt")
categories = [line::String for line in readlines(catFile)]

classes = Dict{String, Int}()
for i in 1:length(categories)
    classes[categories[i]] = i
end

shapeids = Dict()
shape_names = Dict()
datapath = Dict()

shapeids["train"] = [line::String for line in readlines(joinpath(root, "modelnet$(num_classes)_train.txt"))]
shapeids["test"] = [line::String for line in readlines(joinpath(root, "modelnet$(num_classes)_test.txt"))]

split_ = "train"
shape_names[split_] = [join(split(shapeids[split_][i], "_")[1:end-1], "_") for i in 1:length(shapeids[split_])]
datapath[split_] = [(shape_names[split_][i], joinpath(root, shape_names[split_][i], (shapeids[split_][i])*".txt")) for i in 1:length(shapeids[split_])]

split_ = "test"
shape_names[split_] = [join(split(shapeids[split_][i], "_")[1:end-1], "_") for i in 1:length(shapeids[split_])]
datapath[split_] = [(shape_names[split_][i], joinpath(root, shape_names[split_][i], (shapeids[split_][i])*".txt")) for i in 1:length(shapeids[split_])]

# input: width*height*channel*minibatch

# Fetching the train and validation data and getting them into proper shape
X = [datapoint(p) for p in datapath["train"]]
data = [X[i][1] for i in 1:length(X)]
labels = onehotbatch(cat([X[i][2] for i in 1:length(X)]..., dims=1), 1:num_classes)
train = [(cat(data[i]..., dims = 3), labels[:,i]) for i in partition(1:length(data), batch_size)]

VAL = [datapoint(p) for p in datapath["test"]]
valX = cat([VAL[i][1] for i in 1:length(VAL)]..., dims=3)
valY = onehotbatch(cat([VAL[i][2] for i in 1:length(VAL)]..., dims=1), 1:num_classes)

# Defining the loss and accuracy functions

stn = STNkd(3)
fstn = STNkd(64)
m = PointNet(num_classes)

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(cpu(m(x)), 1:num_classes) .== onecold(cpu(y), 1:num_classes))

# Defining the callback and the optimizer

evalcb = throttle(() -> @show(accuracy(valX, valY)), 10)

opt = ADAM()

# Starting to train models

@show("Training Start")
Flux.train!(loss, params(m), train, opt, cb = evalcb)

# Accuracy of valset of ModelNet dataset
@show(accuracy(valX, valY))

