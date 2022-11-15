using Flux, Statistics

function and(x1, x2)
    if x1 == 1 && x2 == 1
        return 1
    else 
        return 0
    end 
end

noisy = rand(Float32, 2, 1000)                                 
truth = map(col -> and(col...), eachcol(noisy .> 0.5))           

model = Chain(Dense(2 => 3, tanh), BatchNorm(3), Dense(3 => 2), softmax)

out1 = model(noisy)                                  

mat = Flux.onehotbatch(truth, [true, false])          
data = Flux.DataLoader((noisy, mat), batchsize=64, shuffle=true);
first(data) .|> summary                                        

pars = Flux.params(model)  
opt = Flux.Adam(0.01)  

for epoch in 1:1_000
    Flux.train!(pars, data, opt) do x, y
        Flux.crossentropy(model(x), y)
    end
end

pars  
opt
out2 = model(noisy)

mean((out2[1,:] .> 0.5) .== truth) 

using Plots  

p_true = scatter(noisy[1,:], noisy[2,:], zcolor=truth, title="True classification", legend=false)
p_raw =  scatter(noisy[1,:], noisy[2,:], zcolor=out1[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter(noisy[1,:], noisy[2,:], zcolor=out2[1,:], title="Trained network", legend=false)

plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330)) 
