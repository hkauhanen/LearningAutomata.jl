using LearningAutomata
using Luxor
using Random

Random.seed!(123)


localfilename = "logo.svg"
destfilename = "../docs/src/assets/logo.svg"

res = 2.0


la1 = LRPLearner(3, 0.02)
la1.W = [0.2, 0.1, 0.7]
la2 = LRPLearner(3, 0.02)
la2.W = [0.7, 0.2, 0.1]
la3 = LRPLearner(3, 0.02)
la3.W = [0.1, 0.7, 0.2]

environment1 = [0.0, 0.9, 0.9]
environment2 = [0.9, 0.0, 0.9]
environment3 = [0.9, 0.9, 0.0]

function hist2simplex(x)
    x = TernaryPlots.tern2cart.(x[:, 3], x[:, 1], x[:, 2])

    Point.([(res * 2 * 100 .* (y[1] - 0.5), 20 - res * 105 - res * 2 * 105 .* (y[2] - 0.866)) for y in x])
end

maxiter = 500

history1 = hist2simplex(simulate!(la1, maxiter, environment1))
history2 = hist2simplex(simulate!(la2, maxiter, environment2))
history3 = hist2simplex(simulate!(la3, maxiter, environment3))


Drawing(res*300, res*300, localfilename)

origin()

setline(res*4)
setcolor("black")
ngon(Point(res*0, res*30), res*130, 3, deg2rad(270), action=:stroke)

vertices = ngon(Point(res*0, res*30), res*130, 3, deg2rad(270), vertices=true)

setline(res*4)

setcolor(Luxor.julia_purple)
circle(vertices[1], res*10, action=:fill)
move(history1[1])
[line(history1[t]) for t in 2:length(history1)]
strokepath()

setcolor(Luxor.julia_red)
move(history3[1])
[line(history3[t]) for t in 2:length(history3)]
strokepath()
circle(vertices[2], res*10, action=:fill)

setcolor(Luxor.julia_green)
move(history2[1])
[line(history2[t]) for t in 2:length(history2)]
strokepath()
circle(vertices[3], res*10, action=:fill)

finish()

preview()

cp(localfilename, destfilename, force=true)
