rm -r logs/sin1d3-1000
rm -r logs/sin1d3-1000-alt-l
rm -r logs/sin1d3-1000-joint
rm -r logs/sin1d3-1000-alt-t2

#scp -r yhgu@10.26.2.6:approximate-nn/logs/sin1d3-1000 logs/sin1d3-1000
scp -r yhgu@10.26.2.6:approximate-nn/logs/sin1d3-1000-alt-t2 logs/sin1d3-1000-alt-t2
#scp -r yhgu@10.26.2.6:approximate-nn/logs/sin1d3-1000-joint-reinit logs/sin1d3-1000-joint-reinit

