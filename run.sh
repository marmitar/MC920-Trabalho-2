#!/usr/bin/sh

./run.sh

# Section 2.2
python main.py imagens/baboon.png -g -d ninke -o resultados/execucao.png
# Section 4.1
mkdir -p resultados/var
for image in baboon peppers monalisa watch; do
    cp build/floyd/unidirecional/$image.png resultados/var/unidir_$image.png
    cp build/floyd/alternada/$image.png resultados/var/alternada_$image.png
done
# Section 4.2
mkdir -p resultados/dists
for image in baboon peppers monalisa watch; do
    for file in $(ls build/alternada/$image); do
        cp build/alternada/$image/$file resultados/dists/${image}_$file
    done
done
# Section 4.3
mkdir -p resultados/extra
for image in peppers watch; do
    for dist in unidirecional alternada espiral hilbert; do
        cp build/floyd/$dist/$image.png resultados/extra/${image}_${dist}.png
    done
done
