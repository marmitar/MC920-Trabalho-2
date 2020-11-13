#!/usr/bin/bash

prog="python3 main.py $@"
build=build

varredura=(unidirecional alternada hilbert espiral)
dists=(floyd stevenson burkes sierra stucki jarvis)
imagens=$(ls imagens/*.png)

# pastas de saída
for v in ${varredura[@]}; do
    for d in ${dists[@]}; do
        mkdir -p $build/$v/$d
        mkdir -p $build/grayscale/$v/$d
    done
done

proc=0
# pontilhados
for file in ${imagens[@]}; do
    out=$(basename $file)
    echo -n $out ...

    for v in ${varredura[@]}; do
        for d in ${dists[@]}; do
            $prog -o $build/$v/$d/$out $file -v $v -d $d &
            pids[${proc}]=$!
            proc=$((proc+1))
            $prog -o $build/grayscale/$v/$d/$out $file -v $v -d $d -g &
            pids[${proc}]=$!
            proc=$((proc+1))
        done
    done
    echo ' 'waiting
done

# espera processos encerrarem
for pid in ${pids[@]}; do
    wait $pid
done

# copia em outras pastas
for img in ${imagens[@]}; do
    out=$(basename $img)
    m="${out%.*}"
    for v in ${varredura[@]}; do
        for d in ${dists[@]}; do
            mkdir -p $build/$d/$v
            mkdir -p $build/$d/$m
            mkdir -p $build/$v/$m

            cp $build/$v/$d/$out $build/$d/$v/$m.png
            cp $build/$v/$d/$out $build/$d/$m/$v.png
            cp $build/$v/$d/$out $build/$v/$m/$d.png
        done
    done
done
echo Done!
