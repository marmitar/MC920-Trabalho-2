#!/usr/bin/bash

prog="python3 main.py $@"
build=build

varredura=(unidirecional alternada hilbert espiral)
dists=(floyd stevenson burkes sierra stucki jarvis)

# pastas de saída
for v in ${varredura[@]}; do
    mkdir -p $build/$v
done
for d in ${dists[@]}; do
    mkdir -p $build/$d
done

proc=0
# pontilhados
for file in $(ls imagens/*.png); do
    out=$(basename $file)
    echo -n $out ...

    for v in ${varredura[@]}; do
        for d in ${dists[@]}; do
            $prog -o $build/$v/$d\_$out $file -v $v -d $d &
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
for file in $(ls imagens/*.png); do
    out=$(basename $file)
    for v in ${varredura[@]}; do
        for d in ${dists[@]}; do
            cp $build/$v/$d\_$out $build/$d/$v\_$out
        done
    done
done
echo Done!
