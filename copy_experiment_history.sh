#! /bin/sh
destdir='/home/s2125048/thesis/model/metric/8june'
for pathname in /home/s2125048/thesis/model/20200605_114555/*/*.obj; do
    cp -i $pathname $destdir/$(basename $(dirname "$pathname"))_$(basename "$pathname")
done