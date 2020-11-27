for file in *.m; do
    mv "$file" "$(basename "$file" .m).jl"
done
#
for f in *.m; do
    mv -- "$f" "${f%.m}.jl"
done
#
for f in *.m; do
    mv -- "$f" "$(basename -- "$f" .m).jl"
done
