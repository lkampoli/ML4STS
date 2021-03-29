# usage: ./conversion.sh filename-to-convert
while IFS= read -r file; do
    while IFS= read -r sheet; do
        in2csv --sheet "$sheet" "$file" > "${file%.*}-${sheet}.csv"
    done < <(in2csv -n "$file")
done < <(find . -name '*.xls' -o -name '*.xlsx')
