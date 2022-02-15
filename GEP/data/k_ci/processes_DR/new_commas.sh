for f in *.csv; 
  do
    #filename=$(basename -- "$f")
    #extension="${f##*.}"
    #filename="${f%.*}"
    #echo $filename $extension
    sed -e "s/,/ /g" < "$f" > "${f}.csv_";
  done

for f in *.csv_; 
  do
    filename=$(basename -- "$f")
    extension="${f##*.}"
    filename="${f%.*}"
    echo $filename $extension
    mv "$filename" > "${filename}.csv";
  done
