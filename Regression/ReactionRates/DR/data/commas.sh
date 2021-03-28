for f in *.txt; 
  do 
	  sed -e "s/ /,/g" < "$f" > "${f}.comma"; 
  done
