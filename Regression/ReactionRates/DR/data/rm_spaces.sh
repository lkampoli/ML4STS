for file in *' '*; do [ -f "$file" ] && mv "$file" "`echo $file|tr -d '[:space:]'`"; done
