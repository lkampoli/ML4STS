for dir in `ls -d [DEGHKMNRS]*`

do
   echo $dir
   cd $dir
   ./clean.sh
   rm *.txt
   pwd
   cd ..
   pwd
done
