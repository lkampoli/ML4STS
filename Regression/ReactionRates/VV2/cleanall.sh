for dir in `ls -d [DEGHKMNRS]*`

do
   echo $dir
   cd $dir
   ./clean.sh
   pwd
   cd ..
   pwd
done

