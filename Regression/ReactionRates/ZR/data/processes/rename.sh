find . -depth -name "*[ ()]*" -execdir rename 's/[ )]/_/g; s/\(//g' "{}" \;

