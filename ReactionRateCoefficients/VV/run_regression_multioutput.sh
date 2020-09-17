#!/bin/bash

cd src

declare -a dataset_down=('VV_DOWN_N2_RATES-0->1'   \
#                         'VV_DOWN_N2_RATES-1->2'   \
#                         'VV_DOWN_N2_RATES-2->3'   \
#                         'VV_DOWN_N2_RATES-3->4'   \
#                         'VV_DOWN_N2_RATES-4->5'   \
#                         'VV_DOWN_N2_RATES-5->6'   \
#                         'VV_DOWN_N2_RATES-6->7'   \
#                         'VV_DOWN_N2_RATES-7->8'   \
#                         'VV_DOWN_N2_RATES-8->9'   \
#                         'VV_DOWN_N2_RATES-9->10'  \
#                         'VV_DOWN_N2_RATES-10->11' \
#                         'VV_DOWN_N2_RATES-11->12' \
#                         'VV_DOWN_N2_RATES-12->13' \
#                         'VV_DOWN_N2_RATES-13->14' \
#                         'VV_DOWN_N2_RATES-14->15' \
#                         'VV_DOWN_N2_RATES-15->16' \
#                         'VV_DOWN_N2_RATES-16->17' \
#                         'VV_DOWN_N2_RATES-17->18' \
#                         'VV_DOWN_N2_RATES-18->19' \
#                         'VV_DOWN_N2_RATES-19->20' \
#                         'VV_DOWN_N2_RATES-20->21' \
#                         'VV_DOWN_N2_RATES-21->22' \
#                         'VV_DOWN_N2_RATES-22->23' \
#                         'VV_DOWN_N2_RATES-23->24' \
#                         'VV_DOWN_N2_RATES-24->25' \
#                         'VV_DOWN_N2_RATES-25->26' \
#                         'VV_DOWN_N2_RATES-26->27' \
#                         'VV_DOWN_N2_RATES-27->28' \
#                         'VV_DOWN_N2_RATES-28->29' \
#                         'VV_DOWN_N2_RATES-29->30' \
#                         'VV_DOWN_N2_RATES-30->31' \
#                         'VV_DOWN_N2_RATES-31->32' \
#                         'VV_DOWN_N2_RATES-32->33' \
#                         'VV_DOWN_N2_RATES-33->34' \
#                         'VV_DOWN_N2_RATES-34->35' \
#                         'VV_DOWN_N2_RATES-35->36' \
#                         'VV_DOWN_N2_RATES-36->37' \
#                         'VV_DOWN_N2_RATES-37->38' \
#                         'VV_DOWN_N2_RATES-38->39' \
#                         'VV_DOWN_N2_RATES-39->40' \
#                         'VV_DOWN_N2_RATES-40->41' \
#                         'VV_DOWN_N2_RATES-41->42' \
#                         'VV_DOWN_N2_RATES-42->43' \
#                         'VV_DOWN_N2_RATES-43->44' \
#                         'VV_DOWN_N2_RATES-44->45' \
                         'VV_DOWN_N2_RATES-45->46' \
#                         'VV_DOWN_O2_RATES-0->1'   \
#                         'VV_DOWN_O2_RATES-1->2'   \
#                         'VV_DOWN_O2_RATES-2->3'   \
#                         'VV_DOWN_O2_RATES-3->4'   \
#                         'VV_DOWN_O2_RATES-4->5'   \
#                         'VV_DOWN_O2_RATES-5->6'   \
#                         'VV_DOWN_O2_RATES-6->7'   \
#                         'VV_DOWN_O2_RATES-7->8'   \
#                         'VV_DOWN_O2_RATES-8->9'   \
#                         'VV_DOWN_O2_RATES-9->10'  \
#                         'VV_DOWN_O2_RATES-10->11' \
#                         'VV_DOWN_O2_RATES-11->12' \
#                         'VV_DOWN_O2_RATES-12->13' \
#                         'VV_DOWN_O2_RATES-13->14' \
#                         'VV_DOWN_O2_RATES-14->15' \
#                         'VV_DOWN_O2_RATES-15->16' \
#                         'VV_DOWN_O2_RATES-16->17' \
#                         'VV_DOWN_O2_RATES-17->18' \
#                         'VV_DOWN_O2_RATES-18->19' \
#                         'VV_DOWN_O2_RATES-19->20' \
#                         'VV_DOWN_O2_RATES-20->21' \
#                         'VV_DOWN_O2_RATES-21->22' \
#                         'VV_DOWN_O2_RATES-22->23' \
#                         'VV_DOWN_O2_RATES-23->24' \
#                         'VV_DOWN_O2_RATES-24->25' \
#                         'VV_DOWN_O2_RATES-25->26' \
#                         'VV_DOWN_O2_RATES-26->27' \
#                         'VV_DOWN_O2_RATES-27->28' \
#                         'VV_DOWN_O2_RATES-28->29' \
#                         'VV_DOWN_O2_RATES-29->30' \
#                         'VV_DOWN_O2_RATES-30->31' \
#                         'VV_DOWN_O2_RATES-31->32' \
#                         'VV_DOWN_O2_RATES-32->33' \
#                         'VV_DOWN_O2_RATES-33->34' \
#                         'VV_DOWN_O2_RATES-34->35' \
#                         'VV_DOWN_NO_RATES-0->1'   \
#                         'VV_DOWN_NO_RATES-1->2'   \
#                         'VV_DOWN_NO_RATES-2->3'   \
#                         'VV_DOWN_NO_RATES-3->4'   \
#                         'VV_DOWN_NO_RATES-4->5'   \
#                         'VV_DOWN_NO_RATES-5->6'   \
#                         'VV_DOWN_NO_RATES-6->7'   \
#                         'VV_DOWN_NO_RATES-7->8'   \
#                         'VV_DOWN_NO_RATES-8->9'   \
#                         'VV_DOWN_NO_RATES-9->10'  \
#                         'VV_DOWN_NO_RATES-10->11' \
#                         'VV_DOWN_NO_RATES-11->12' \
#                         'VV_DOWN_NO_RATES-12->13' \
#                         'VV_DOWN_NO_RATES-13->14' \
#                         'VV_DOWN_NO_RATES-14->15' \
#                         'VV_DOWN_NO_RATES-15->16' \
#                         'VV_DOWN_NO_RATES-16->17' \
#                         'VV_DOWN_NO_RATES-17->18' \
#                         'VV_DOWN_NO_RATES-18->19' \
#                         'VV_DOWN_NO_RATES-19->20' \
#                         'VV_DOWN_NO_RATES-20->21' \
#                         'VV_DOWN_NO_RATES-21->22' \
#                         'VV_DOWN_NO_RATES-22->23' \
#                         'VV_DOWN_NO_RATES-23->24' \
#                         'VV_DOWN_NO_RATES-24->25' \
#                         'VV_DOWN_NO_RATES-25->26' \
#                         'VV_DOWN_NO_RATES-26->27' \
#                         'VV_DOWN_NO_RATES-27->28' \
#                         'VV_DOWN_NO_RATES-28->29' \
#                         'VV_DOWN_NO_RATES-29->30' \
#                         'VV_DOWN_NO_RATES-30->31' \
#                         'VV_DOWN_NO_RATES-31->32' \
#                         'VV_DOWN_NO_RATES-32->33' \
#                         'VV_DOWN_NO_RATES-33->34' \
#                         'VV_DOWN_NO_RATES-34->35' \
#                         'VV_DOWN_NO_RATES-35->36' \
#                         'VV_DOWN_NO_RATES-36->37' \
#                         'VV_DOWN_NO_RATES-37->38'
                          )

declare -a dataset_up=('VV_UP_N2_RATES-1->0'   \
                       'VV_UP_N2_RATES-2->1'   \
                       'VV_UP_N2_RATES-3->2'   \
                       'VV_UP_N2_RATES-4->3'   \
                       'VV_UP_N2_RATES-5->4'   \
                       'VV_UP_N2_RATES-6->5'   \
                       'VV_UP_N2_RATES-7->6'   \
                       'VV_UP_N2_RATES-8->7'   \
                       'VV_UP_N2_RATES-9->8'   \
                       'VV_UP_N2_RATES-10->9'  \
                       'VV_UP_N2_RATES-11->10' \
                       'VV_UP_N2_RATES-12->11' \
                       'VV_UP_N2_RATES-13->12' \
                       'VV_UP_N2_RATES-14->13' \
                       'VV_UP_N2_RATES-15->14' \
                       'VV_UP_N2_RATES-16->15' \
                       'VV_UP_N2_RATES-17->16' \
                       'VV_UP_N2_RATES-18->17' \
                       'VV_UP_N2_RATES-19->18' \
                       'VV_UP_N2_RATES-20->19' \
                       'VV_UP_N2_RATES-21->20' \
                       'VV_UP_N2_RATES-22->21' \
                       'VV_UP_N2_RATES-23->22' \
                       'VV_UP_N2_RATES-24->23' \
                       'VV_UP_N2_RATES-25->24' \
                       'VV_UP_N2_RATES-26->25' \
                       'VV_UP_N2_RATES-27->26' \
                       'VV_UP_N2_RATES-28->27' \
                       'VV_UP_N2_RATES-29->28' \
                       'VV_UP_N2_RATES-30->29' \
                       'VV_UP_N2_RATES-31->30' \
                       'VV_UP_N2_RATES-32->31' \
                       'VV_UP_N2_RATES-33->32' \
                       'VV_UP_N2_RATES-34->33' \
                       'VV_UP_N2_RATES-35->34' \
                       'VV_UP_N2_RATES-36->35' \
                       'VV_UP_N2_RATES-37->36' \
                       'VV_UP_N2_RATES-38->37' \
                       'VV_UP_N2_RATES-39->38' \
                       'VV_UP_N2_RATES-40->39' \
                       'VV_UP_N2_RATES-41->40' \
                       'VV_UP_N2_RATES-42->41' \
                       'VV_UP_N2_RATES-43->42' \
                       'VV_UP_N2_RATES-44->43' \
                       'VV_UP_N2_RATES-45->44' \
                       'VV_UP_N2_RATES-46->45' \
                       'VV_UP_O2_RATES-1->0'   \
                       'VV_UP_O2_RATES-2->1'   \
                       'VV_UP_O2_RATES-3->2'   \
                       'VV_UP_O2_RATES-4->3'   \
                       'VV_UP_O2_RATES-5->4'   \
                       'VV_UP_O2_RATES-6->5'   \
                       'VV_UP_O2_RATES-7->6'   \
                       'VV_UP_O2_RATES-8->7'   \
                       'VV_UP_O2_RATES-9->8'   \
                       'VV_UP_O2_RATES-10->9'  \
                       'VV_UP_O2_RATES-11->10' \
                       'VV_UP_O2_RATES-12->11' \
                       'VV_UP_O2_RATES-13->12' \
                       'VV_UP_O2_RATES-14->13' \
                       'VV_UP_O2_RATES-15->14' \
                       'VV_UP_O2_RATES-16->15' \
                       'VV_UP_O2_RATES-17->16' \
                       'VV_UP_O2_RATES-18->17' \
                       'VV_UP_O2_RATES-19->18' \
                       'VV_UP_O2_RATES-20->19' \
                       'VV_UP_O2_RATES-21->20' \
                       'VV_UP_O2_RATES-22->21' \
                       'VV_UP_O2_RATES-23->22' \
                       'VV_UP_O2_RATES-24->23' \
                       'VV_UP_O2_RATES-25->24' \
                       'VV_UP_O2_RATES-26->25' \
                       'VV_UP_O2_RATES-27->26' \
                       'VV_UP_O2_RATES-28->27' \
                       'VV_UP_O2_RATES-29->28' \
                       'VV_UP_O2_RATES-30->29' \
                       'VV_UP_O2_RATES-31->30' \
                       'VV_UP_O2_RATES-32->31' \
                       'VV_UP_O2_RATES-33->32' \
                       'VV_UP_O2_RATES-34->33' \
                       'VV_UP_O2_RATES-35->34' \
                       'VV_UP_NO_RATES-1->0'   \
                       'VV_UP_NO_RATES-2->1'   \
                       'VV_UP_NO_RATES-3->2'   \
                       'VV_UP_NO_RATES-4->3'   \
                       'VV_UP_NO_RATES-5->4'   \
                       'VV_UP_NO_RATES-6->5'   \
                       'VV_UP_NO_RATES-7->6'   \
                       'VV_UP_NO_RATES-8->7'   \
                       'VV_UP_NO_RATES-9->8'   \
                       'VV_UP_NO_RATES-10->9'  \
                       'VV_UP_NO_RATES-11->10' \
                       'VV_UP_NO_RATES-12->11' \
                       'VV_UP_NO_RATES-13->12' \
                       'VV_UP_NO_RATES-14->13' \
                       'VV_UP_NO_RATES-15->14' \
                       'VV_UP_NO_RATES-16->15' \
                       'VV_UP_NO_RATES-17->16' \
                       'VV_UP_NO_RATES-18->17' \
                       'VV_UP_NO_RATES-19->18' \
                       'VV_UP_NO_RATES-20->19' \
                       'VV_UP_NO_RATES-21->20' \
                       'VV_UP_NO_RATES-22->21' \
                       'VV_UP_NO_RATES-23->22' \
                       'VV_UP_NO_RATES-24->23' \
                       'VV_UP_NO_RATES-25->24' \
                       'VV_UP_NO_RATES-26->25' \
                       'VV_UP_NO_RATES-27->26' \
                       'VV_UP_NO_RATES-28->27' \
                       'VV_UP_NO_RATES-29->28' \
                       'VV_UP_NO_RATES-30->29' \
                       'VV_UP_NO_RATES-31->30' \
                       'VV_UP_NO_RATES-32->31' \
                       'VV_UP_NO_RATES-33->32' \
                       'VV_UP_NO_RATES-34->33' \
                       'VV_UP_NO_RATES-35->34' \
                       'VV_UP_NO_RATES-36->35' \
                       'VV_UP_NO_RATES-37->36' \
                       'VV_UP_NO_RATES-38->37' )

for i in "${dataset_down[@]}";
do
  python3 regression_multioutput_down.py $i
done

for i in "${dataset_up[@]}";
do
  python3 regression_multioutput_up.py $i
done

cd ..
