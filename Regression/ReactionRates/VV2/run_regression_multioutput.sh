#!/bin/bash

cd src

declare -a dataset_down=('kvvs_down_n2_no-1->0'   \
                         'kvvs_down_n2_no-2->1'   \
                         'kvvs_down_n2_no-3->2'   \
                         'kvvs_down_n2_no-4->3'   \
                         'kvvs_down_n2_no-5->4'   \
                         'kvvs_down_n2_no-6->5'   \
                         'kvvs_down_n2_no-7->6'   \
                         'kvvs_down_n2_no-8->7'   \
                         'kvvs_down_n2_no-9->8'   \
                         'kvvs_down_n2_no-10->9'  \
                         'kvvs_down_n2_no-11->10' \
                         'kvvs_down_n2_no-12->11' \
                         'kvvs_down_n2_no-13->12' \
                         'kvvs_down_n2_no-14->13' \
                         'kvvs_down_n2_no-15->14' \
                         'kvvs_down_n2_no-16->15' \
                         'kvvs_down_n2_no-17->16' \
                         'kvvs_down_n2_no-18->17' \
                         'kvvs_down_n2_no-19->18' \
                         'kvvs_down_n2_no-20->19' \
                         'kvvs_down_n2_no-21->20' \
                         'kvvs_down_n2_no-22->21' \
                         'kvvs_down_n2_no-23->22' \
                         'kvvs_down_n2_no-24->23' \
                         'kvvs_down_n2_no-25->24' \
                         'kvvs_down_n2_no-26->25' \
                         'kvvs_down_n2_no-27->26' \
                         'kvvs_down_n2_no-28->27' \
                         'kvvs_down_n2_no-29->28' \
                         'kvvs_down_n2_no-30->29' \
                         'kvvs_down_n2_no-31->30' \
                         'kvvs_down_n2_no-32->31' \
                         'kvvs_down_n2_no-33->32' \
                         'kvvs_down_n2_no-34->33' \
                         'kvvs_down_n2_no-35->34' \
                         'kvvs_down_n2_no-36->35' \
                         'kvvs_down_n2_no-37->36' \
                         'kvvs_down_n2_no-38->37' \
                         'kvvs_down_n2_no-39->38' \
                         'kvvs_down_n2_no-40->39' \
                         'kvvs_down_n2_no-41->40' \
                         'kvvs_down_n2_no-42->41' \
                         'kvvs_down_n2_no-43->42' \
                         'kvvs_down_n2_no-44->43' \
                         'kvvs_down_n2_no-45->44' \
                         'kvvs_down_n2_no-46->45' \
                         'kvvs_down_n2_o2-1->0' \
                         'kvvs_down_n2_o2-2->1' \
                         'kvvs_down_n2_o2-3->2' \
                         'kvvs_down_n2_o2-4->3' \
                         'kvvs_down_n2_o2-5->4' \
                         'kvvs_down_n2_o2-6->5' \
                         'kvvs_down_n2_o2-7->6' \
                         'kvvs_down_n2_o2-8->7' \
                         'kvvs_down_n2_o2-9->8' \
                         'kvvs_down_n2_o2-10->9' \
                         'kvvs_down_n2_o2-11->10' \
                         'kvvs_down_n2_o2-12->11' \
                         'kvvs_down_n2_o2-13->12' \
                         'kvvs_down_n2_o2-14->13' \
                         'kvvs_down_n2_o2-15->14' \
                         'kvvs_down_n2_o2-16->15' \
                         'kvvs_down_n2_o2-17->16' \
                         'kvvs_down_n2_o2-18->17' \
                         'kvvs_down_n2_o2-19->18' \
                         'kvvs_down_n2_o2-20->19' \
                         'kvvs_down_n2_o2-21->20' \
                         'kvvs_down_n2_o2-22->21' \
                         'kvvs_down_n2_o2-23->22' \
                         'kvvs_down_n2_o2-24->23' \
                         'kvvs_down_n2_o2-25->24' \
                         'kvvs_down_n2_o2-26->25' \
                         'kvvs_down_n2_o2-27->26' \
                         'kvvs_down_n2_o2-28->27' \
                         'kvvs_down_n2_o2-29->28' \
                         'kvvs_down_n2_o2-30->29' \
                         'kvvs_down_n2_o2-31->30' \
                         'kvvs_down_n2_o2-32->31' \
                         'kvvs_down_n2_o2-33->32' \
                         'kvvs_down_n2_o2-34->33' \
                         'kvvs_down_n2_o2-35->34' \
                         'kvvs_down_n2_o2-36->35' \
                         'kvvs_down_n2_o2-37->36' \
                         'kvvs_down_n2_o2-38->37' \
                         'kvvs_down_n2_o2-39->38' \
                         'kvvs_down_n2_o2-40->39' \
                         'kvvs_down_n2_o2-41->40' \
                         'kvvs_down_n2_o2-42->41' \
                         'kvvs_down_n2_o2-43->42' \
                         'kvvs_down_n2_o2-44->43' \
                         'kvvs_down_n2_o2-45->44' \
                         'kvvs_down_n2_o2-46->45' \
                         'kvvs_down_no_n2-1->0' \
                         'kvvs_down_no_n2-2->1' \
                         'kvvs_down_no_n2-3->2' \
                         'kvvs_down_no_n2-4->3' \
                         'kvvs_down_no_n2-5->4' \
                         'kvvs_down_no_n2-6->5' \
                         'kvvs_down_no_n2-7->6' \
                         'kvvs_down_no_n2-8->7' \
                         'kvvs_down_no_n2-9->8' \
                         'kvvs_down_no_n2-10->9' \
                         'kvvs_down_no_n2-11->10' \
                         'kvvs_down_no_n2-12->11' \
                         'kvvs_down_no_n2-13->12' \
                         'kvvs_down_no_n2-14->13' \
                         'kvvs_down_no_n2-15->14' \
                         'kvvs_down_no_n2-16->15' \
                         'kvvs_down_no_n2-17->16' \
                         'kvvs_down_no_n2-18->17' \
                         'kvvs_down_no_n2-19->18' \
                         'kvvs_down_no_n2-20->19' \
                         'kvvs_down_no_n2-21->20' \
                         'kvvs_down_no_n2-22->21' \
                         'kvvs_down_no_n2-23->22' \
                         'kvvs_down_no_n2-24->23' \
                         'kvvs_down_no_n2-25->24' \
                         'kvvs_down_no_n2-26->25' \
                         'kvvs_down_no_n2-27->26' \
                         'kvvs_down_no_n2-28->27' \
                         'kvvs_down_no_n2-29->28' \
                         'kvvs_down_no_n2-30->29' \
                         'kvvs_down_no_n2-31->30' \
                         'kvvs_down_no_n2-32->31' \
                         'kvvs_down_no_n2-33->32' \
                         'kvvs_down_no_n2-34->33' \
                         'kvvs_down_no_n2-35->34' \
                         'kvvs_down_no_n2-36->35' \
                         'kvvs_down_no_n2-37->36' \
                         'kvvs_down_no_n2-38->37' \
                         'kvvs_down_no_o2-1->0' \
                         'kvvs_down_no_o2-2->1' \
                         'kvvs_down_no_o2-3->2' \
                         'kvvs_down_no_o2-4->3' \
                         'kvvs_down_no_o2-5->4' \
                         'kvvs_down_no_o2-6->5' \
                         'kvvs_down_no_o2-7->6' \
                         'kvvs_down_no_o2-8->7' \
                         'kvvs_down_no_o2-9->8' \
                         'kvvs_down_no_o2-10->9' \
                         'kvvs_down_no_o2-11->10' \
                         'kvvs_down_no_o2-12->11' \
                         'kvvs_down_no_o2-13->12' \
                         'kvvs_down_no_o2-14->13' \
                         'kvvs_down_no_o2-15->14' \
                         'kvvs_down_no_o2-16->15' \
                         'kvvs_down_no_o2-17->16' \
                         'kvvs_down_no_o2-18->17' \
                         'kvvs_down_no_o2-19->18' \
                         'kvvs_down_no_o2-20->19' \
                         'kvvs_down_no_o2-21->20' \
                         'kvvs_down_no_o2-22->21' \
                         'kvvs_down_no_o2-23->22' \
                         'kvvs_down_no_o2-24->23' \
                         'kvvs_down_no_o2-25->24' \
                         'kvvs_down_no_o2-26->25' \
                         'kvvs_down_no_o2-27->26' \
                         'kvvs_down_no_o2-28->27' \
                         'kvvs_down_no_o2-29->28' \
                         'kvvs_down_no_o2-30->29' \
                         'kvvs_down_no_o2-31->30' \
                         'kvvs_down_no_o2-32->31' \
                         'kvvs_down_no_o2-33->32' \
                         'kvvs_down_no_o2-34->33' \
                         'kvvs_down_no_o2-35->34' \
                         'kvvs_down_no_o2-36->35' \
                         'kvvs_down_no_o2-37->36' \
                         'kvvs_down_no_o2-38->37' \
                         'kvvs_down_o2_n2-1->0' \
                         'kvvs_down_o2_n2-2->1' \
                         'kvvs_down_o2_n2-3->2' \
                         'kvvs_down_o2_n2-4->3' \
                         'kvvs_down_o2_n2-5->4' \
                         'kvvs_down_o2_n2-6->5' \
                         'kvvs_down_o2_n2-7->6' \
                         'kvvs_down_o2_n2-8->7' \
                         'kvvs_down_o2_n2-9->8' \
                         'kvvs_down_o2_n2-10->9' \
                         'kvvs_down_o2_n2-11->10' \
                         'kvvs_down_o2_n2-12->11' \
                         'kvvs_down_o2_n2-13->12' \
                         'kvvs_down_o2_n2-14->13' \
                         'kvvs_down_o2_n2-15->14' \
                         'kvvs_down_o2_n2-16->15' \
                         'kvvs_down_o2_n2-17->16' \
                         'kvvs_down_o2_n2-18->17' \
                         'kvvs_down_o2_n2-19->18' \
                         'kvvs_down_o2_n2-20->19' \
                         'kvvs_down_o2_n2-21->20' \
                         'kvvs_down_o2_n2-22->21' \
                         'kvvs_down_o2_n2-23->22' \
                         'kvvs_down_o2_n2-24->23' \
                         'kvvs_down_o2_n2-25->24' \
                         'kvvs_down_o2_n2-26->25' \
                         'kvvs_down_o2_n2-27->26' \
                         'kvvs_down_o2_n2-28->27' \
                         'kvvs_down_o2_n2-29->28' \
                         'kvvs_down_o2_n2-30->29' \
                         'kvvs_down_o2_n2-31->30' \
                         'kvvs_down_o2_n2-32->31' \
                         'kvvs_down_o2_n2-33->32' \
                         'kvvs_down_o2_n2-34->33' \
                         'kvvs_down_o2_n2-35->34' \
                         'kvvs_down_o2_no-1->0' \
                         'kvvs_down_o2_no-2->1' \
                         'kvvs_down_o2_no-3->2' \
                         'kvvs_down_o2_no-4->3' \
                         'kvvs_down_o2_no-5->4' \
                         'kvvs_down_o2_no-6->5' \
                         'kvvs_down_o2_no-7->6' \
                         'kvvs_down_o2_no-8->7' \
                         'kvvs_down_o2_no-9->8' \
                         'kvvs_down_o2_no-10->9' \
                         'kvvs_down_o2_no-11->10' \
                         'kvvs_down_o2_no-12->11' \
                         'kvvs_down_o2_no-13->12' \
                         'kvvs_down_o2_no-14->13' \
                         'kvvs_down_o2_no-15->14' \
                         'kvvs_down_o2_no-16->15' \
                         'kvvs_down_o2_no-17->16' \
                         'kvvs_down_o2_no-18->17' \
                         'kvvs_down_o2_no-19->18' \
                         'kvvs_down_o2_no-20->19' \
                         'kvvs_down_o2_no-21->20' \
                         'kvvs_down_o2_no-22->21' \
                         'kvvs_down_o2_no-23->22' \
                         'kvvs_down_o2_no-24->23' \
                         'kvvs_down_o2_no-25->24' \
                         'kvvs_down_o2_no-26->25' \
                         'kvvs_down_o2_no-27->26' \
                         'kvvs_down_o2_no-28->27' \
                         'kvvs_down_o2_no-29->28' \
                         'kvvs_down_o2_no-30->29' \
                         'kvvs_down_o2_no-31->30' \
                         'kvvs_down_o2_no-32->31' \
                         'kvvs_down_o2_no-33->32' \
                         'kvvs_down_o2_no-34->33' \
                         'kvvs_down_o2_no-35->34' )

declare -a dataset_up=('kvvs_up_n2_no-0->1' \
                       'kvvs_up_n2_no-1->2' \
                       'kvvs_up_n2_no-2->3' \
                       'kvvs_up_n2_no-3->4' \
                       'kvvs_up_n2_no-4->5' \
                       'kvvs_up_n2_no-5->6' \
                       'kvvs_up_n2_no-6->7' \
                       'kvvs_up_n2_no-7->8' \
                       'kvvs_up_n2_no-8->9' \
                       'kvvs_up_n2_no-9->10' \
                       'kvvs_up_n2_no-10->11' \
                       'kvvs_up_n2_no-11->12' \
                       'kvvs_up_n2_no-12->13' \
                       'kvvs_up_n2_no-13->14' \
                       'kvvs_up_n2_no-14->15' \
                       'kvvs_up_n2_no-15->16' \
                       'kvvs_up_n2_no-16->17' \
                       'kvvs_up_n2_no-17->18' \
                       'kvvs_up_n2_no-18->19' \
                       'kvvs_up_n2_no-19->20' \
                       'kvvs_up_n2_no-20->21' \
                       'kvvs_up_n2_no-21->22' \
                       'kvvs_up_n2_no-22->23' \
                       'kvvs_up_n2_no-23->24' \
                       'kvvs_up_n2_no-24->25' \
                       'kvvs_up_n2_no-25->26' \
                       'kvvs_up_n2_no-26->27' \
                       'kvvs_up_n2_no-27->28' \
                       'kvvs_up_n2_no-28->29' \
                       'kvvs_up_n2_no-29->30' \
                       'kvvs_up_n2_no-30->31' \
                       'kvvs_up_n2_no-31->32' \
                       'kvvs_up_n2_no-32->33' \
                       'kvvs_up_n2_no-33->34' \
                       'kvvs_up_n2_no-34->35' \
                       'kvvs_up_n2_no-35->36' \
                       'kvvs_up_n2_no-36->37' \
                       'kvvs_up_n2_no-37->38' \
                       'kvvs_up_n2_no-38->39' \
                       'kvvs_up_n2_no-39->40' \
                       'kvvs_up_n2_no-40->41' \
                       'kvvs_up_n2_no-41->42' \
                       'kvvs_up_n2_no-42->43' \
                       'kvvs_up_n2_no-43->44' \
                       'kvvs_up_n2_no-44->45' \
                       'kvvs_up_n2_no-45->46' \
                       'kvvs_up_n2_o2-0->1' \
                       'kvvs_up_n2_o2-1->2' \
                       'kvvs_up_n2_o2-2->3' \
                       'kvvs_up_n2_o2-3->4' \
                       'kvvs_up_n2_o2-4->5' \
                       'kvvs_up_n2_o2-5->6' \
                       'kvvs_up_n2_o2-6->7' \
                       'kvvs_up_n2_o2-7->8' \
                       'kvvs_up_n2_o2-8->9' \
                       'kvvs_up_n2_o2-9->10' \
                       'kvvs_up_n2_o2-10->11' \
                       'kvvs_up_n2_o2-11->12' \
                       'kvvs_up_n2_o2-12->13' \
                       'kvvs_up_n2_o2-13->14' \
                       'kvvs_up_n2_o2-14->15' \
                       'kvvs_up_n2_o2-15->16' \
                       'kvvs_up_n2_o2-16->17' \
                       'kvvs_up_n2_o2-17->18' \
                       'kvvs_up_n2_o2-18->19' \
                       'kvvs_up_n2_o2-19->20' \
                       'kvvs_up_n2_o2-20->21' \
                       'kvvs_up_n2_o2-21->22' \
                       'kvvs_up_n2_o2-22->23' \
                       'kvvs_up_n2_o2-23->24' \
                       'kvvs_up_n2_o2-24->25' \
                       'kvvs_up_n2_o2-25->26' \
                       'kvvs_up_n2_o2-26->27' \
                       'kvvs_up_n2_o2-27->28' \
                       'kvvs_up_n2_o2-28->29' \
                       'kvvs_up_n2_o2-29->30' \
                       'kvvs_up_n2_o2-30->31' \
                       'kvvs_up_n2_o2-31->32' \
                       'kvvs_up_n2_o2-32->33' \
                       'kvvs_up_n2_o2-33->34' \
                       'kvvs_up_n2_o2-34->35' \
                       'kvvs_up_n2_o2-35->36' \
                       'kvvs_up_n2_o2-36->37' \
                       'kvvs_up_n2_o2-37->38' \
                       'kvvs_up_n2_o2-38->39' \
                       'kvvs_up_n2_o2-39->40' \
                       'kvvs_up_n2_o2-40->41' \
                       'kvvs_up_n2_o2-41->42' \
                       'kvvs_up_n2_o2-42->43' \
                       'kvvs_up_n2_o2-43->44' \
                       'kvvs_up_n2_o2-44->45' \
                       'kvvs_up_n2_o2-45->46' \
                       'kvvs_up_no_n2-0->1' \
                       'kvvs_up_no_n2-1->2' \
                       'kvvs_up_no_n2-2->3' \
                       'kvvs_up_no_n2-3->4' \
                       'kvvs_up_no_n2-4->5' \
                       'kvvs_up_no_n2-5->6' \
                       'kvvs_up_no_n2-6->7' \
                       'kvvs_up_no_n2-7->8' \
                       'kvvs_up_no_n2-8->9' \
                       'kvvs_up_no_n2-9->10' \
                       'kvvs_up_no_n2-10->11' \
                       'kvvs_up_no_n2-11->12' \
                       'kvvs_up_no_n2-12->13' \
                       'kvvs_up_no_n2-13->14' \
                       'kvvs_up_no_n2-14->15' \
                       'kvvs_up_no_n2-15->16' \
                       'kvvs_up_no_n2-16->17' \
                       'kvvs_up_no_n2-17->18' \
                       'kvvs_up_no_n2-18->19' \
                       'kvvs_up_no_n2-19->20' \
                       'kvvs_up_no_n2-20->21' \
                       'kvvs_up_no_n2-21->22' \
                       'kvvs_up_no_n2-22->23' \
                       'kvvs_up_no_n2-23->24' \
                       'kvvs_up_no_n2-24->25' \
                       'kvvs_up_no_n2-25->26' \
                       'kvvs_up_no_n2-26->27' \
                       'kvvs_up_no_n2-27->28' \
                       'kvvs_up_no_n2-28->29' \
                       'kvvs_up_no_n2-29->30' \
                       'kvvs_up_no_n2-30->31' \
                       'kvvs_up_no_n2-31->32' \
                       'kvvs_up_no_n2-32->33' \
                       'kvvs_up_no_n2-33->34' \
                       'kvvs_up_no_n2-34->35' \
                       'kvvs_up_no_n2-35->36' \
                       'kvvs_up_no_n2-36->37' \
                       'kvvs_up_no_n2-37->38' \
                       'kvvs_up_no_o2-0->1' \
                       'kvvs_up_no_o2-1->2' \
                       'kvvs_up_no_o2-2->3' \
                       'kvvs_up_no_o2-3->4' \
                       'kvvs_up_no_o2-4->5' \
                       'kvvs_up_no_o2-5->6' \
                       'kvvs_up_no_o2-6->7' \
                       'kvvs_up_no_o2-7->8' \
                       'kvvs_up_no_o2-8->9' \
                       'kvvs_up_no_o2-9->10' \
                       'kvvs_up_no_o2-10->11' \
                       'kvvs_up_no_o2-11->12' \
                       'kvvs_up_no_o2-12->13' \
                       'kvvs_up_no_o2-13->14' \
                       'kvvs_up_no_o2-14->15' \
                       'kvvs_up_no_o2-15->16' \
                       'kvvs_up_no_o2-16->17' \
                       'kvvs_up_no_o2-17->18' \
                       'kvvs_up_no_o2-18->19' \
                       'kvvs_up_no_o2-19->20' \
                       'kvvs_up_no_o2-20->21' \
                       'kvvs_up_no_o2-21->22' \
                       'kvvs_up_no_o2-22->23' \
                       'kvvs_up_no_o2-23->24' \
                       'kvvs_up_no_o2-24->25' \
                       'kvvs_up_no_o2-25->26' \
                       'kvvs_up_no_o2-26->27' \
                       'kvvs_up_no_o2-27->28' \
                       'kvvs_up_no_o2-28->29' \
                       'kvvs_up_no_o2-29->30' \
                       'kvvs_up_no_o2-30->31' \
                       'kvvs_up_no_o2-31->32' \
                       'kvvs_up_no_o2-32->33' \
                       'kvvs_up_no_o2-33->34' \
                       'kvvs_up_no_o2-34->35' \
                       'kvvs_up_no_o2-35->36' \
                       'kvvs_up_no_o2-36->37' \
                       'kvvs_up_no_o2-37->38' \
                       'kvvs_up_o2_n2-0->1' \
                       'kvvs_up_o2_n2-1->2' \
                       'kvvs_up_o2_n2-2->3' \
                       'kvvs_up_o2_n2-3->4' \
                       'kvvs_up_o2_n2-4->5' \
                       'kvvs_up_o2_n2-5->6' \
                       'kvvs_up_o2_n2-6->7' \
                       'kvvs_up_o2_n2-7->8' \
                       'kvvs_up_o2_n2-8->9' \
                       'kvvs_up_o2_n2-9->10' \
                       'kvvs_up_o2_n2-10->11' \
                       'kvvs_up_o2_n2-11->12' \
                       'kvvs_up_o2_n2-12->13' \
                       'kvvs_up_o2_n2-13->14' \
                       'kvvs_up_o2_n2-14->15' \
                       'kvvs_up_o2_n2-15->16' \
                       'kvvs_up_o2_n2-16->17' \
                       'kvvs_up_o2_n2-17->18' \
                       'kvvs_up_o2_n2-18->19' \
                       'kvvs_up_o2_n2-19->20' \
                       'kvvs_up_o2_n2-20->21' \
                       'kvvs_up_o2_n2-21->22' \
                       'kvvs_up_o2_n2-22->23' \
                       'kvvs_up_o2_n2-23->24' \
                       'kvvs_up_o2_n2-24->25' \
                       'kvvs_up_o2_n2-25->26' \
                       'kvvs_up_o2_n2-26->27' \
                       'kvvs_up_o2_n2-27->28' \
                       'kvvs_up_o2_n2-28->29' \
                       'kvvs_up_o2_n2-29->30' \
                       'kvvs_up_o2_n2-30->31' \
                       'kvvs_up_o2_n2-31->32' \
                       'kvvs_up_o2_n2-32->33' \
                       'kvvs_up_o2_n2-33->34' \
                       'kvvs_up_o2_n2-34->35' \
                       'kvvs_up_o2_no-0->1' \
                       'kvvs_up_o2_no-1->2' \
                       'kvvs_up_o2_no-2->3' \
                       'kvvs_up_o2_no-3->4' \
                       'kvvs_up_o2_no-4->5' \
                       'kvvs_up_o2_no-5->6' \
                       'kvvs_up_o2_no-6->7' \
                       'kvvs_up_o2_no-7->8' \
                       'kvvs_up_o2_no-8->9' \
                       'kvvs_up_o2_no-9->10' \
                       'kvvs_up_o2_no-10->11' \
                       'kvvs_up_o2_no-11->12' \
                       'kvvs_up_o2_no-12->13' \
                       'kvvs_up_o2_no-13->14' \
                       'kvvs_up_o2_no-14->15' \
                       'kvvs_up_o2_no-15->16' \
                       'kvvs_up_o2_no-16->17' \
                       'kvvs_up_o2_no-17->18' \
                       'kvvs_up_o2_no-18->19' \
                       'kvvs_up_o2_no-19->20' \
                       'kvvs_up_o2_no-20->21' \
                       'kvvs_up_o2_no-21->22' \
                       'kvvs_up_o2_no-22->23' \
                       'kvvs_up_o2_no-23->24' \
                       'kvvs_up_o2_no-24->25' \
                       'kvvs_up_o2_no-25->26' \
                       'kvvs_up_o2_no-26->27' \
                       'kvvs_up_o2_no-27->28' \
                       'kvvs_up_o2_no-28->29' \
                       'kvvs_up_o2_no-29->30' \
                       'kvvs_up_o2_no-30->31' \
                       'kvvs_up_o2_no-31->32' \
                       'kvvs_up_o2_no-32->33' \
                       'kvvs_up_o2_no-33->34' \
                       'kvvs_up_o2_no-34->35' )

for i in "${dataset_down[@]}";
do
  python3 regression_multioutput_down.py $i
done

for i in "${dataset_up[@]}";
do
  python3 regression_multioutput_up.py $i
done

cd ..