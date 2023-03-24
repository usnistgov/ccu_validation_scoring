#!/bin/sh


echo 'user_id	file_id	segment_id	norm	status' > ../data/norms.tab
for x in F1:7:20 F2:15:24 F4:6:11 F5:5:12 F6:6:11 F7:2:15 F8:5:11 F9:2:11 F10:6:15 F11:6:13  ; do
    file=`echo $x|awk -F: '{print $1}'`
    on=`echo $x|awk -F: '{print $2}'`
    off=`echo $x|awk -F: '{print $3}'`
    for seg in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020 0021 0022 0023 0024 0025 0026 0027 0028 0029 0030 0031 0032 0033 0034 0035 0036 0037 0038 0039 0040 ; do
	s=`echo $seg|sed 's/^0*//'`
	state=`echo $on $off $s | awk '{ if ($3 >= $1 && $3 <= $2) {print "on"} else {print "off"} }'`
	if [ $state = 'on' ] ; then
    	    echo "1012\t$file\t${file}_$seg\t101\tadhere" >> ../data/norms.tab
	else
    	    echo "1012\t$file\t${file}_$seg\tnone\tEMPTY_NA" >> ../data/norms.tab
	fi
        done
done

exit

dir=../../../pass_submissions/pass_submissions_WeightedF1/ND/system1
si=system_output.index.tab
echo "file_id\tis_processed\tmessage\tfile_path" > $dir/$si
for file in F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 ; do
    echo "$file\tTrue\t\t$file.tab" >> $dir/$si
    echo "file_id\tnorm\tstart\tend\tstatus\tllr" > $dir/$file.tab
done
for x in F3:20.0:100.0 F4:20.0:60.0 F5:25.0:55.0 F6:5.0:75.0 F7:25.0:55.0 F8:25.0:75.0 F9:25.0:65.0 F10:20.0:55.0 F11:5.0:35.0  ; do
    file=`echo $x|awk -F: '{print $1}'`
    on=`echo $x|awk -F: '{print $2}'`
    off=`echo $x|awk -F: '{print $3}'`
    echo "$file\t101\t$on\t$off\tadhere\t0.0" >> $dir/$file.tab
done


echo 'file_id	segment_id	start	end' > segments.tab
for x in F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 ; do
    for seg in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020 0021 0022 0023 0024 0025 0026 0027 0028 0029 0030 0031 0032 0033 0034 0035 0036 0037 0038 0039 0040 ; do
	b=`echo $seg| awk '{print ($1-1)*5".0"}'`
	e=`echo $seg| awk '{print $1*5".0"}'`
    	echo "$x\t${x}_$seg\t$b\t$e" >> segments.tab
    done
done
