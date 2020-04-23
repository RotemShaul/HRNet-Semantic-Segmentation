#!/bin/bash

# Shlomit 	June 2019
# ./seq_arr.sh -e 4 -c "bsub -H -J "test[1-4]" sleep 5"
# ./seq_arr.sh -f c.sh -e 10

RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

display_usage() { 
    echo -e "\nUsage:\n    ${GREEN}$(basename $0) -f bsub_file -e max_element -c bsub_command -d ended${NC}\n"
    echo -e "\nExamples:\n"
    echo -e "Send bsub file with sequence array:"
    echo -e "    ${BLUE}$(basename $0) -f my_bsub.sh -e 4${NC}\n"
    echo -e "Send bsub file with sequence array + set dependecy condition (default is 'done'): "
    echo -e "    ${BLUE}$(basename $0) -f my_bsub.sh -e 4 -d ended${NC}\n"
    echo -e "Send the submit in the command line:"
    echo -e "    ${BLUE}$(basename $0) -e 5 -c \"bsub -H -J \"test[1-5]\" sleep 4\"${NC}\n"
    exit;
}

if [ $# == 0 ]; then display_usage; fi

echo -e "\n${RED}NOTE: the bsub command must have the -H option!!!${NC}\n\n"

while [ $# -gt 0 ]
do
    case "$1" in
    -e)
        shift
        MAX_ELEMENT=$1
        shift
        ;;
    -f) 
        shift
        BSUB_FILE=$1
        shift
        ;;
    -c) 
        shift
        BSUB_COMMAND=$1
        shift
        ;;
    -d) 
        shift
        DEPENDECY_EXP=$1
        shift
        ;;
    *)
        display_usage
        ;;
    esac
done 
 
if [[ (-z "$BSUB_FILE" && -z "$BSUB_COMMAND") || -z "$MAX_ELEMENT" ]]; then display_usage; fi

if [ -z "$DEPENDECY_EXP" ]; then
    DEPENDECY_EXP="done"
fi

if [  "$BSUB_COMMAND" ]; then
    echo $BSUB_COMMAND
    jobid=$(${BSUB_COMMAND} |& tee |grep Job| sed 's/Job <\([0-9]*\).*/\1/')
else
    echo $BSUB_FILE
    jobid=$(bsub < $BSUB_FILE |& tee |grep Job| sed 's/Job <\([0-9]*\).*/\1/')
fi
echo Job $jobid
 
for (( i=1; i<=$MAX_ELEMENT-1; i++ ))
do
    j=$((i+1))
    bmod -w "$DEPENDECY_EXP($jobid[$i])" "$jobid[$j]"
done
 
#sleep 3
bresume $jobid
