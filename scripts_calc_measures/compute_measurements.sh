#!/bin/bash
# Obtains the final vessel based segmentation
 
# PARAMETERS
OPFRONT_QUEUE="day"
MEMORY="10G"


BASEDIR="/scratch/agarcia/Tests/LabelRefinement_THIRONA/"
BASEBIN="/scratch/agarcia/Tests/LabelRefinement_THIRONA/Scripts/"

FOLDER_CTS="${BASEDIR}/CTs/"
FOLDER_SEGMENTATIONS="${BASEDIR}/Segmentations/"

NOW="$(date +%d-%m-%Y_%H-%M-%S)"
FOLDER_LOGS="${FOLDER_SEGMENTATIONS}/logs"
LOG_FILE="${FOLDER_LOGS}/measurements_airways_${NOW}.log"
MEASURES_BIN="${BASEBIN}/measurements_individual.sh"


#Export paths to required libraries
BASEDIR_LIBS="/archive/pulmo/Code_APerez/Cluster/Libraries/"
# libboost_filesystem.so.1.47.0
PATH_LIBRARIES="${BASEDIR_LIBS}/boost_cluster/bin/lib/"
# libCGAL.so
PATH_LIBRARIES="${BASEDIR_LIBS}/CGAL/CGAL-4.4/lib/:${PATH_LIBRARIES}"
# libgmp.so
PATH_LIBRARIES="${BASEDIR_LIBS}/GMP/build/lib/:${PATH_LIBRARIES}"
# libgsl.so
PATH_LIBRARIES="${BASEDIR_LIBS}/GSL_unnecesary/:${PATH_LIBRARIES}"
# libgts-0.7.so
PATH_LIBRARIES="${BASEDIR_LIBS}/GTS/build/lib/:${PATH_LIBRARIES}"
# libmpfr.so
PATH_LIBRARIES="${BASEDIR_LIBS}/MPFR/build/lib/:${PATH_LIBRARIES}"
# libkdtree.so
PATH_LIBRARIES="${BASEDIR_LIBS}/kdtree/build/lib/:${PATH_LIBRARIES}"


mkdir -p $FOLDER_SEGMENTATIONS
mkdir -p $FOLDER_LOGS

echo "FOLDER_CTS: $FOLDER_CTS" > $LOG_FILE
echo "FOLDER_SEGMENTATIONS: $FOLDER_SEGMENTATIONS" >> $LOG_FILE
eval cat $LOG_FILE

LIST_INPUT_FILES=$(find $FOLDER_SEGMENTATIONS -type f -maxdepth 1)


for IN_INPUT_FILE in $LIST_INPUT_FILES
do
    IN_FILE=$(basename $IN_INPUT_FILE)
    IN_FILE_NOEXT=${IN_FILE%_manual-airways.nii.gz}
    JOB_NAME="OPr_${IN_FILE_NOEXT}"

    IN_VOL_FILE="${FOLDER_CTS}/${IN_FILE_NOEXT}.dcm"
    IN_SEGMEN_FILE=${IN_INPUT_FILE}

    CALL="${MEASURES_BIN} ${IN_VOL_FILE} ${IN_SEGMEN_FILE} ${FOLDER_SEGMENTATIONS} ${PATH_LIBRARIES}"
    
    echo $CALL >> $LOG_FILE
    #echo $CALL && eval $CALL
    echo $CALL | qsub -q ${OPFRONT_QUEUE} -l h_vmem=${MEMORY} -j y -N ${JOB_NAME} -o ${FOLDER_LOGS}/${JOB_NAME}_${NOW}.log
done
