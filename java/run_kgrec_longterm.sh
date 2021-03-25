mvn clean compile;
export MAVEN_OPTS="-Xmx11000M"


for g_s in 2 3 4 8
do
    for g in sim div
    do
        mvn exec:java -Dexec.mainClass="gfar.RunDDOAInTime" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s}"
    done
done