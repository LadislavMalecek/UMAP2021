mvn clean compile;
# export MAVEN_OPTS="-Xmx9000M"


for g_s in 2 3 4 8
do
    for g in sim div
    do
        mvn exec:java -Dexec.mainClass="gfar.AggregateRecommedations" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s}"

        mvn exec:java -Dexec.mainClass="gfar.RunDDOA" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s}"

        mvn exec:java -Dexec.mainClass="gfar.RunGreedyAlgorithms" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s}"

        mvn exec:java -Dexec.mainClass="gfar.RunGFAR" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s}"
    done
done