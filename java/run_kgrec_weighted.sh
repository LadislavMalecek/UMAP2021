mvn clean compile;
export MAVEN_OPTS="-Xmx11000M"


# before running this
# - generate results of standard non-weighted script
# - generate weights using python script

for g_s in 2 3 4 8
do
    for g in sim
    do
        mvn exec:java -Dexec.mainClass="gfar.AggregateRecommedations" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s} --userPref"

        mvn exec:java -Dexec.mainClass="gfar.RunDDOA" \
            -Dexec.args="--data=kgrec --groupType=${g} --groupSize=${g_s} --userPref"
    done
done
