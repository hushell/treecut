import os
import time

#os.system("source /scratch/a1/sge/settings.sh")
os.system("rm -f log/*")

name = "tcucm_"
bash_del = False

datasets = ["train","test"]
grps = [range(x,x+10) for x in range(1,200,10)]
strid = 10

for ds in datasets:
    name_data = ds[:2]
    for j in xrange(1, 200, strid):
        img_s = j
        img_t = j+strid-1
        print "------------ training machine (%d,%d) -------------" % (img_s,img_t)

        script = \
            "#!/bin/bash" + "\n" \
            + "#" + "\n" \
            + "# use current working directory for input and output - defaults is" + "\n" \
            + "# to use the users home directory" + "\n" \
            + "#$ -cwd" + "\n" \
            + "#" + "\n" \
            + "# name this job" + "\n" \
            + "#$ -N %s_%d2%d" % (name_data,img_s,img_t) + "\n" \
            + "#" + "\n" \
            + "# send stdout and stderror to this file" + "\n" \
            + "#$ -o log/%s_%d2%d.out" % (name_data,img_s,img_t) + "\n" \
            + "#$ -j y" + "\n\n" \
            + "#see where the job is being run" + "\n" \
            + "hostname" + "\n\n" \
            + 'export PATH=/scratch/a1/sge/bin/lx-amd64:/bin:/sbin:/usr/local/bin:/usr/bin:/usr/local/apps/bin:/usr/bin/X11:$PATH' + '\n' \
            + "# print date and time" + "\n" \
            + "date" + "\n" \
            + "# experiment code" + "\n" \
            + "matlab -nodisplay -nosplash -nodesktop -r 'test23_paral %s %d %d'" % (ds,img_s,img_t) + "\n" \
            + "# print date and time again" + "\n" \
            + "date" + "\n"

        bash_file = open("sge_%s_%d2%d.sh" % (name_data,img_s,img_t), "w")
        bash_file.write(script)
        bash_file.close()

        os.system("chmod +x " + "sge_%s_%d2%d.sh" % (name_data,img_s,img_t))
        #os.system("sh " + "sge_%s_%d2%d.sh" % (name_data,img_s,img_t))
        os.system("qsub " + "sge_%s_%d2%d.sh" % (name_data,img_s,img_t))
        time.sleep(1.13)
        if bash_del == True:
            os.system("rm -f " + "sge_%s_%d2%d.sh" % (name_data,img_s,img_t))

