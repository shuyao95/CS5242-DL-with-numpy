# change the format into the right one for grading

ORI_DIR="/Users/shuyao/Downloads/Assignment Submission"
# ORI_DIR="/Users/shuyao/Downloads/sbtest"
TARGET_DIR="/Users/shuyao/Downloads/submissions"
TEMP="/Users/shuyao/Downloads/temp"

for f in `ls "${ORI_DIR}"`; do 
    ID="$(cut -d'.' -f1 <<<${f})"
    ext="$(cut -d'.' -f2 <<<${f})"
    echo "Process student: ${ID}..."
    # check the extension is zip
    if [ $ext = 'zip' ]; then
        unzip -qq "${ORI_DIR}/${f}" -d "${TEMP}"
        # remove redundent folder
        if [ -d "${TEMP}/${ID}/${ID}" ]; then
            echo "  - remove redundent folder"
            mv "${TEMP}/${ID}" "${TEMP}/temp"
            mv "${TEMP}/${ID}/${ID}" "${TEMP}"
            rm -rf "${TEMP}/temp"
        fi
        # add ID folder
        if [ ! -d "${TEMP}/${ID}" ]; then
            echo "  - add ID folder"
            mv "${TEMP}" "${TEMP}/../${ID}"
            mkdir "${TEMP}"
            mv "${TEMP}/../${ID}" "${TEMP}"
        fi
        # add codes folder
        if [ ! -d "${TEMP}/${ID}/codes" ]; then
            echo "  - add codes folder"
            mv "${TEMP}/${ID}" "${TEMP}/codes"
            mkdir "${TEMP}/${ID}"
            mv "${TEMP}/codes" "${TEMP}/${ID}/"
        fi
        # rm -rf "${ORI_DIR}/${f}"
        mv "${TEMP}/${ID}" "${TARGET_DIR}"
    else
        echo "student: $ID, error: wrong compression"
    fi
done