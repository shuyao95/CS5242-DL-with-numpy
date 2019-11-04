DIR="/Users/shuyao/Downloads/submissions"

for f in `ls "${DIR}"`; do 
    echo "Process student: ${f}..."
    python grade.py --root "${DIR}/${f}"
done