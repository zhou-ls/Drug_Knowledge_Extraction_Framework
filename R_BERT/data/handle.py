import json
import re

# path = "label.json"
# with open(path, "r") as f:
#     row_data = json.load(f)
#     print(type(row_data))
#     print(row_data["negative"])
import xlwt
import random
paths=['train.txt','test.txt']
resultPaths=['train.xls','test.xls']

for i in range(len(paths)):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet("Sheet_1")
    with open(paths[i], 'r', encoding='utf-8') as f:
        for indix,line in enumerate(f.readlines()):
            label=str(line[0])
            if line[0] == '0':
                label_des = "negative"
            else:
                label_des = "positive"
            # pattern = re.compile(r'\s(.*?)\$')
            # subject="<e1>"+re.findall(pattern,line)[0]+"</e1>"
            # line=line.replace(re.findall(pattern, line)[0], subject)
            #
            # pattern = re.compile(r'\$(.*?)\$')
            # object = "<e2>"+re.findall(pattern, line)[0]+"</e2>"
            # line=line.replace( "$"+re.findall(pattern, line)[0] + "$", object)
            sheet.write(indix,0,label_des)
            sheet.write(indix,1,line[2:-1])
    workbook.save(resultPaths[i])

