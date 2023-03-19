import json

content = '{"ROOT" :  "C:/Users/Baaaatttlllllllleeee/PycharmProjects/MyBaseline/data",  "ValAnn" : "/annotations",  "ValImg" : "val2017"}'

result = {'acc' : '0.94', 'loss' : '0.0002', 'epoch' : '6', 'memory' : '123MB', 'data_time' : '0.34s'}

def write2end(content, file):
    cntr = 1
    file.write("\n{")
    for k,v in content.items():
        if cntr == 5:
            file.write(f'\"{k}\" : \"{v}\"')
            file.write("}")
            return
        file.write(f'\"{k}\" : \"{v}\" ,')
        cntr += 1


with open('PATH.json', 'a') as f:
    write2end(result, f)