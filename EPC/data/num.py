# -*- coding: utf-8 -*-
f = open(r'all1.txt', 'r', encoding='utf-8')
num = 0
num1 = 0
while True:
    con = f.readline()
    if con == '':
        break
    if con.strip().split()[0] == '0':
        num += 1
    else:
        num1 += 1

print(num)
print(num1)
