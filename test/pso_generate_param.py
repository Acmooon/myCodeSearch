import csv
import numpy as np

def is_in_history(history, param):
    for row in history:
        # print(f'row:{row},param:{param}')
        if row[0]==param[0] and row[1]==param[1] and row[2]==param[2] and row[3]==param[3] and row[4]==param[4]:
            return True
    return False

readPath='./src/param_history.csv'
read_file=open(readPath,'r',encoding='utf-8',newline='')
reader = csv.reader(read_file)
history=[]
for row in reader:
    history.append(row)

writePath='./src/param_list.csv'
write_file=open(writePath,'w',encoding='utf-8',newline='')
result_writer = csv.writer(write_file)

header=['pso_options_c1', 'pso_options_c2', 'pso_options_w','pso_n_particles','pso_iter']
result_writer.writerow(header)
num=0

c1_range=np.arange(3,5,1)
c2_range=np.arange(3,5,1)
w_range=np.arange(3,5,1)
n_range=np.arange(200,401,50)
i_range=np.arange(200,301,25)

for c1 in c1_range:
    for c2 in c2_range:
        for w in w_range:
            for n in n_range:
                for i in i_range:
                    param=[c1,c2,w,n,i]
                    if is_in_history(history,param):
                        continue
                    else:
                        result_writer.writerow(param)
                        num+=1

print(f'c1_range_len:{len(c1_range)}')
print(f'c2_range_len:{len(c2_range)}')
print(f'w_range_len:{len(w_range)}')
print(f'n_range_len:{len(n_range)}')
print(f'i_range_len:{len(i_range)}')
print(f'write {num} lines')