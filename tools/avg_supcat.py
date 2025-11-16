base_dir = '../work_dirs/tiny/clip/SimpleMultiModalModule_setting29/'
# split1_path = base_dir + 'split1/test_log_clip_split1_config.txt'
# split2_path = base_dir + 'split2/test_split2.log'
# split3_path = base_dir + 'split3/test_split3.log'
# split4_path = base_dir + 'split4/test_split4.log'
# split5_path = base_dir + 'split5/test_split5.log'
split1_path = base_dir + 'split1/test_log_clip_split1_config.txt'
split2_path = base_dir + 'split2/test_log_clip_split2_config.txt'
split3_path = base_dir + 'split3/test_log_clip_split3_config.txt'
split4_path = base_dir + 'split4/test_log_clip_split4_config.txt'
split5_path = base_dir + 'split5/test_log_clip_split5_config.txt'
supcats = ['animal','animal_body','animal_face','bird','clothes','furniture','hand','person','vehicle']

res = {}
for supcat in supcats:
    res [supcat] = {}
    for split in ['split_1','split_2','split_3','split_4','split_5']:
        res[supcat][split] = {}
        for th in ['PCK@0.05','PCK@0.1','PCK@0.15','PCK@0.2','PCK@0.25']:
            res[supcat][split][th] = 0

for i,file_path in enumerate([split1_path,split2_path,split3_path,split4_path,split5_path]):
    split = 'split_' + str(i+1)
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_splits = line.split()
            if len(line_splits) == 0:
                continue
            supcat_name = line_splits[0]
            if supcat_name in supcats and line_splits[2] == 'mean:':
                pck_th = line_splits[1]
                pck_val = float(line_splits[3])
                res[supcat_name][split][pck_th] = pck_val

print('debug')
for supcat in supcats:
    res[supcat]['avg'] = {}
    for th in ['PCK@0.05','PCK@0.1','PCK@0.15','PCK@0.2','PCK@0.25']:
        count = 0
        performance_sum = 0
        for split in ['split_1','split_2','split_3','split_4','split_5']:
            if res[supcat][split][th] <= 0:
                continue
            performance_sum += res[supcat][split][th]
            count += 1
        performance_avg = performance_sum / count
        res[supcat]['avg'][th] = performance_avg

clear_res = {}
for supcat in supcats:
    clear_res[supcat] = res[supcat]['avg']

print('debug')