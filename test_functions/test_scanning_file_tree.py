# %%
import glob
import os
base_path = '/Users/klara/Documents/uni/bachelorarbeit'
print(base_path)
# %%
paths = [path for path in os.scandir(base_path) if path.name.endswith('.pdf')]
print(paths)
# %%
for fp in paths:
    print(fp.path)
# %%
paths2 = [path for path in glob.glob(base_path + '/') if path.endswith('.pdf')]
print(paths2)

# %%
print(os.listdir(base_path))
# %%
# best option
all_paths = [os.path.join(r,file) for r,d,f in os.walk(base_path) for file in f if file.endswith('.pdf')]
print(all_paths)
print(len(all_paths))
# %%
path = all_paths[0]
id = path.split('/')[-1].split('.')[0]

# %%
print(id)
# %%
print(hash(path))
# %%
