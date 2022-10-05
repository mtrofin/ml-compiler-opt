import csv
import glob


f_to_freq = {}
f_to_mod = {}
for filename in glob.iglob('/work/muppet_corpus_stats/**', recursive=True):
  if not filename.endswith('.stats'):
    continue
  with open(filename) as f:
    reader = csv.reader(f)
    for l in reader:
      fname = l[0].split('.llvm.')[0]
      md5 = l[1]
      f_to_mod[fname] = filename
      value = float(l[2])
      f_to_freq[fname] = value

total = sum(v for _, v in f_to_freq.items())

rev_freqs = sorted([(freq, fn) for fn, freq in f_to_freq.items()], reverse=True)
important = []
t = 0.0
mods = set()
for freq, fn in rev_freqs:
  if t > 0.99 * total:
    break
  t += freq
  important.append((freq, fn))
  mods.add(f_to_mod[fn])
print(len(important))
print(len(mods))