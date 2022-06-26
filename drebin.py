import staticAnalyzer
import sys
import os
import settings


#usage: python drebin.py .
dirs = [settings.GOOD_APK_DIR, settings.BAD_APK_DIR]

for cur_dir in dirs:
  count = 0
  dir_total = len(os.listdir(cur_dir))

  for file in os.listdir(cur_dir):
      print(cur_dir, ": ", count, "of", dir_total)
      count += 1

      if file.endswith("apk"):
          full_path_apk = cur_dir + file
          print("\n[I]\t", "Extracting data from", file)
          staticAnalyzer.run(full_path_apk, sys.argv[1])
      else:
          print("\n[I]\t", file, "not an APK...")
