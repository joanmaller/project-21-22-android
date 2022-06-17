import staticAnalyzer
import sys
import os
import settings


#usage: python drebin.py .
dirs = [settings.GOOD_APK_DIR, settings.BAD_APK_DIR]

for dir in dirs:
  for file in os.listdir(dir):

      if file.endswith("apk"):
          full_path_apk = apk_dir + file
          print("\n[I]\t", "Extracting data from", file)
          staticAnalyzer.run(full_path_apk, sys.argv[1])
      else:
          print("\n[I]\t", file, "not an APK...")
