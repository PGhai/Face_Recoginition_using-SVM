import sys
import os
import bz2
from bz2 import decompress


path = "F:\\new"


def process_path(path):
	for(dirpath,dirnames,files)in os.walk(path):
		count=0
		for filename in files:
			if "fa" in filename and filename[-3:] =="bz2":
				print('fa found')
			elif "fb" in filename and filename[-3:] =="bz2":
				print('fb found')
			elif filename[-3:] =="png":
				print('png')
				continue
			else:
				os.system('del '+path+'\\'+filename)
				print(filename+ ' deleted')
				continue		
			count=count+1
			filepath = os.path.join(dirpath, filename)
			newfilepath = os.path.join(dirpath, filename + '.ppm')
			with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
				for data in iter(lambda : file.read(100 * 1024), b''):
					new_file.write(data)
			print('bgdhd'+newfilepath)		
			os.system('python process_face.py --shape-predictor shape_predictor_68_face_landmarks.dat --image '+newfilepath+' --path '+path)

for(dirpath,dirnames,files)in os.walk(path):
	if (len(dirnames)!=0):
		for directory in dirnames:
			print(directory)
			print(directory[-3:])
			#print (directory[-3:]=="est")
			process_path(path+'\\'+directory)
			for(dirpath,dirnames,files)in os.walk(path+'\\'+directory):
				for filename in files:
					if filename[-3:] =="png":
						print('png')
						continue
					else:
						os.system('del '+path+'\\'+directory+'\\'+filename)
						print(filename+ ' deleted')
						continue		