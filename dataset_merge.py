# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:56:09 2018

@author: fatoks
"""
## this code read folder FINAL/videoXX/ to merge the files in one folder 
import os
import glob
import shutil
import xml.etree.ElementTree as ET
dir1=r'./annotations'#xml folder
dir2=r'./images'# jpg folder
i=1
dirList = os.listdir("./FINAL/") # current directory
for dir in dirList: #explore folders
    for xml_file in glob.glob('./FINAL/%s'%dir + '/*.xml'):
        file_xml=os.path.basename(xml_file)
        filename_xml, file_extension_xml = os.path.splitext(file_xml)     #name of XML file
        for image in glob.glob('./FINAL/%s'%dir+'/*.jpg'):
            file_img=os.path.basename(image)
            filename_img, file_extension_img = os.path.splitext(file_img)#name of jpg file

            if filename_xml==filename_img:#pairs
                new_name1 = os.path.join(dir1,file_xml)
                new_name2=os.path.join(dir2,file_img)
                if not os.path.exists(new_name1):
                    shutil.copy(xml_file, new_name1)
                    shutil.copy(image, new_name2)
                    tree=ET.parse(new_name1)#edit XML if duplicated name
                    root=tree.getroot()
                    new_tag=root.find('filename')
                    new_tag.text=os.path.splitext(os.path.basename(new_name1))[0]+'.jpg'
                    tree.write(new_name1)
                    #print('copied')
                    i+=1
                else:  # folder exists, file exists as well
                    ii = 1
                    while True:
                        new_name1 = os.path.join(dir1,filename_xml + "_" + str(ii) + file_extension_xml)
                        new_name2 = os.path.join(dir2,filename_img + "_" + str(ii) + file_extension_img)
                        
                        if not os.path.exists(new_name1):
                            shutil.copy(xml_file, new_name1)
                            shutil.copy(image, new_name2)
                            tree=ET.parse(new_name1)
                            root=tree.getroot()
                            new_tag=root.find('filename')
                            new_tag.text=os.path.splitext(os.path.basename(new_name1))[0]+'.jpg'
                            tree.write(new_name1)
                            #print "Copied", xml_file, "as", new_name1
                            break 
                       # else:
                            #print('image rename')
                        ii += 1
print('Finished')
